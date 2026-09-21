"""DOWNSIDE_V1 Pass 002 — integrity-corrected research harness.
Research only. No live model, no trading, no changes to BALANCED_FORWARD_V1.
"""
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd

INPUT=Path("/kaggle/working/v4_clean_research_dataset.parquet")
OUT=Path("/kaggle/working/downside_v1_pass002"); OUT.mkdir(parents=True,exist_ok=True)

df=pd.read_parquet(INPUT)
required={"Ticker","Date","Close","Eligible","IntegrityFirewall"}
missing=required-set(df.columns)
if missing: raise RuntimeError(f"Missing required columns: {sorted(missing)}")
df["Date"]=pd.to_datetime(df["Date"])
df=df.sort_values(["Ticker","Date"]).reset_index(drop=True)
if df.duplicated(["Ticker","Date"]).any(): raise RuntimeError("Duplicate ticker/date rows")
bad_price=df["Close"].notna() & (df["Close"]<=0)
bad_eligible=bad_price & df["Eligible"].fillna(False).astype(bool)
print(f"Global non-positive Close rows: {int(bad_price.sum()):,}; eligible: {int(bad_eligible.sum()):,}")
if bad_eligible.any():
    df.loc[bad_eligible,["Date","Ticker","Close","Eligible"]].to_csv(OUT/"eligible_bad_prices.csv",index=False)
    raise RuntimeError("FAIL CLOSED: non-positive Close reached eligible universe.")
# Invalid non-eligible history must not enter rolling features. Convert it to NaN so
# pct_change/rolling calculations cannot propagate impossible prices into later rows.
df.loc[bad_price,"Close"]=np.nan

g=df.groupby("Ticker",sort=False,group_keys=False)
# Explicit no-fill removes pandas pct_change ambiguity.
for k in [1,3,6,12]:
    df[f"ret{k}"]=g["Close"].pct_change(k,fill_method=None)
df["weak1"]=-df["ret1"]; df["weak3"]=-df["ret3"]; df["weak6"]=-df["ret6"]; df["weak12"]=-df["ret12"]
df["peak12"]=g["Close"].transform(lambda x:x.rolling(12,min_periods=6).max())
df["drawdown_depth"]=-(df["Close"]/df["peak12"]-1)
df["mret"]=g["Close"].pct_change(fill_method=None)
df["negret"]=df["mret"].where(df["mret"]<0,0.0)
df["downvol6"]=g["negret"].transform(lambda x:x.rolling(6,min_periods=3).std())
df["vol3"]=g["mret"].transform(lambda x:x.rolling(3,min_periods=3).std())
df["vol12"]=g["mret"].transform(lambda x:x.rolling(12,min_periods=6).std())
df["vol_expansion"]=df["vol3"]/df["vol12"].replace(0,np.nan)

# Exact next-row target is allowed only when it is the next calendar month.
df["next_date"]=g["Date"].shift(-1)
df["next_close"]=g["Close"].shift(-1)
df["next_eligible"]=g["Eligible"].shift(-1).fillna(False).astype(bool)
df["next_firewall"]=g["IntegrityFirewall"].shift(-1).fillna(True).astype(bool)
curp=df["Date"].dt.to_period("M"); nxtp=df["next_date"].dt.to_period("M")
df["consecutive_next_month"]=(nxtp-curp).apply(lambda x:(getattr(x,"n",None)==1) if pd.notna(x) else False)
df["fwd1"]=df["next_close"]/df["Close"]-1

features=["weak1","weak3","weak6","weak12","drawdown_depth","downvol6","vol_expansion"]
finite=np.isfinite(df[features]).all(axis=1)
# Enforce BOTH signal-month and forward-month eligibility/integrity.
mask=(df["Eligible"].fillna(False).astype(bool) & df["next_eligible"] &
      ~df["IntegrityFirewall"].fillna(True).astype(bool) & ~df["next_firewall"] &
      df["consecutive_next_month"] & finite & np.isfinite(df["fwd1"]))
use=df.loc[mask,["Date","Ticker","fwd1"]+features].copy()

# Fail closed on residual extreme monthly returns. Report before failing rather than silently winsorizing.
q=use["fwd1"].quantile([0,.0001,.001,.01,.5,.99,.999,.9999,1])
print("\nForward 1M return sanity quantiles:\n",q.to_string())
extreme=(use["fwd1"].abs()>=0.90)
print(f"Residual |1M return| >= 90%: {int(extreme.sum()):,} / {len(use):,}")
if extreme.any():
    sample=use.loc[extreme,["Date","Ticker","fwd1"]].sort_values("fwd1").head(20)
    sample.to_csv(OUT/"residual_extremes.csv",index=False)
    raise RuntimeError("FAIL CLOSED: residual >=90% monthly returns remain after eligibility/firewall. Inspect residual_extremes.csv.")

# Temporary market proxy retained only for Pass 002 because historical SPY total-return
# series is not present in this parquet. Promotion remains prohibited until explicit SPY.
use["market_fwd1"]=use.groupby("Date")["fwd1"].transform("median")
use["fwd_excess1"]=use["fwd1"]-use["market_fwd1"]
use["worst_decile"]=use.groupby("Date")["fwd_excess1"].transform(lambda x:x<=x.quantile(.10))

months=np.array(sorted(use["Date"].dt.to_period("M").unique()))
if len(months)<60: raise RuntimeError("Insufficient clean monthly history")
a=int(len(months)*.60); b=int(len(months)*.80)
train=set(months[:a]); val=set(months[a:b])
use["split"]=np.where(use["Date"].dt.to_period("M").isin(train),"train",
              np.where(use["Date"].dt.to_period("M").isin(val),"validation","test"))

def rank(s): return s.groupby(use["Date"]).rank(pct=True)
scores={"NEG_6M":rank(use["weak6"]),"NEG_12M":rank(use["weak12"]),
        "DRAWDOWN":rank(use["drawdown_depth"]),"DOWNVOL":rank(use["downvol6"])}
use["EQ_COMPOSITE"]=pd.concat([rank(use[x]) for x in features],axis=1).mean(axis=1)
scores["EQ_COMPOSITE"]=use["EQ_COMPOSITE"]

rows=[]
for split in ["validation","test"]:
    sm=use["split"].eq(split)
    for name,score in scores.items():
        x=use.loc[sm,["Date","fwd1","fwd_excess1","worst_decile"]].copy(); x["score"]=score.loc[sm]
        x["pred"]=x.groupby("Date")["score"].transform(lambda z:z>=z.quantile(.90))
        tp=int((x.pred & x.worst_decile).sum()); pp=int(x.pred.sum()); actual=int(x.worst_decile.sum())
        rows.append(dict(split=split,model=name,n=len(x),months=x["Date"].nunique(),
          precision_worst_decile=tp/pp if pp else np.nan,recall_worst_decile=tp/actual if actual else np.nan,
          bearish_basket_mean_fwd_return=x.loc[x.pred,"fwd1"].mean(),
          bearish_basket_mean_fwd_excess=x.loc[x.pred,"fwd_excess1"].mean(),
          universe_mean_fwd_return=x["fwd1"].mean(),
          median_fwd_return=x["fwd1"].median()))
results=pd.DataFrame(rows)
results.to_csv(OUT/"baseline_results.csv",index=False)
use.to_parquet(OUT/"research_frame.parquet",index=False)
manifest={"status":"RESEARCH_ONLY_AWAITING_SPY","model":"DOWNSIDE_V1_PASS_002",
 "input":str(INPUT),"rows":len(use),"months":len(months),"date_min":str(use.Date.min().date()),
 "date_max":str(use.Date.max().date()),"features":features,
 "target":"worst decile 1M forward excess return; temporary cross-sectional median benchmark",
 "integrity":["Eligible at t","Eligible at t+1","firewall clear at t and t+1","exact next calendar month","abs forward return <90% fail-closed"],
 "split":{"train":"first 60% months","validation":"next 20%","test":"final 20%"},
 "promotion_blocker":"Explicit adjusted SPY total-return benchmark still required.",
 "results_sha256":hashlib.sha256((OUT/"baseline_results.csv").read_bytes()).hexdigest()}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
print("\nPASS 002 RESULTS\n",results.to_string(index=False))
print("\nWrote",OUT)

"""DOWNSIDE_V1 Pass 003 — remove future eligibility conditioning + explicit SPY.
Research only. Frozen Pass-002 baseline scores; no tuning, no live model.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

INPUT=Path("/kaggle/working/v4_clean_research_dataset.parquet")
SPY_INPUT=Path("/kaggle/working/spy_total_return_monthly.csv")
OUT=Path("/kaggle/working/downside_v1_pass003"); OUT.mkdir(parents=True,exist_ok=True)

df=pd.read_parquet(INPUT)
required={"Ticker","Date","Close","Eligible","IntegrityFirewall"}
missing=required-set(df.columns)
if missing: raise RuntimeError(f"Missing required columns: {sorted(missing)}")
df["Date"]=pd.to_datetime(df["Date"])
df=df.sort_values(["Ticker","Date"]).reset_index(drop=True)
if df.duplicated(["Ticker","Date"]).any(): raise RuntimeError("Duplicate ticker/date rows")

bad_price=df["Close"].notna() & (df["Close"]<=0)
bad_eligible=bad_price & df["Eligible"].eq(True)
print(f"Global non-positive Close rows: {int(bad_price.sum()):,}; eligible: {int(bad_eligible.sum()):,}")
if bad_eligible.any():
    df.loc[bad_eligible,["Date","Ticker","Close","Eligible"]].to_csv(OUT/"eligible_bad_prices.csv",index=False)
    raise RuntimeError("FAIL CLOSED: non-positive Close reached eligible signal universe.")
df.loc[bad_price,"Close"]=np.nan

g=df.groupby("Ticker",sort=False,group_keys=False)
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

df["next_date"]=g["Date"].shift(-1)
df["next_close"]=g["Close"].shift(-1)
curp=df["Date"].dt.to_period("M"); nxtp=df["next_date"].dt.to_period("M")
df["consecutive_next_month"]=(nxtp-curp).apply(lambda x:(getattr(x,"n",None)==1) if pd.notna(x) else False)
df["fwd1"]=df["next_close"]/df["Close"]-1

features=["weak1","weak3","weak6","weak12","drawdown_depth","downvol6","vol_expansion"]
finite=np.isfinite(df[features]).all(axis=1)

# CRITICAL PASS-003 CHANGE:
# Selection is based ONLY on information available at signal month t.
# We do NOT require next-month eligibility or next-month firewall status.
signal_mask=(df["Eligible"].eq(True) & ~df["IntegrityFirewall"].eq(True) & finite)
signal=df.loc[signal_mask,["Date","Ticker","fwd1","next_date","next_close","consecutive_next_month"]+features].copy()
signal["target_observed"]=(signal["consecutive_next_month"] & np.isfinite(signal["next_close"]) & (signal["next_close"]>0) & np.isfinite(signal["fwd1"]))

coverage=signal.groupby("Date").agg(signals=("Ticker","size"),observed=("target_observed","sum")).reset_index()
coverage["missing"]=coverage["signals"]-coverage["observed"]
coverage["coverage"]=coverage["observed"]/coverage["signals"]
coverage.to_csv(OUT/"forward_target_coverage_by_month.csv",index=False)
missing=signal.loc[~signal["target_observed"],["Date","Ticker","next_date","next_close"]]
missing.to_csv(OUT/"missing_forward_targets.csv",index=False)
print(f"Signal observations at t: {len(signal):,}")
print(f"Observed next-month targets: {int(signal.target_observed.sum()):,}")
print(f"Missing/unobserved next-month targets: {int((~signal.target_observed).sum()):,}")
print(f"Overall target coverage: {signal.target_observed.mean():.4%}")
print("IMPORTANT: missing targets are audited, not treated as zero and not used to define eligibility.")

use=signal.loc[signal["target_observed"]].copy()
q=use["fwd1"].quantile([0,.0001,.001,.01,.5,.99,.999,.9999,1])
print("\nForward 1M return sanity quantiles:\n",q.to_string())
impossible=(use["fwd1"] < -1.0) | ~np.isfinite(use["fwd1"])
if impossible.any():
    use.loc[impossible,["Date","Ticker","fwd1"]].to_csv(OUT/"impossible_forward_returns.csv",index=False)
    raise RuntimeError("FAIL CLOSED: impossible/non-finite forward returns remain.")

if not SPY_INPUT.exists():
    raise RuntimeError(f"Missing {SPY_INPUT}. Run the Pass-003 launcher cell that creates adjusted SPY monthly data first.")
spy=pd.read_csv(SPY_INPUT)
spy["Date"]=pd.to_datetime(spy["Date"])
if not {"Date","SPY_Close"}.issubset(spy.columns): raise RuntimeError("SPY file must contain Date, SPY_Close")
spy=spy.sort_values("Date").drop_duplicates("Date")
spy["Month"]=spy["Date"].dt.to_period("M")
spy["SPY_fwd1"]=spy["SPY_Close"].shift(-1)/spy["SPY_Close"]-1
spy["SPY_next_month"]=spy["Month"].shift(-1)
spy["SPY_consecutive"]=(spy["SPY_next_month"]-spy["Month"]).apply(lambda x:(getattr(x,"n",None)==1) if pd.notna(x) else False)
spy.loc[~spy["SPY_consecutive"],"SPY_fwd1"]=np.nan
spy_map=spy.set_index("Month")["SPY_fwd1"]
use["Month"]=use["Date"].dt.to_period("M")
use["spy_fwd1"]=use["Month"].map(spy_map)
spy_missing=int(use["spy_fwd1"].isna().sum())
print(f"Rows missing explicit SPY forward return: {spy_missing:,}")
use=use.loc[np.isfinite(use["spy_fwd1"])].copy()
if use.empty: raise RuntimeError("No observations remain after SPY alignment.")
use["fwd_excess1"]=use["fwd1"]-use["spy_fwd1"]
use["worst_decile"]=use.groupby("Date")["fwd_excess1"].transform(lambda x:x<=x.quantile(.10))

months=np.array(sorted(use["Month"].unique()))
if len(months)<60: raise RuntimeError("Insufficient clean monthly history")
a=int(len(months)*.60); b=int(len(months)*.80)
train=set(months[:a]); val=set(months[a:b])
use["split"]=np.where(use["Month"].isin(train),"train",np.where(use["Month"].isin(val),"validation","test"))

def rank(s): return s.groupby(use["Date"]).rank(pct=True)
scores={"NEG_6M":rank(use["weak6"]),"NEG_12M":rank(use["weak12"]),
        "DRAWDOWN":rank(use["drawdown_depth"]),"DOWNVOL":rank(use["downvol6"])}
use["EQ_COMPOSITE"]=pd.concat([rank(use[x]) for x in features],axis=1).mean(axis=1)
scores["EQ_COMPOSITE"]=use["EQ_COMPOSITE"]

rows=[]
for split in ["validation","test"]:
    sm=use["split"].eq(split)
    for name,score in scores.items():
        x=use.loc[sm,["Date","fwd1","spy_fwd1","fwd_excess1","worst_decile"]].copy(); x["score"]=score.loc[sm]
        x["pred"]=x.groupby("Date")["score"].transform(lambda z:z>=z.quantile(.90))
        tp=int((x.pred & x.worst_decile).sum()); pp=int(x.pred.sum()); actual=int(x.worst_decile.sum())
        rows.append(dict(split=split,model=name,n=len(x),months=x["Date"].nunique(),
          precision_worst_decile=tp/pp if pp else np.nan,recall_worst_decile=tp/actual if actual else np.nan,
          bearish_basket_mean_fwd_return=x.loc[x.pred,"fwd1"].mean(),
          bearish_basket_mean_spy_return=x.loc[x.pred,"spy_fwd1"].mean(),
          bearish_basket_mean_fwd_excess=x.loc[x.pred,"fwd_excess1"].mean(),
          universe_mean_fwd_return=x["fwd1"].mean(),median_fwd_return=x["fwd1"].median()))
results=pd.DataFrame(rows)
results.to_csv(OUT/"baseline_results.csv",index=False)
use.to_parquet(OUT/"research_frame.parquet",index=False)

manifest={"status":"RESEARCH_ONLY_PASS003","model":"DOWNSIDE_V1_PASS_003",
 "input":str(INPUT),"spy_input":str(SPY_INPUT),"rows":len(use),"months":len(months),
 "date_min":str(use.Date.min().date()),"date_max":str(use.Date.max().date()),"features":features,
 "target":"worst decile 1M forward excess return versus explicit adjusted SPY",
 "integrity":["Eligibility only at signal month t","signal-month firewall only","exact next calendar month required for observed target","missing forward targets audited separately","no next-month eligibility conditioning"],
 "limitation":"Missing/delisted forward outcomes remain unresolved rather than imputed. Results are conditional on an observable next-month price; promotion prohibited until delisting/terminal-return treatment is specified.",
 "split":{"train":"first 60% months","validation":"next 20%","test":"final 20%"},
 "results_sha256":hashlib.sha256((OUT/"baseline_results.csv").read_bytes()).hexdigest()}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
print("\nPASS 003 RESULTS\n",results.to_string(index=False))
print("\nWrote",OUT)

"""DOWNSIDE_V1 research harness.

Research only. Does not alter BALANCED_FORWARD_V1 or Snapshot #001.
Expected input is the integrity-clean monthly research parquet created by V4.
The script discovers common column aliases, builds lagged downside features, creates
forward excess-return labels, evaluates simple baselines plus a logistic model with
chronological splits, and writes auditable CSV/JSON outputs.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

INPUT=Path("/kaggle/working/v4_clean_research_dataset.parquet")
OUT=Path("/kaggle/working/downside_v1")
OUT.mkdir(parents=True,exist_ok=True)

def pick(df,*names):
    m={c.lower():c for c in df.columns}
    for n in names:
        if n.lower() in m:return m[n.lower()]
    raise KeyError(f"Missing one of {names}. Available: {list(df.columns)}")

df=pd.read_parquet(INPUT)
ticker=pick(df,"Ticker","ticker")
date=pick(df,"Date","Month","month","MonthEnd")
close=pick(df,"Close","close","Price","price")
df[date]=pd.to_datetime(df[date])
df=df.sort_values([ticker,date]).copy()
g=df.groupby(ticker,group_keys=False)

# Features use information available at month t only.
for k in [1,3,6,12]:
    df[f"ret{k}"]=g[close].pct_change(k)
df["weak1"]=-df["ret1"]; df["weak3"]=-df["ret3"]
df["weak6"]=-df["ret6"]; df["weak12"]=-df["ret12"]
df["peak12"]=g[close].transform(lambda x:x.rolling(12,min_periods=6).max())
df["drawdown12"]=df[close]/df["peak12"]-1
df["drawdown_depth"]=-df["drawdown12"]

# Monthly downside-volatility and volatility-expansion proxies.
df["mret"]=g[close].pct_change()
df["negret"]=df["mret"].where(df["mret"]<0,0.0)
df["downvol6"]=g["negret"].transform(lambda x:x.rolling(6,min_periods=3).std())
df["vol3"]=g["mret"].transform(lambda x:x.rolling(3,min_periods=3).std())
df["vol12"]=g["mret"].transform(lambda x:x.rolling(12,min_periods=6).std())
df["vol_expansion"]=df["vol3"]/df["vol12"].replace(0,np.nan)

# Cross-sectional SPY proxy: median eligible-stock return. This avoids assuming SPY
# exists in the cleaned security universe; replace with an explicit SPY total-return
# series in the next research pass when available.
market=df.groupby(date)["mret"].median().rename("market_ret")
df=df.join(market,on=date)
df["fwd1"]=g[close].shift(-1)/df[close]-1
df["market_fwd1"]=df.groupby(date)["fwd1"].transform("median")
df["fwd_excess1"]=df["fwd1"]-df["market_fwd1"]
df["worst_decile"]=df.groupby(date)["fwd_excess1"].transform(lambda x:x<=x.quantile(.10))

features=["weak1","weak3","weak6","weak12","drawdown_depth","downvol6","vol_expansion"]
use=df.dropna(subset=features+["fwd_excess1"]).copy()
months=np.array(sorted(use[date].dt.to_period("M").unique()))
if len(months)<60: raise RuntimeError("Insufficient monthly history")
a=int(len(months)*.60); b=int(len(months)*.80)
train_m=set(months[:a]); val_m=set(months[a:b]); test_m=set(months[b:])
use["split"]=np.where(use[date].dt.to_period("M").isin(train_m),"train",
              np.where(use[date].dt.to_period("M").isin(val_m),"validation","test"))

def rank_score(s): return s.groupby(use[date]).rank(pct=True)
scores={
 "NEG_6M":rank_score(use["weak6"]),
 "NEG_12M":rank_score(use["weak12"]),
 "DRAWDOWN":rank_score(use["drawdown_depth"]),
 "DOWNVOL":rank_score(use["downvol6"])
}
# Fixed equal-weight candidate; no optimization on validation/test.
use["EQ_COMPOSITE"]=pd.concat([rank_score(use[x]) for x in features],axis=1).mean(axis=1)
scores["EQ_COMPOSITE"]=use["EQ_COMPOSITE"]

rows=[]
for split in ["validation","test"]:
    mask=use["split"].eq(split)
    for name,score in scores.items():
        sub=use.loc[mask,[date,"fwd_excess1","worst_decile"]].copy()
        sub["score"]=score.loc[mask]
        sub["pred"]=sub.groupby(date)["score"].transform(lambda x:x>=x.quantile(.90))
        tp=(sub.pred & sub.worst_decile).sum(); pp=sub.pred.sum(); actual=sub.worst_decile.sum()
        rows.append({"split":split,"model":name,"n":len(sub),
          "precision_worst_decile":float(tp/pp) if pp else None,
          "recall_worst_decile":float(tp/actual) if actual else None,
          "bearish_basket_mean_fwd_excess":float(sub.loc[sub.pred,"fwd_excess1"].mean()),
          "all_mean_fwd_excess":float(sub["fwd_excess1"].mean())})
results=pd.DataFrame(rows)
results.to_csv(OUT/"baseline_results.csv",index=False)
use[[date,ticker,"split","fwd_excess1","worst_decile"]+features+["EQ_COMPOSITE"]].to_parquet(OUT/"research_frame.parquet",index=False)
manifest={"status":"RESEARCH_ONLY","model":"DOWNSIDE_V1","input":str(INPUT),
 "features":features,"target":"worst decile 1M forward excess return",
 "split":{"train":"first 60% months","validation":"next 20%","test":"final 20%"},
 "note":"Market excess proxy is cross-sectional median for this first harness; explicit SPY adjusted total-return benchmark required before promotion.",
 "results_sha256":hashlib.sha256((OUT/"baseline_results.csv").read_bytes()).hexdigest()}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
print(results.to_string(index=False))
print("\nWrote:",OUT)

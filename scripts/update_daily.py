import json, datetime, yfinance as yf
from pathlib import Path

SNAPSHOT=Path("data/snapshot.json")
ENTRY=Path("data/forward_entry_001.json")
LATEST=Path("data/latest.json")
ENTRY_DATE="2026-09-21"

s=json.loads(SNAPSHOT.read_text())
tickers=[x["t"] for x in s["positions"]]
all_t=tickers+["SPY"]
d=yf.download(all_t,period="10d",interval="1d",auto_adjust=True,progress=False,threads=True)

def series(field,t):
    try: return d[field][t].dropna()
    except Exception: return None

# Immutable prospective clock: first market session after the Sep 20 lock.
if not ENTRY.exists():
    entries=[]
    missing=[]
    for t in tickers:
        try:
            z=d["Open"][t].dropna()
            row=z[z.index.strftime("%Y-%m-%d")==ENTRY_DATE]
            if row.empty: raise ValueError("entry date missing")
            entries.append({"t":t,"entry":float(row.iloc[0])})
        except Exception:
            entries.append({"t":t,"entry":None}); missing.append(t)
    try:
        z=d["Open"]["SPY"].dropna(); row=z[z.index.strftime("%Y-%m-%d")==ENTRY_DATE]
        spy_entry=float(row.iloc[0]) if not row.empty else None
    except Exception: spy_entry=None
    if missing or spy_entry is None:
        raise RuntimeError("Fail closed: incomplete forward-entry coverage: "+",".join(missing))
    ENTRY.write_text(json.dumps({
      "snapshot":"001","model":s["model"],"entryDate":ENTRY_DATE,
      "entryBasis":"Adjusted market open; first trading session after snapshot lock",
      "positions":entries,"spyEntry":spy_entry,"immutable":True
    },indent=2))

e=json.loads(ENTRY.read_text())
emap={x["t"]:x["entry"] for x in e["positions"]}
out=[]; missing=[]
for x in s["positions"]:
    t=x["t"]
    try:
        z=d["Close"][t].dropna()
        close=float(z.iloc[-1])
        day=float(close/z.iloc[-2]-1) if len(z)>1 else None
        entry=emap[t]
        prospective=float(close/entry-1)
        diagnostic=float(close/x["p"]-1)
        out.append({"t":t,"close":close,"day":day,"sinceEntry":prospective,"sinceSignalDiagnostic":diagnostic})
    except Exception:
        missing.append(t)
        out.append({"t":t,"close":None,"day":None,"sinceEntry":None,"sinceSignalDiagnostic":None})

coverage=sum(x["close"] is not None for x in out)
if coverage != len(tickers):
    raise RuntimeError(f"Fail closed: market coverage {coverage}/{len(tickers)}; missing={missing}")

spy_close=float(d["Close"]["SPY"].dropna().iloc[-1])
spy_since_entry=float(spy_close/e["spyEntry"]-1)
portfolio_since_entry=sum(x["sinceEntry"]*.02 for x in out)
portfolio_day=sum(x["day"]*.02 for x in out if x["day"] is not None)
LATEST.write_text(json.dumps({
 "updated":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "entryDate":e["entryDate"],"entryBasis":e["entryBasis"],
 "coverage":coverage,"expectedCoverage":len(tickers),
 "portfolioDay":portfolio_day,"portfolioSinceEntry":portfolio_since_entry,
 "spySinceEntry":spy_since_entry,"excessSinceEntry":portfolio_since_entry-spy_since_entry,
 "positions":out
},indent=2))

import json,datetime,yfinance as yf
from pathlib import Path
s=json.loads(Path('data/snapshot.json').read_text())
t=[x['t'] for x in s['positions']]
all_t=t+['SPY']
d=yf.download(all_t,period='5d',interval='1d',auto_adjust=True,progress=False,threads=True)
out=[]
for x in s['positions']:
 try:
  z=d['Close'][x['t']].dropna(); close=float(z.iloc[-1]); day=float(close/z.iloc[-2]-1) if len(z)>1 else None
  out.append({'t':x['t'],'close':close,'day':day})
 except Exception: out.append({'t':x['t'],'close':None,'day':None})
spy=None
try:
 z=d['Close']['SPY'].dropna(); spy=float(z.iloc[-1]/z.iloc[0]-1) if len(z)>1 else None
except Exception: pass
Path('data/latest.json').write_text(json.dumps({'updated':datetime.datetime.now(datetime.timezone.utc).isoformat(),'positions':out,'spySince':spy},indent=2))

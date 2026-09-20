import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

PRACTICE_URL = "https://api-fxpractice.oanda.com"
LIVE_URL = "https://api-fxtrade.oanda.com"
GRANULARITIES = ["M1","M2","M4","M5","M10","M15","M30","H1","H2","H3","H4","H6","H8","H12","D","W"]
TF_SECONDS = {"M1":60,"M2":120,"M4":240,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H2":7200,"H3":10800,"H4":14400,"H6":21600,"H8":28800,"H12":43200,"D":86400,"W":604800}


def api_base(env):
    return LIVE_URL if env.lower().startswith("live") else PRACTICE_URL


def oanda_get(path, token, env, params=None):
    r = requests.get(api_base(env)+path, headers={"Authorization":f"Bearer {token}","Accept":"application/json"}, params=params, timeout=30)
    if not r.ok:
        try: detail=r.json()
        except Exception: detail=r.text
        raise RuntimeError(f"OANDA API {r.status_code}: {detail}")
    return r.json()


def credentials():
    token=os.getenv("OANDA_TOKEN",""); env=os.getenv("OANDA_ENVIRONMENT","Practice")
    try:
        token=st.secrets.get("OANDA_TOKEN",token); env=st.secrets.get("OANDA_ENVIRONMENT",env)
    except Exception: pass
    return token,env

@st.cache_data(ttl=300, show_spinner=False)
def accounts(token,env):
    ids=tuple(x.get("id") for x in oanda_get("/v3/accounts",token,env).get("accounts",[]) if x.get("id"))
    if not ids: raise RuntimeError("No OANDA accounts are authorized for this token.")
    return ids

@st.cache_data(ttl=3600, show_spinner=False)
def instruments(token,account,env):
    rows=[]
    for x in oanda_get(f"/v3/accounts/{account}/instruments",token,env).get("instruments",[]):
        if x.get("type")=="CURRENCY":
            rows.append({"name":x["name"],"pipLocation":int(x.get("pipLocation",-4)),"displayPrecision":int(x.get("displayPrecision",5))})
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)

@st.cache_data(ttl=10, show_spinner=False)
def candles(token,account,env,symbol,tf,count,daily_alignment,alignment_timezone,weekly_alignment):
    p={"price":"M","granularity":tf,"count":min(int(count),5000),"smooth":"false","dailyAlignment":daily_alignment,"alignmentTimezone":alignment_timezone,"weeklyAlignment":weekly_alignment}
    d=oanda_get(f"/v3/accounts/{account}/instruments/{symbol}/candles",token,env,p)
    rows=[]
    for x in d.get("candles",[]):
        m=x.get("mid")
        if m: rows.append((pd.to_datetime(x["time"],utc=True),float(m["o"]),float(m["h"]),float(m["l"]),float(m["c"]),bool(x.get("complete",False))))
    if not rows: raise RuntimeError("No candles returned.")
    return pd.DataFrame(rows,columns=["time","open","high","low","close","complete"]).sort_values("time").reset_index(drop=True)


def strat_states(df,mintick):
    o=df.open.to_numpy(float); h=df.high.to_numpy(float); l=df.low.to_numpy(float); c=df.close.to_numpy(float); n=len(df)
    out=np.full(n,"N/A",dtype=object)
    if n<2:return out
    ph=h[:-1]; pl=l[:-1]; ch=h[1:]; cl=l[1:]; co=o[1:]; cc=c[1:]
    inside=(ch<=ph)&(cl>=pl); outside=(ch>ph)&(cl<pl); up=(ch>ph)&~outside; down=(cl<pl)&~outside
    green=cc>=co; body=np.abs(cc-co); eff=np.maximum(body,mintick)
    upper=ch-np.maximum(co,cc); lower=np.minimum(co,cc)-cl
    hammer=(lower>=2*eff)&(lower>upper); shooter=(upper>=2*eff)&(upper>lower)
    s=np.full(n-1,"N/A",dtype=object); s[inside]="1"
    normal=outside&~hammer&~shooter; s[normal]=np.where(green[normal],"3G","3R"); s[outside&hammer]="3-H"; s[outside&shooter]="3-SS"
    normal=up&~hammer&~shooter; s[normal]=np.where(green[normal],"2UG","2UR"); s[up&hammer]="2-H"; s[up&shooter]="2-SS"
    normal=down&~hammer&~shooter; s[normal]=np.where(green[normal],"2DG","2DR"); s[down&hammer]="2-H"; s[down&shooter]="2-SS"
    out[1:]=s; return out


def evaluate_from_c(df, i):
    """A=i-2, B=i-1, C=i. First B-side break is entry; opposite B side SL; A same-side extreme TP.
    Uses selected-TF OHLC. Returns Ambiguous whenever intrabar ordering cannot be known exactly.
    """
    a=df.iloc[i-2]; b=df.iloc[i-1]; c=df.iloc[i]
    long_valid=a.high>b.high
    short_valid=a.low<b.low
    up=c.high>b.high; dn=c.low<b.low
    if not up and not dn: return "No trigger",None,None
    if up and dn: return "Ambiguous entry",None,None
    direction="Long" if up else "Short"
    if direction=="Long" and not long_valid:return "Invalid target",direction,None
    if direction=="Short" and not short_valid:return "Invalid target",direction,None
    entry=b.high if direction=="Long" else b.low
    sl=b.low if direction=="Long" else b.high
    tp=a.high if direction=="Long" else a.low
    # Entry occurs during C. From that point, same C may hit TP or SL.
    # If both are present in C range, ordering is unknown at selected TF.
    if direction=="Long":
        hit_tp=c.high>=tp; hit_sl=c.low<=sl
    else:
        hit_tp=c.low<=tp; hit_sl=c.high>=sl
    if hit_tp and hit_sl:return "Ambiguous outcome",direction,(entry,sl,tp)
    if hit_tp:return "Win",direction,(entry,sl,tp)
    if hit_sl:return "Loss",direction,(entry,sl,tp)
    for j in range(i+1,len(df)):
        x=df.iloc[j]
        if direction=="Long": ht=x.high>=tp; hs=x.low<=sl
        else: ht=x.low<=tp; hs=x.high>=sl
        if ht and hs:return "Ambiguous outcome",direction,(entry,sl,tp)
        if ht:return "Win",direction,(entry,sl,tp)
        if hs:return "Loss",direction,(entry,sl,tp)
    return "Open",direction,(entry,sl,tp)


def discover_symbol(df,symbol,mintick):
    df=df[df.complete].copy().reset_index(drop=True)
    states=strat_states(df,mintick)
    rows=[]
    for i in range(2,len(df)):
        A=str(states[i-2]); B=str(states[i-1]); C=str(states[i])
        if "N/A" in (A,B,C):continue
        outcome,direction,levels=evaluate_from_c(df,i)
        rows.append({"Symbol":symbol,"A":A,"B":B,"C":C,"Setup":f"{A} ➔ {B}","Triggered C":C,"Direction":direction or "—","Outcome":outcome,"Candle C Time":df.time.iloc[i]})
    return pd.DataFrame(rows)


def aggregate(raw,min_trades):
    t=raw[raw.Outcome.isin(["Win","Loss","Ambiguous outcome"])].copy()
    if t.empty:return pd.DataFrame()
    g=t.groupby(["Setup","Direction"],dropna=False)
    out=g.Outcome.agg(Trades="count",Wins=lambda s:(s=="Win").sum(),Losses=lambda s:(s=="Loss").sum(),Ambiguous=lambda s:(s=="Ambiguous outcome").sum()).reset_index()
    out["Resolved"]=out.Wins+out.Losses
    out["Success Rate %"]=np.where(out.Resolved>0,out.Wins/out.Resolved*100,np.nan)
    out=out[out.Resolved>=min_trades].sort_values(["Success Rate %","Resolved"],ascending=[False,False]).reset_index(drop=True)
    return out


def main():
    st.set_page_config(page_title="The Strat Actionable Pattern Discovery",layout="wide")
    st.title("The Strat — Actionable Pattern Discovery")
    st.caption("A and B define the setup. C is actionable: first break of B enters; opposite side of B is SL; same-side extreme of A is TP.")
    token0,env0=credentials()
    with st.sidebar:
        env=st.selectbox("Environment",["Practice","Live"],index=0 if not str(env0).lower().startswith("live") else 1)
        token=st.text_input("OANDA API Token",value=token0,type="password")
        if not token: st.info("Enter OANDA API token."); st.stop()
        try: accs=accounts(token,env)
        except Exception as e: st.error(str(e)); st.stop()
        account=st.selectbox("Account",accs)
        try: inst=instruments(token,account,env)
        except Exception as e: st.error(str(e)); st.stop()
        tf=st.selectbox("Selected Timeframe",GRANULARITIES,index=GRANULARITIES.index("H1"))
        n=st.number_input("Selected Timeframe Bars",500,5000,5000,100)
        scan_all=st.checkbox("Scan all OANDA Forex symbols",False)
        default=["EUR_USD"] if "EUR_USD" in inst.name.values else [inst.name.iloc[0]]
        syms=list(inst.name) if scan_all else st.multiselect("Symbols",list(inst.name),default=default)
        min_trades=st.number_input("Minimum resolved trades per setup",1,500,30)
        workers=st.slider("Parallel symbols",1,30,10)
        st.divider(); st.caption("OANDA candle alignment")
        daily_alignment=st.number_input("Daily alignment hour",0,23,17)
        alignment_timezone=st.text_input("Alignment timezone","America/New_York")
        weekly_alignment=st.selectbox("Weekly alignment",["Friday","Saturday","Sunday","Monday"],index=0)
        run=st.button("Discover actionable patterns",type="primary",width="stretch")
    st.info("Intrabar ordering is never guessed. If one selected-timeframe candle contains both possible entry sides, or both TP and SL after entry, that occurrence is marked ambiguous. A lower-timeframe resolver can be added next without changing the setup statistics model.")
    if not run:return
    if not syms:st.warning("Select at least one symbol.");return
    meta=inst.set_index("name").to_dict("index")
    raws=[]; errors=[]
    def job(s):
        d=candles(token,account,env,s,tf,int(n),int(daily_alignment),alignment_timezone,weekly_alignment)
        mintick=10.0**int(meta[s]["pipLocation"])
        return discover_symbol(d,s,mintick)
    prog=st.progress(0); status=st.empty()
    with ThreadPoolExecutor(max_workers=min(int(workers),len(syms))) as ex:
        fut={ex.submit(job,s):s for s in syms}
        for k,f in enumerate(as_completed(fut),1):
            s=fut[f]
            try: raws.append(f.result())
            except Exception as e: errors.append({"Symbol":s,"Error":str(e)})
            prog.progress(k/len(syms));status.text(f"Processed {k}/{len(syms)} symbols")
    prog.empty();status.empty()
    if not raws:st.error("No results.");return
    raw=pd.concat(raws,ignore_index=True)
    stats=aggregate(raw,int(min_trades))
    c1,c2,c3,c4=st.columns(4)
    resolved=int(raw.Outcome.isin(["Win","Loss"]).sum()); wins=int((raw.Outcome=="Win").sum())
    c1.metric("Symbols",raw.Symbol.nunique());c2.metric("Setups observed",raw.Setup.nunique());c3.metric("Resolved trades",resolved);c4.metric("Overall win rate",f"{wins/resolved*100:.2f}%" if resolved else "N/A")
    st.subheader("Highest-success actionable A → B patterns")
    if stats.empty: st.warning("No setup meets the minimum resolved-trades filter.")
    else: st.dataframe(stats,width="stretch",hide_index=True,height=min(900,38*(len(stats)+1)))
    st.caption("Success Rate = Wins / (Wins + Losses). Ambiguous and still-open occurrences are excluded from the denominator. C's final Strat vocabulary is recorded for analysis but is not required to enter the trade.")
    with st.expander("Breakdown by symbol"):
        x=raw[raw.Outcome.isin(["Win","Loss"])].groupby(["Symbol","Setup","Direction"]).Outcome.agg(Trades="count",Wins=lambda s:(s=="Win").sum()).reset_index()
        if not x.empty:
            x["Success Rate %"]=x.Wins/x.Trades*100
            x=x[x.Trades>=int(min_trades)].sort_values(["Success Rate %","Trades"],ascending=[False,False])
        st.dataframe(x,width="stretch",hide_index=True)
    with st.expander("Raw historical occurrences"):
        st.dataframe(raw.sort_values("Candle C Time",ascending=False),width="stretch",hide_index=True,height=700)
    if errors:
        with st.expander(f"Errors ({len(errors)})"):st.dataframe(pd.DataFrame(errors),width="stretch",hide_index=True)

if __name__=="__main__":main()

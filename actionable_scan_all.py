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


def evaluate_bar_ohlc(b_high, b_low, a_high, a_low, c_high, c_low):
    """Same selected-timeframe entry/outcome rule used by the scanner."""
    up = c_high > b_high
    dn = c_low < b_low
    if not up and not dn:
        return 0, 0
    if up and dn:
        return 0, 2

    direction = 1 if up else -1
    valid = a_high > b_high if direction == 1 else a_low < b_low
    if not valid:
        return direction, 4

    sl = b_low if direction == 1 else b_high
    tp = a_high if direction == 1 else a_low
    hit_tp = c_high >= tp if direction == 1 else c_low <= tp
    hit_sl = c_low <= sl if direction == 1 else c_high >= sl

    # Conservative scanner rule: SL has priority inside one selected-TF bar.
    if hit_sl:
        return direction, -1
    if hit_tp:
        return direction, 1
    return direction, 3


def resolve_active_ohlc(direction, sl, tp, h, l):
    """Resolve a previously opened trade exactly like the scanner."""
    hit_tp = h >= tp if direction == 1 else l <= tp
    hit_sl = l <= sl if direction == 1 else h >= sl
    if hit_sl:
        return -1
    if hit_tp:
        return 1
    return 0


def discover_symbol(df, symbol, mintick, tf, time_bucket_filter=False):
    """Replay all occurrences using scanner counting; optionally filter B by time bucket."""
    df = df[df.complete].copy().reset_index(drop=True)
    states = strat_states(df, mintick)
    n = len(df)
    if n < 4:
        return pd.DataFrame()

    highs = df.high.to_numpy(float)
    lows = df.low.to_numpy(float)
    active = []
    rows = []

    latest_b_time = pd.Timestamp(df.time.iloc[-1])
    bucket_label = "All"
    target_weekday = None
    target_hour = None
    if time_bucket_filter and tf == "D":
        target_weekday = latest_b_time.dayofweek
        bucket_label = latest_b_time.day_name()
    elif time_bucket_filter and tf in ("H4", "H8"):
        target_hour = latest_b_time.hour
        bucket_label = f"{target_hour:02d}:00 UTC"

    for c_idx in range(2, n):
        # Resolve every previously opened trade first.
        keep = []
        for tr in active:
            result = resolve_active_ohlc(
                tr["dir"], tr["sl"], tr["tp"], highs[c_idx], lows[c_idx]
            )
            if result == 1:
                tr["row"]["Outcome"] = "Win"
            elif result == -1:
                tr["row"]["Outcome"] = "Loss"
            else:
                keep.append(tr)
        active = keep

        # Then allow this same candle to create another A->B->C occurrence.
        a_idx, b_idx = c_idx - 2, c_idx - 1
        A = str(states[a_idx])
        B = str(states[b_idx])
        C = str(states[c_idx])
        if "N/A" in (A, B, C):
            continue

        if time_bucket_filter:
            b_time = pd.Timestamp(df.time.iloc[b_idx])
            if tf == "D" and b_time.dayofweek != target_weekday:
                continue
            if tf in ("H4", "H8") and b_time.hour != target_hour:
                continue

        direction, code = evaluate_bar_ohlc(
            highs[b_idx], lows[b_idx], highs[a_idx], lows[a_idx],
            highs[c_idx], lows[c_idx]
        )

        side = "Long" if direction == 1 else ("Short" if direction == -1 else "—")
        outcome = {
            0: "No trigger",
            1: "Win",
            -1: "Loss",
            2: "Ambiguous entry",
            3: "Open",
            4: "Invalid target",
        }[code]

        row = {
            "Symbol": symbol,
            "A": A,
            "B": B,
            "C": C,
            "Setup": f"{A} ➔ {B}",
            "Triggered C": C,
            "Direction": side,
            "Outcome": outcome,
            "Candle C Time": df.time.iloc[c_idx],
            "Time Bucket": bucket_label,
        }
        rows.append(row)

        # Track every open occurrence independently. Multiple simultaneous
        # trades, including the same setup/direction, are allowed.
        if code == 3:
            active.append({
                "dir": direction,
                "sl": lows[b_idx] if direction == 1 else highs[b_idx],
                "tp": highs[a_idx] if direction == 1 else lows[a_idx],
                "row": row,
            })

    return pd.DataFrame(rows)


def aggregate(raw,min_trades):
    t=raw[raw.Outcome.isin(["Win","Loss"])].copy()
    if t.empty:return pd.DataFrame()
    g=t.groupby(["Setup","Direction"],dropna=False)
    out=g.Outcome.agg(Trades="count",Wins=lambda s:(s=="Win").sum(),Losses=lambda s:(s=="Loss").sum()).reset_index()
    out["Resolved"]=out.Wins+out.Losses
    out["Success Rate %"]=np.where(out.Resolved>0,out.Wins/out.Resolved*100,np.nan)
    out=out[out.Resolved>=min_trades].sort_values(["Success Rate %","Resolved"],ascending=[False,False]).reset_index(drop=True)
    return out


def main():
    st.set_page_config(page_title="The Strat Actionable Pattern Discovery",layout="wide")
    st.title("The Strat — Actionable Pattern Discovery")
    st.caption("A and B define the setup. C is actionable: a one-sided break of B enters; opposite side of B is SL; same-side extreme of A is TP. Historical outcomes use the scanner active-trade replay method.")
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
        time_bucket_filter=st.checkbox(
            "Filter historical setups by current B time bucket", False,
            help="D: same weekday as current B. H4/H8: same UTC candle-start hour as current B. Other timeframes are unchanged."
        )
        workers=st.slider("Parallel symbols",1,30,10)
        st.divider(); st.caption("OANDA candle alignment")
        daily_alignment=st.number_input("Daily alignment hour",0,23,17)
        alignment_timezone=st.text_input("Alignment timezone","America/New_York")
        weekly_alignment=st.selectbox("Weekly alignment",["Friday","Saturday","Sunday","Monday"],index=0)
        run=st.button("Discover actionable patterns",type="primary",width="stretch")
    st.info("Historical counting now matches the scanner: all triggered occurrences are tracked independently, including multiple simultaneous open trades. Each new candle resolves all prior active trades before evaluating a new C. A C that breaks both sides of B is non-directional and excluded. SL has priority when TP and SL occur in the same selected-timeframe candle.")
    if not run:return
    if not syms:st.warning("Select at least one symbol.");return
    meta=inst.set_index("name").to_dict("index")
    raws=[]; errors=[]
    def job(s):
        d=candles(token,account,env,s,tf,int(n),int(daily_alignment),alignment_timezone,weekly_alignment)
        mintick=10.0**int(meta[s]["pipLocation"])
        return discover_symbol(d,s,mintick,tf,time_bucket_filter)
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
    st.caption("Success Rate = Wins / (Wins + Losses). Once direction is established, a bar touching SL is a Loss even if it also touches TP. Ambiguous-entry and still-open occurrences are excluded from the denominator. C's final Strat vocabulary is recorded for analysis but is not required to enter the trade.")
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

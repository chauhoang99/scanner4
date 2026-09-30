import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="OANDA Range Breakout Scanner", layout="wide")

OANDA_GRANULARITY = {
    "M15": "M15",
    "H1": "H1",
    "H4": "H4",
    "H8": "H8",
    "D1": "D",
    "W1": "W",
    "MN": "M",
}

DEFAULT_INSTRUMENTS = [
    "EUR_USD","GBP_USD","USD_JPY","AUD_USD","USD_CAD","NZD_USD",
    "EUR_JPY","GBP_JPY","USD_CHF","EUR_GBP","AUD_JPY","CAD_JPY",
    "SGD_JPY","USD_SGD","EUR_SGD","GBP_SGD","AUD_SGD","CAD_SGD",
    "XAU_USD","XAG_USD"
]

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

def oanda_host(environment):
    return (
        "https://api-fxpractice.oanda.com"
        if environment == "practice"
        else "https://api-fxtrade.oanda.com"
    )

@st.cache_data(ttl=3600, show_spinner=False)
def discover_account_id(token, environment):
    r = requests.get(
        f"{oanda_host(environment)}/v3/accounts",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"OANDA {r.status_code}: {r.text[:500]}")
    accounts = r.json().get("accounts", [])
    if not accounts:
        raise RuntimeError("No OANDA accounts are authorized for this API token.")
    return accounts[0]["id"]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_oanda_instruments(token, environment, account_id):
    r = requests.get(
        f"{oanda_host(environment)}/v3/accounts/{account_id}/instruments",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"OANDA {r.status_code}: {r.text[:500]}")
    items = r.json().get("instruments", [])
    items.sort(key=lambda x: (x.get("type", ""), x.get("displayName", x.get("name", ""))))
    return items

def fetch_candles(token, environment, instrument, granularity, count=80):
    url = f"{oanda_host(environment)}/v3/instruments/{instrument}/candles"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params={
            "price": "M",
            "granularity": granularity,
            "count": min(int(count), 5000),
        },
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"{instrument}: OANDA {r.status_code}: {r.text[:250]}")

    rows = []
    for c in r.json().get("candles", []):
        if not c.get("complete", False):
            continue
        m = c["mid"]
        rows.append({
            "time": pd.to_datetime(c["time"], utc=True),
            "open": float(m["o"]),
            "high": float(m["h"]),
            "low": float(m["l"]),
            "close": float(m["c"]),
            "volume": int(c.get("volume", 0)),
        })

    df = pd.DataFrame(rows)
    if len(df) < 16:
        raise RuntimeError(f"{instrument}: not enough completed candles.")
    return df.reset_index(drop=True)

def detect_patterns(df, enabled):
    """Same OB / EB / OEB definitions and precedence as the supplied dashboard."""
    out = []
    for i in range(1, len(df)):
        c, p = df.iloc[i], df.iloc[i - 1]

        is_ob = c.high > p.high and c.low < p.low

        curr_bull = c.close > c.open
        curr_bear = c.close < c.open
        prev_bull = p.close > p.open
        prev_bear = p.close < p.open

        bull_eb = curr_bull and prev_bear and c.close >= p.open and c.open <= p.close
        bear_eb = curr_bear and prev_bull and c.close <= p.open and c.open >= p.close
        is_eb = bull_eb or bear_eb
        is_oeb = is_ob and is_eb

        typ = None
        if is_oeb and "OEB" in enabled:
            typ = "OEB"
        elif is_ob and "OB" in enabled:
            typ = "OB"
        elif is_eb and "EB" in enabled:
            typ = "EB"

        if typ:
            out.append({
                "barIndex": i,
                "type": typ,
                "high": float(c.high),
                "low": float(c.low),
            })
    return out

def find_recent_breakout(df, enabled, forward_window, recent_bars):
    """
    Find the newest setup-range breakout whose breakout candle is in the
    latest `recent_bars` completed candles.

    Breakout definition follows the supplied strategy:
      LONG  = completed candle CLOSE > setup high
      SHORT = completed candle CLOSE < setup low

    A setup may trigger only from T+1 through T+forward_window.
    """
    patterns = detect_patterns(df, enabled)
    if not patterns:
        return None

    first_recent = max(0, len(df) - recent_bars)
    matches = []

    for setup in patterns:
        s = setup["barIndex"]
        end = min(len(df) - 1, s + forward_window)

        for j in range(s + 1, end + 1):
            c = df.iloc[j]

            if c.close > setup["high"]:
                direction = "LONG"
            elif c.close < setup["low"]:
                direction = "SHORT"
            else:
                continue

            # This range has broken. Only keep it if that FIRST breakout
            # happened within the latest N completed bars.
            if j >= first_recent:
                matches.append({
                    "setupIndex": s,
                    "breakoutIndex": j,
                    "patternType": setup["type"],
                    "direction": direction,
                    "rangeHigh": setup["high"],
                    "rangeLow": setup["low"],
                    "setupTime": df.iloc[s].time,
                    "breakoutTime": df.iloc[j].time,
                    "breakoutClose": float(c.close),
                    "barsAgo": len(df) - 1 - j,
                })
            break

    if not matches:
        return None

    # Display only the newest recent breakout for an instrument.
    return max(matches, key=lambda x: x["breakoutIndex"])

def scan_one(token, environment, instrument, granularity, candle_count,
             enabled, forward_window, recent_bars):
    df = fetch_candles(token, environment, instrument, granularity, candle_count)
    hit = find_recent_breakout(df, enabled, forward_window, recent_bars)
    return instrument, df, hit

def breakout_chart(df, hit, instrument, tf):
    # User requested the last 15 completed bars.
    view = df.tail(15).copy()
    start_index = len(df) - len(view)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=view["time"],
        open=view["open"],
        high=view["high"],
        low=view["low"],
        close=view["close"],
        name=instrument,
    ))

    # Draw the detected range across the visible chart.
    fig.add_hline(
        y=hit["rangeHigh"],
        line_dash="dash",
        annotation_text=f'{hit["patternType"]} High',
        annotation_position="top left",
    )
    fig.add_hline(
        y=hit["rangeLow"],
        line_dash="dash",
        annotation_text=f'{hit["patternType"]} Low',
        annotation_position="bottom left",
    )

    # Mark setup if it is within the 15-bar view.
    if hit["setupIndex"] >= start_index:
        setup = df.iloc[hit["setupIndex"]]
        fig.add_trace(go.Scatter(
            x=[setup.time],
            y=[setup.high],
            mode="markers+text",
            marker=dict(size=11, symbol="diamond"),
            text=[hit["patternType"]],
            textposition="top center",
            name="Range setup",
            hovertemplate=(
                f'{hit["patternType"]} range'
                '<br>%{x}'
                f'<br>High {hit["rangeHigh"]}'
                f'<br>Low {hit["rangeLow"]}'
                '<extra></extra>'
            ),
        ))

    b = df.iloc[hit["breakoutIndex"]]
    fig.add_trace(go.Scatter(
        x=[b.time],
        y=[b.close],
        mode="markers+text",
        marker=dict(
            size=13,
            symbol="triangle-up" if hit["direction"] == "LONG" else "triangle-down",
        ),
        text=[hit["direction"]],
        textposition="top center" if hit["direction"] == "LONG" else "bottom center",
        name="Breakout",
        hovertemplate=(
            f'{hit["direction"]} breakout'
            '<br>%{x}'
            f'<br>Close {hit["breakoutClose"]}'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title=f"{instrument} · {tf} · {hit['patternType']} → {hit['direction']}",
        height=430,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig

st.title("OANDA Recent Range Breakout Scanner")
st.caption(
    "Scans all OANDA instruments for OB / EB / OEB setup ranges whose first "
    "closing-price breakout occurred within the latest completed bars."
)

with st.sidebar:
    st.header("Scanner Controls")

    environment = st.selectbox(
        "OANDA environment",
        ["practice", "live"],
        index=0 if get_secret("OANDA_ENV", "practice") == "practice" else 1,
    )

    token = get_secret("OANDA_API_KEY") or get_secret("OANDA_API_TOKEN")
    if not token:
        token = st.text_input("OANDA API token", type="password")

    account_id = get_secret("OANDA_ACCOUNT_ID")

    tf = st.radio(
        "Timeframe",
        list(OANDA_GRANULARITY),
        index=list(OANDA_GRANULARITY).index("D1"),
        horizontal=True,
    )

    enabled = st.multiselect(
        "Range pattern types",
        ["OB", "EB", "OEB"],
        default=["OB", "EB", "OEB"],
    )

    forward_window = st.slider(
        "Breakout window after range",
        1, 20, 10,
        help="A setup range can trigger only during this many bars after the setup candle.",
    )

    recent_bars = st.slider(
        "Breakout must be within last N bars",
        1, 20, 10,
        help="The breakout candle itself must be this recent.",
    )

    # Enough history to identify recent setups while keeping the all-instrument
    # scan light. The chart itself always displays only the final 15 bars.
    candle_count = st.slider("Candles fetched per instrument", 30, 300, 80, 10)

    max_workers = st.slider(
        "Parallel requests",
        2, 12, 6,
        help="Lower this if OANDA rate-limits the scanner.",
    )

    scan = st.button("Scan All Instruments", type="primary", use_container_width=True)

if not token:
    st.info('Add OANDA_API_KEY to `.streamlit/secrets.toml`, or enter the token in the sidebar.')
    st.stop()

try:
    if not account_id:
        account_id = discover_account_id(token, environment)
    instrument_meta = fetch_oanda_instruments(token, environment, account_id)
    all_instruments = [x["name"] for x in instrument_meta]
    display_names = {
        x["name"]: x.get("displayName", x["name"])
        for x in instrument_meta
    }
except Exception as e:
    st.warning(f"Could not load account instrument list: {e}")
    all_instruments = DEFAULT_INSTRUMENTS
    display_names = {x: x for x in all_instruments}

st.caption(
    f"{len(all_instruments)} instruments available • charts show the last 15 completed bars • "
    "breakouts require a candle close beyond the setup range"
)

if not scan:
    st.info("Choose the timeframe/settings and click **Scan All Instruments**.")
    st.stop()

if not enabled:
    st.warning("Select at least one range pattern type.")
    st.stop()

progress = st.progress(0, text="Scanning OANDA instruments…")
status = st.empty()

hits = []
errors = []
completed = 0

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(
            scan_one,
            token,
            environment,
            instrument,
            OANDA_GRANULARITY[tf],
            candle_count,
            set(enabled),
            forward_window,
            recent_bars,
        ): instrument
        for instrument in all_instruments
    }

    for future in as_completed(futures):
        instrument = futures[future]
        completed += 1

        try:
            inst, df, hit = future.result()
            if hit:
                hits.append({
                    "instrument": inst,
                    "df": df,
                    "hit": hit,
                })
        except Exception as e:
            errors.append(f"{instrument}: {e}")

        progress.progress(
            completed / len(all_instruments),
            text=f"Scanning {completed}/{len(all_instruments)} instruments…",
        )

progress.empty()
status.empty()

# Most recent breakout first, then instrument name.
hits.sort(key=lambda x: (x["hit"]["barsAgo"], x["instrument"]))

st.subheader(f"Matches: {len(hits)}")

if not hits:
    st.warning(
        f"No range breakouts were found within the last {recent_bars} completed {tf} bars "
        f"using the selected setup types."
    )
else:
    summary_rows = []
    for x in hits:
        h = x["hit"]
        summary_rows.append({
            "Instrument": x["instrument"],
            "Name": display_names.get(x["instrument"], x["instrument"]),
            "Pattern": h["patternType"],
            "Direction": h["direction"],
            "Breakout Bars Ago": h["barsAgo"],
            "Setup Time": h["setupTime"],
            "Breakout Time": h["breakoutTime"],
            "Range High": h["rangeHigh"],
            "Range Low": h["rangeLow"],
            "Breakout Close": h["breakoutClose"],
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Two chart cards per row.
    for row_start in range(0, len(hits), 2):
        cols = st.columns(2)

        for col, item in zip(cols, hits[row_start:row_start + 2]):
            h = item["hit"]
            inst = item["instrument"]

            with col:
                st.markdown(
                    f"### {display_names.get(inst, inst)}  \n"
                    f"`{inst}` · **{h['patternType']}** · **{h['direction']}** · "
                    f"breakout **{h['barsAgo']} bars ago**"
                )
                st.plotly_chart(
                    breakout_chart(item["df"], h, inst, tf),
                    use_container_width=True,
                    key=f"chart_{inst}_{h['breakoutIndex']}",
                )

if errors:
    with st.expander(f"Instrument errors ({len(errors)})"):
        for err in errors:
            st.text(err)

with st.expander("Scanner definition"):
    st.markdown(
        f"""
**Range setup**
- OB: current high > previous high AND current low < previous low.
- EB: same opposite-direction body-engulfing definition as the supplied dashboard.
- OEB: both OB and EB; OEB keeps precedence over OB/EB.

**Breakout**
- LONG: a completed candle closes above the setup candle high.
- SHORT: a completed candle closes below the setup candle low.
- The first breakout of each setup is used.
- It must occur within the selected **breakout window** after the setup.
- It must also be within the latest **{recent_bars} completed bars**.

**Display**
- Every matching instrument is listed.
- Each instrument gets a candlestick chart of exactly the latest **15 completed bars**.
- Dashed horizontal levels are the setup range high and low.
        """
    )

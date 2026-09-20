import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# OANDA CONFIGURATION
# ============================================================
PRACTICE_URL = "https://api-fxpractice.oanda.com"
LIVE_URL = "https://api-fxtrade.oanda.com"

GRANULARITIES = [
    "S5", "S10", "S15", "S30", "M1", "M2", "M4", "M5", "M10", "M15",
    "M30", "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D", "W", "M"
]

TIMEFRAME_SECONDS = {
    "S5": 5, "S10": 10, "S15": 15, "S30": 30,
    "M1": 60, "M2": 120, "M4": 240, "M5": 300, "M10": 600,
    "M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H3": 10800,
    "H4": 14400, "H6": 21600, "H8": 28800, "H12": 43200,
}


@dataclass
class Level:
    price: float
    is_res: bool
    start_index: int
    touches: int


def get_credentials() -> Tuple[str, str]:
    """Read API token/environment. Account ID is discovered from OANDA."""
    token = ""
    env = "Practice"
    try:
        token = st.secrets.get("OANDA_TOKEN", "")
        env = st.secrets.get("OANDA_ENVIRONMENT", "Practice")
    except Exception:
        pass
    token = token or os.getenv("OANDA_TOKEN", "")
    env = env or os.getenv("OANDA_ENVIRONMENT", "Practice")
    return token, env


def discover_accounts(token: str, env: str) -> list:
    """Return accounts authorized by the supplied OANDA token."""
    data = oanda_get("/v3/accounts", token, env)
    return data.get("accounts", [])


def api_base(env: str) -> str:
    return LIVE_URL if env.lower().startswith("live") else PRACTICE_URL


def oanda_get(path: str, token: str, env: str, params: Optional[dict] = None) -> dict:
    url = api_base(env) + path
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(f"OANDA API {response.status_code}: {detail}")
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def discover_account_ids(token: str, env: str) -> tuple:
    accounts = discover_accounts(token, env)
    ids = tuple(a.get("id") for a in accounts if a.get("id"))
    if not ids:
        raise RuntimeError("No OANDA accounts are authorized for this API token.")
    return ids


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_instruments(token: str, account_id: str, env: str) -> pd.DataFrame:
    data = oanda_get(f"/v3/accounts/{account_id}/instruments", token, env)
    rows = []
    for x in data.get("instruments", []):
        rows.append({
            "name": x.get("name"),
            "displayName": x.get("displayName"),
            "type": x.get("type"),
            "pipLocation": x.get("pipLocation"),
            "displayPrecision": x.get("displayPrecision"),
            "marginRate": x.get("marginRate"),
        })
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)


@st.cache_data(ttl=5, show_spinner=False)
def fetch_candles(
    token: str,
    account_id: str,
    env: str,
    instrument: str,
    granularity: str,
    count: int,
    daily_alignment: int,
    alignment_timezone: str,
    weekly_alignment: str,
    price: str = "M",
) -> pd.DataFrame:
    params = {
        "price": price,
        "granularity": granularity,
        "count": min(int(count), 5000),
        "smooth": "false",
        "dailyAlignment": int(daily_alignment),
        "alignmentTimezone": alignment_timezone,
        "weeklyAlignment": weekly_alignment,
    }
    data = oanda_get(
        f"/v3/accounts/{account_id}/instruments/{instrument}/candles",
        token,
        env,
        params,
    )

    rows = []
    for c in data.get("candles", []):
        mid = c.get("mid")
        if not mid:
            continue
        rows.append({
            "time": pd.to_datetime(c["time"], utc=True),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": int(c.get("volume", 0)),
            "complete": bool(c.get("complete", False)),
        })

    if not rows:
        raise RuntimeError("OANDA returned no candles for this instrument/timeframe.")

    return pd.DataFrame(rows).sort_values("time").drop_duplicates("time").reset_index(drop=True)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def pine_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    # TradingView ta.atr() uses Wilder/RMA smoothing.
    tr = true_range(df)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def strat_state(o: float, h: float, l: float, c: float, ph: float, pl: float, mintick: float) -> str:
    body = abs(c - o)
    eff_body = max(body, mintick)
    body_top = max(o, c)
    body_bottom = min(o, c)
    upper_wick = h - body_top
    lower_wick = body_bottom - l

    is_hammer = lower_wick >= 2.0 * eff_body and lower_wick > upper_wick
    is_shooter = upper_wick >= 2.0 * eff_body and upper_wick > lower_wick

    col_tag = "G" if c >= o else "R"
    if h > ph and l < pl:
        base_state = "3"
    elif h > ph:
        base_state = "2U"
    elif l < pl:
        base_state = "2D"
    else:
        base_state = "1"

    if base_state == "1":
        return "1"
    if base_state in ("2U", "2D"):
        if is_shooter:
            return "2-SS"
        if is_hammer:
            return "2-H"
        return base_state + col_tag
    if is_shooter:
        return "3-SS"
    if is_hammer:
        return "3-H"
    return "3" + col_tag


def compute_strat_states(df: pd.DataFrame, mintick: float) -> pd.Series:
    prev_h = df["high"].shift(1)
    prev_l = df["low"].shift(1)
    states = []
    for i in range(len(df)):
        if i == 0 or pd.isna(prev_h.iloc[i]) or pd.isna(prev_l.iloc[i]):
            states.append(None)
        else:
            states.append(strat_state(
                df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i],
                df["close"].iloc[i], prev_h.iloc[i], prev_l.iloc[i], mintick
            ))
    return pd.Series(states, index=df.index, dtype="object")


def build_sr_levels(
    df: pd.DataFrame,
    min_touches: int,
    lookback: int,
    tolerance_mult: float,
    min_wick_mult: float,
    invalidation: str,
) -> Tuple[List[Level], List[str]]:
    """Sequentially reproduce the Pine active-level lifecycle.

    The returned final levels are the active levels at the latest bar. sr_contexts
    are calculated from those final levels, matching the Pine script's use of
    f_get_sr_context(i, active_levels) inside its historical scan.
    """
    work = df.copy()
    work["atr"] = pine_atr(work, 14)
    work["body_top"] = work[["open", "close"]].max(axis=1)
    work["body_bottom"] = work[["open", "close"]].min(axis=1)
    work["upper_wick"] = work["high"] - work["body_top"]
    work["lower_wick"] = work["body_bottom"] - work["low"]

    active: List[Level] = []

    for i in range(len(work)):
        atr = work["atr"].iloc[i]
        if pd.isna(atr):
            continue
        tol = atr * tolerance_mult
        min_wick = atr * min_wick_mult
        o = work["open"].iloc[i]
        h = work["high"].iloc[i]
        l = work["low"].iloc[i]
        c = work["close"].iloc[i]
        body_top = work["body_top"].iloc[i]
        body_bottom = work["body_bottom"].iloc[i]
        upper_wick = work["upper_wick"].iloc[i]
        lower_wick = work["lower_wick"].iloc[i]

        # 1. Update existing levels, invalidate first, then count touches.
        new_active = []
        for lvl in active:
            violated = (c > lvl.price) if lvl.is_res and invalidation == "Close" else False
            if lvl.is_res and invalidation != "Close":
                violated = h > lvl.price
            if not lvl.is_res and invalidation == "Close":
                violated = c < lvl.price
            if not lvl.is_res and invalidation != "Close":
                violated = l < lvl.price
            if violated:
                continue

            if lvl.is_res:
                new_touch = upper_wick >= min_wick and abs(body_top - lvl.price) <= tol
            else:
                new_touch = lower_wick >= min_wick and abs(body_bottom - lvl.price) <= tol
            if new_touch:
                lvl.touches += 1
            new_active.append(lvl)
        active = new_active

        # 2. Detect resistance cluster.
        if upper_wick >= min_wick:
            count = 1
            earliest = 0
            total = body_top
            for j in range(1, min(lookback, i) + 1):
                if (
                    work["upper_wick"].iloc[i - j] >= min_wick
                    and abs(work["body_top"].iloc[i - j] - body_top) <= tol
                ):
                    count += 1
                    earliest = j
                    total += work["body_top"].iloc[i - j]
            if count >= min_touches:
                avg = total / count
                duplicate = any(abs(avg - x.price) <= tol * 1.5 for x in active)
                if not duplicate:
                    active.append(Level(avg, True, i - earliest, count))

        # 3. Detect support cluster.
        if lower_wick >= min_wick:
            count = 1
            earliest = 0
            total = body_bottom
            for j in range(1, min(lookback, i) + 1):
                if (
                    work["lower_wick"].iloc[i - j] >= min_wick
                    and abs(work["body_bottom"].iloc[i - j] - body_bottom) <= tol
                ):
                    count += 1
                    earliest = j
                    total += work["body_bottom"].iloc[i - j]
            if count >= min_touches:
                avg = total / count
                duplicate = any(abs(avg - x.price) <= tol * 1.5 for x in active)
                if not duplicate:
                    active.append(Level(avg, False, i - earliest, count))

    return active, []


def sr_context_from_levels(row: pd.Series, levels: List[Level]) -> str:
    ctx = "Neutral"
    c, h, l = row["close"], row["high"], row["low"]
    for lvl in levels:
        p = lvl.price
        if lvl.is_res:
            if c > p:
                return "Break Resistance"
            if h >= p and c <= p:
                return "Sweep Resistance"
        else:
            if c < p:
                return "Break Support"
            if l <= p and c >= p:
                return "Sweep Support"
    return ctx


def state_is_bullish(state: str) -> bool:
    return state.endswith("G") or state in ("2-H", "3-H")


def add_htf_state(base: pd.DataFrame, htf: pd.DataFrame, mintick: float, prefix: str) -> pd.DataFrame:
    x = htf.copy()
    x[f"{prefix}_state"] = compute_strat_states(x, mintick)
    x[f"{prefix}_prev_high"] = x["high"].shift(1)
    x[f"{prefix}_prev_low"] = x["low"].shift(1)

    # Merge the latest HTF candle whose start time is <= the base candle.
    right = x[["time", f"{prefix}_state", "high", "low"]].rename(columns={
        "time": f"{prefix}_time",
        "high": f"{prefix}_high",
        "low": f"{prefix}_low",
    })
    return pd.merge_asof(
        base.sort_values("time"),
        right.sort_values(f"{prefix}_time"),
        left_on="time",
        right_on=f"{prefix}_time",
        direction="backward",
    )


def dow_name(ts: pd.Timestamp) -> str:
    return ts.day_name()


def four_hour_session(ts: pd.Timestamp) -> Tuple[int, str]:
    h = ts.hour
    sid = h // 4
    labels = [
        "00:00 - 04:00", "04:00 - 08:00", "08:00 - 12:00",
        "12:00 - 16:00", "16:00 - 20:00", "20:00 - 00:00"
    ]
    return sid, labels[sid]


def fmt_state(state) -> str:
    return "N/A" if state is None or pd.isna(state) else str(state)


def eight_hour_session(ts: pd.Timestamp) -> Tuple[int, str]:
    h = ts.hour
    sid = h // 8
    labels = [
        "00:00 - 08:00", "08:00 - 16:00", "16:00 - 00:00"
    ]
    return sid, labels[sid]


def calculate_prediction(
    base: pd.DataFrame,
    htf1: pd.DataFrame,
    htf2: pd.DataFrame,
    use_offset: bool,
    lookback_n: int,
    max_history: int,
    filter_htf: bool,
    filter_htf2: bool,
    filter_sr: bool,
    filter_dow: bool,
    dow_shift_days: int,
    filter_4h: bool,
    filter_8h: bool,
    sr_levels: List[Level],
    mintick: float,
) -> Dict:
    df = base.copy().reset_index(drop=True)
    df["ltf_state"] = compute_strat_states(df, mintick)

    # Attach HTF states directly from OANDA HTF candles. This preserves the
    # active HTF candle rather than building it from a potentially stale LTF feed.
    df = add_htf_state(df, htf1, mintick, "htf1")
    df = add_htf_state(df, htf2, mintick, "htf2")
    df = df.reset_index(drop=True)

    n = len(df)
    off = 1 if use_offset else 0
    current_end = n - 1 - off
    current_start = current_end - lookback_n + 1

    if current_start < 0:
        raise RuntimeError("Not enough base candles for the requested pattern length.")

    current_states = df.loc[current_start:current_end, "ltf_state"].tolist()
    current_seq = " ➔ ".join(fmt_state(x) for x in current_states)

    current_htf1 = fmt_state(df.loc[current_end, "htf1_state"])
    current_htf2 = fmt_state(df.loc[current_end, "htf2_state"])

    current_ts = df.loc[current_end, "time"]
    current_ts_shifted = current_ts + pd.Timedelta(days=dow_shift_days)
    current_dow = dow_name(current_ts_shifted)
    current_sid, current_session = four_hour_session(current_ts)
    current_sid8, current_session8 = eight_hour_session(current_ts)
    current_sr = sr_context_from_levels(df.loc[current_end], sr_levels)

    matches = []
    earliest_endpoint = lookback_n + off  # corresponds to Pine i = 1 + off
    latest_i = min(n - 1 - lookback_n - off, max_history)

    if latest_i >= 1 + off:
        for i in range(1 + off, latest_i + 1):
            # Pine pattern: ltf_state[i+k] for k=0..lookback-1.
            # In chronological dataframe this endpoint is n-1-(i+lookback-1).
            hist_end = n - 1 - (i + lookback_n - 1)
            hist_start = hist_end - lookback_n + 1
            next_idx = hist_end + 1
            if hist_start < 0 or next_idx >= n:
                continue

            hist_states = df.loc[hist_start:hist_end, "ltf_state"].tolist()
            if hist_states != current_states:
                continue

            is_match = True
            if filter_htf and fmt_state(df.loc[hist_end, "htf1_state"]) != current_htf1:
                is_match = False
            if is_match and filter_htf2 and fmt_state(df.loc[hist_end, "htf2_state"]) != current_htf2:
                is_match = False
            if is_match and filter_sr:
                hist_sr = sr_context_from_levels(df.loc[hist_end], sr_levels)
                if hist_sr != current_sr:
                    is_match = False
            if is_match and filter_dow:
                hist_ts = df.loc[hist_end, "time"] + pd.Timedelta(days=dow_shift_days)
                if dow_name(hist_ts) != current_dow:
                    is_match = False
            if is_match and filter_4h:
                hist_sid, _ = four_hour_session(df.loc[hist_end, "time"])
                if hist_sid != current_sid:
                    is_match = False
            if is_match and filter_8h:
                hist_sid8, _ = eight_hour_session(df.loc[hist_end, "time"])
                if hist_sid8 != current_sid8:
                    is_match = False

            if is_match:
                matches.append(fmt_state(df.loc[next_idx, "ltf_state"]))

    total = len(matches)
    counts = pd.Series(matches, dtype="object").value_counts() if matches else pd.Series(dtype=int)
    if total:
        highest_state = str(counts.index[0])
        highest_prob = float(counts.iloc[0] / total * 100)
        bull_prob = float(sum(state_is_bullish(x) for x in matches) / total * 100)
    else:
        highest_state = "N/A"
        highest_prob = 0.0
        bull_prob = 0.0

    return {
        "df": df,
        "current_end": current_end,
        "current_seq": current_seq,
        "current_htf1": current_htf1,
        "current_htf2": current_htf2,
        "current_sr": current_sr,
        "current_dow": current_dow,
        "current_session": current_session,
        "current_session8": current_session8,
        "matches": matches,
        "total_matches": total,
        "highest_state": highest_state,
        "highest_prob": highest_prob,
        "bull_prob": bull_prob,
        "state_counts": counts,
    }


def calculate_all_pattern_stats(result: Dict, lookback_n: int, max_history: int, filter_htf: bool, filter_htf2: bool, filter_sr: bool, filter_dow: bool, dow_shift_days: int, filter_4h: bool, filter_8h: bool, sr_levels: List[Level]) -> pd.DataFrame:
    """Aggregate every found LTF pattern using the same next-bar/bullish logic as the Pine tracker.

    Enabled context filters are held to the currently selected context, exactly like the
    main prediction. Only the LTF pattern sequence varies across the historical scan.
    """
    df = result["df"]
    n = len(df)
    current_end = result["current_end"]
    current_htf1 = result["current_htf1"]
    current_htf2 = result["current_htf2"]
    current_sr = result["current_sr"]
    current_dow = result["current_dow"]
    current_sid, _ = four_hour_session(df.loc[current_end, "time"])
    current_sid8, _ = eight_hour_session(df.loc[current_end, "time"])

    grouped = {}
    # Scan chronological pattern endpoints. A pattern must have one following bar.
    first_end = lookback_n - 1
    last_end = n - 2
    # Match Pine's historical lookback cap by limiting distance from the latest bar.
    first_allowed = max(first_end, n - 1 - max_history - lookback_n)

    for hist_end in range(first_allowed, last_end + 1):
        hist_start = hist_end - lookback_n + 1
        next_idx = hist_end + 1
        if hist_start < 0:
            continue

        is_match = True
        if filter_htf and fmt_state(df.loc[hist_end, "htf1_state"]) != current_htf1:
            is_match = False
        if is_match and filter_htf2 and fmt_state(df.loc[hist_end, "htf2_state"]) != current_htf2:
            is_match = False
        if is_match and filter_sr and sr_context_from_levels(df.loc[hist_end], sr_levels) != current_sr:
            is_match = False
        if is_match and filter_dow:
            hist_ts = df.loc[hist_end, "time"] + pd.Timedelta(days=dow_shift_days)
            if dow_name(hist_ts) != current_dow:
                is_match = False
        if is_match and filter_4h:
            sid, _ = four_hour_session(df.loc[hist_end, "time"])
            if sid != current_sid:
                is_match = False
        if is_match and filter_8h:
            sid8, _ = eight_hour_session(df.loc[hist_end, "time"])
            if sid8 != current_sid8:
                is_match = False
        if not is_match:
            continue

        states = [fmt_state(x) for x in df.loc[hist_start:hist_end, "ltf_state"].tolist()]
        pattern = " ➔ ".join(states)
        next_state = fmt_state(df.loc[next_idx, "ltf_state"])
        grouped.setdefault(pattern, []).append(next_state)

    rows = []
    for pattern, next_states in grouped.items():
        total = len(next_states)
        counts = pd.Series(next_states, dtype="object").value_counts()
        highest_state = str(counts.index[0])
        highest_count = int(counts.iloc[0])
        highest_prob = highest_count / total * 100.0
        bull_count = sum(state_is_bullish(x) for x in next_states)
        bull_prob = bull_count / total * 100.0
        rows.append({
            "Pattern": pattern,
            "Sample Size": total,
            "Most Probable Next Bar": highest_state,
            "Next-Bar Probability %": highest_prob,
            "Bullish Probability %": bull_prob,
        })

    if not rows:
        return pd.DataFrame(columns=["Pattern", "Sample Size", "Most Probable Next Bar", "Next-Bar Probability %", "Bullish Probability %"])
    return pd.DataFrame(rows).sort_values(["Sample Size", "Next-Bar Probability %"], ascending=[False, False]).reset_index(drop=True)


def touch_width(touches: int) -> int:
    if touches < 4:
        return 1
    if touches < 6:
        return 2
    if touches < 8:
        return 3
    if touches < 10:
        return 4
    return 5


def make_chart(df: pd.DataFrame, levels: List[Level], show_bars: int = 250) -> go.Figure:
    d = df.tail(show_bars).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d["time"],
        open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="OANDA Mid",
    ))

    if len(d):
        x0, x1 = d["time"].iloc[0], d["time"].iloc[-1]
        span = x1 - x0
        if span.total_seconds() <= 0:
            span = pd.Timedelta(days=1)
        x2 = x1 + span * 0.04
        for lvl in levels:
            if lvl.price < d["low"].min() * 0.95 or lvl.price > d["high"].max() * 1.05:
                continue
            fig.add_shape(
                type="line", x0=x0, x1=x2, y0=lvl.price, y1=lvl.price,
                line=dict(width=touch_width(lvl.touches), dash="solid"),
            )
            fig.add_annotation(
                x=x2, y=lvl.price,
                text=f"{lvl.touches} touches",
                showarrow=False,
                xanchor="left",
            )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=90, t=40, b=10),
        title="OANDA Mid-price candles + active S/R levels",
    )
    return fig


def main():
    st.set_page_config(page_title="OANDA Key Body Level Rejection Finder", layout="wide")
    st.title("Key Body Level Rejection Finder & Strat Tracker — OANDA")
    st.caption("Python/Streamlit conversion of the supplied Pine Script. Market data is pulled directly from OANDA v20 REST API.")

    token0, env0 = get_credentials()

    with st.sidebar:
        st.header("OANDA Connection")
        environment = st.selectbox("Environment", ["Practice", "Live"], index=0 if env0.lower().startswith("prac") else 1)
        token = st.text_input("OANDA API token", value=token0, type="password", help="Prefer storing this in Streamlit secrets rather than hard-coding it.")

        st.divider()
        st.header("Account")
        account_id = None
        if token:
            try:
                account_ids = discover_account_ids(token, environment)
                if len(account_ids) == 1:
                    account_id = account_ids[0]
                    st.caption(f"Account: {account_id}")
                else:
                    account_id = st.selectbox("OANDA account", list(account_ids))
            except Exception as e:
                st.error(f"Unable to discover OANDA account: {e}")

        st.header("Market")
        if token and account_id:
            try:
                instruments = fetch_instruments(token, account_id, environment)
                names = instruments["name"].tolist()
                default_symbol = names.index("EUR_USD") if "EUR_USD" in names else 0
                instrument = st.selectbox("OANDA instrument", names, index=default_symbol)
                selected_info = instruments[instruments["name"] == instrument].iloc[0]
                pip_location = int(selected_info["pipLocation"])
                display_precision = int(selected_info["displayPrecision"])
            except Exception as e:
                st.error(f"Unable to load OANDA instruments: {e}")
                st.stop()
        else:
            instrument = st.text_input("OANDA instrument", "EUR_USD")
            pip_location = -4
            display_precision = 5

        granularity = st.selectbox("Chart / LTF granularity", GRANULARITIES, index=GRANULARITIES.index("H1"))
        price_component = st.selectbox("Price component", ["M"], index=0, help="The supplied Pine script uses OHLC bars. This conversion uses OANDA midpoint candles.")
        history_bars = st.slider("OANDA historical candles", 500, 5000, 5000, step=100)

        st.divider()
        st.header("S/R Level Detection")
        min_touches = st.number_input("Minimum Body Touches", 2, 6, 3)
        lookback = st.number_input("Lookback Window (Bars)", 5, 100, 20)
        tolerance_mult = st.number_input("Price Tolerance (ATR Mult)", 0.01, 5.0, 0.15, step=0.05)
        min_wick_mult = st.number_input("Min Wick Size (ATR Mult)", 0.0, 5.0, 0.10, step=0.05)
        invalidation = st.selectbox("Violation Trigger", ["Close", "Wick (High/Low)"])

        st.divider()
        st.header("Strat Tracker")
        use_offset = st.checkbox("Use Closed Bars Only (Live Mode)", True)
        htf_input = st.selectbox("HTF 1 Context Timeframe", ["D", "W", "M", "H4", "H1"], index=0)
        filter_htf = st.checkbox("Filter by Active HTF 1 Context", False)
        htf_input2 = st.selectbox("HTF 2 Context Timeframe", ["W", "D", "M", "H4", "H1"], index=0)
        filter_htf2 = st.checkbox("Filter by Active HTF 2 Context", False)
        filter_sr = st.checkbox("Filter by Active S/R Context", False)
        filter_dow = st.checkbox("Filter by Day of Week", False)
        dow_shift_days = st.slider("Day of Week Shift (Days)", -2, 2, 0)
        filter_4h = st.checkbox("Filter by 4-Hour Session", False)
        filter_8h = st.checkbox("Filter by 8-Hour Session", False)
        show_all_patterns = st.checkbox("Display All Found Patterns", False, help="Show every historical pattern sequence, its sample size, most probable next-bar state, next-bar probability, and bullish probability using the Pine tracker logic.")
        lookback_n = st.slider("Pattern Lookback Window (Candles)", 1, 5, 3)
        max_history = st.slider("Historical Lookback Bars", 100, 5000, 4600, step=100)

        st.divider()
        st.header("OANDA Candle Alignment")
        daily_alignment = st.number_input("Daily alignment hour", 0, 23, 17)
        alignment_timezone = st.text_input("Alignment timezone", "America/New_York")
        weekly_alignment = st.selectbox("Weekly alignment", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)

        st.divider()
        chart_bars = st.slider("Chart candles", 50, 500, 250, step=25)
        refresh = st.button("Refresh OANDA data", type="primary", use_container_width=True)

    if not token:
        st.info("Enter your OANDA API token in the sidebar, or configure OANDA_TOKEN in Streamlit secrets. The account ID is discovered automatically from the token.")
        st.stop()

    if not account_id:
        st.info("No OANDA account could be discovered from this API token.")
        st.stop()

    if refresh:
        fetch_candles.clear()
        fetch_instruments.clear()

    try:
        with st.spinner("Fetching fresh OANDA candles…"):
            base = fetch_candles(token, account_id, environment, instrument, granularity, history_bars,
                                 daily_alignment, alignment_timezone, weekly_alignment, price_component)
            htf1 = fetch_candles(token, account_id, environment, instrument, htf_input, min(history_bars, 5000),
                                 daily_alignment, alignment_timezone, weekly_alignment, price_component)
            htf2 = fetch_candles(token, account_id, environment, instrument, htf_input2, min(history_bars, 5000),
                                 daily_alignment, alignment_timezone, weekly_alignment, price_component)

        levels, _ = build_sr_levels(base, int(min_touches), int(lookback), float(tolerance_mult),
                                    float(min_wick_mult), invalidation)

        result = calculate_prediction(
            base=base,
            htf1=htf1,
            htf2=htf2,
            use_offset=use_offset,
            lookback_n=int(lookback_n),
            max_history=int(max_history),
            filter_htf=filter_htf,
            filter_htf2=filter_htf2,
            filter_sr=filter_sr,
            filter_dow=filter_dow,
            dow_shift_days=int(dow_shift_days),
            filter_4h=filter_4h,
            filter_8h=filter_8h,
            sr_levels=levels,
            mintick=10 ** pip_location,
        )

    except Exception as e:
        st.exception(e)
        st.stop()

    df = result["df"]
    end_idx = result["current_end"]
    current_row = df.loc[end_idx]

    # -------------------- status strip --------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Instrument", instrument)
    c2.metric("LTF", granularity)
    c3.metric("Latest OANDA candle", current_row["time"].strftime("%Y-%m-%d %H:%M UTC"))
    c4.metric("Latest close", f"{current_row['close']:.{display_precision}f}")
    c5.metric("Active S/R levels", str(len(levels)))

    if not bool(base.iloc[-1]["complete"]):
        st.success("OANDA returned an active/incomplete LTF candle — live mode can therefore use the current candle without waiting for the bar to close.")
    else:
        st.info("The latest OANDA LTF candle is currently marked complete.")

    # -------------------- chart --------------------
    st.plotly_chart(make_chart(base, levels, chart_bars), use_container_width=True)

    # -------------------- prediction table --------------------
    st.subheader("Strat Probability Tracker")
    rows = []
    if filter_htf:
        rows.append((f"Active HTF 1 Context ({htf_input})", result["current_htf1"]))
    if filter_htf2:
        rows.append((f"Active HTF 2 Context ({htf_input2})", result["current_htf2"]))
    if filter_sr:
        rows.append(("Active S/R Context", result["current_sr"]))
    if filter_dow:
        rows.append(("Day of Week Context", result["current_dow"]))
    if filter_4h:
        rows.append(("4-Hour Session Context", result["current_session"]))
    if filter_8h:
        rows.append(("8-Hour Session Context", result["current_session8"]))

    rows.extend([
        ("Closed Pattern Sequence" if use_offset else "Pattern Sequence (Inc. Live)", result["current_seq"]),
        ("Sample Size (Occurrences)", result["total_matches"]),
        ("Predicted Live Bar Form" if use_offset else "Predicted Next Bar Form",
         f"{result['highest_state']} ({result['highest_prob']:.2f}%)" if result["total_matches"] else "N/A"),
        ("Total Bullish Probability", f"{result['bull_prob']:.2f}%" if result["total_matches"] else "N/A"),
    ])
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]), hide_index=True, use_container_width=True)

    # -------------------- active levels --------------------
    with st.expander("Active S/R levels"):
        level_rows = []
        for x in sorted(levels, key=lambda z: z.price):
            level_rows.append({
                "Type": "Resistance" if x.is_res else "Support",
                "Price": round(x.price, display_precision),
                "Touches": x.touches,
                "Width": touch_width(x.touches),
                "Start candle": x.start_index,
            })
        if level_rows:
            st.dataframe(pd.DataFrame(level_rows), hide_index=True, use_container_width=True)
        else:
            st.write("No active S/R clusters found with the current settings.")

    # -------------------- all found pattern statistics --------------------
    if show_all_patterns:
        st.subheader("All Found Pattern Statistics")
        all_stats = calculate_all_pattern_stats(
            result=result,
            lookback_n=int(lookback_n),
            max_history=int(max_history),
            filter_htf=filter_htf,
            filter_htf2=filter_htf2,
            filter_sr=filter_sr,
            filter_dow=filter_dow,
            dow_shift_days=int(dow_shift_days),
            filter_4h=filter_4h,
            filter_8h=filter_8h,
            sr_levels=levels,
        )
        if len(all_stats):
            display_stats = all_stats.copy()
            display_stats["Next-Bar Probability"] = display_stats.pop("Next-Bar Probability %").map(lambda x: f"{x:.2f}%")
            display_stats["Bullish Probability"] = display_stats.pop("Bullish Probability %").map(lambda x: f"{x:.2f}%")
            st.dataframe(display_stats, hide_index=True, use_container_width=True)
            st.caption("Each row is a complete pattern sequence of the selected lookback length. Sample Size is the number of historical occurrences. Most Probable Next Bar and its probability are calculated from the bars immediately following those occurrences. Bullish Probability uses the Pine rule: states ending in G plus 2-H and 3-H are bullish.")
        else:
            st.write("No historical patterns found for the selected filters.")

    with st.expander("Current pattern next-state distribution"):
        counts = result["state_counts"]
        if len(counts):
            dist = counts.rename("Occurrences").to_frame()
            dist["Probability %"] = dist["Occurrences"] / result["total_matches"] * 100
            st.dataframe(dist, use_container_width=True)
        else:
            st.write("No historical matches for the selected current pattern and filters.")

    # -------------------- raw data --------------------
    with st.expander("Latest OANDA candles"):
        show = base.tail(100).copy()
        show["time"] = show["time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        st.dataframe(show.iloc[::-1], hide_index=True, use_container_width=True)

    st.caption("OANDA candle timestamps are UTC. Sessions are calculated from the OANDA candle timestamp: 4-hour blocks and 8-hour blocks start at 00:00 UTC. OANDA's candle endpoint supports up to 5,000 candles per request; this app requests at most that amount. The original Pine logic was preserved where practical, while chart rendering is implemented with Plotly.")


if __name__ == "__main__":
    main()

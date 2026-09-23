import os
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

PRACTICE_URL = "https://api-fxpractice.oanda.com"
LIVE_URL = "https://api-fxtrade.oanda.com"

GRANULARITIES = [
    "S5", "S10", "S15", "S30", "M1", "M2", "M4", "M5", "M10", "M15",
    "M30", "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D", "W", "M"
]

# -----------------------------------------------------------------------------
# EASY-TO-EDIT WATCHLIST
# -----------------------------------------------------------------------------
# A one-candle entry checks only B (the latest closed candle).
# A two-candle entry checks A -> B (the two latest closed candles).
#
# Exact v7 states:
# 1, 2UG, 2UR, 2DG, 2DR, 2-H, 2-SS, 3G, 3R, 3-H, 3-SS
#
# The requested Type-2 pairs are included below. Add/remove tuples here only.
SCAN_PATTERNS = [
    ("1",),
    ("2-SS",),
    ("2-H",),
    ("2UG", "2UR"),
    ("2DR", "2UG"),
    ("2UG", "2DR"),
    ("2DG", "2UR"),
]


def api_base(env: str) -> str:
    return LIVE_URL if env.lower().startswith("live") else PRACTICE_URL


def get_credentials() -> Tuple[str, str]:
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
    data = oanda_get("/v3/accounts", token, env)
    ids = tuple(a.get("id") for a in data.get("accounts", []) if a.get("id"))
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
    if not rows:
        return pd.DataFrame()
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
        token, env, params,
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


# -----------------------------------------------------------------------------
# V7 STRAT STATE — exact candle classification used by the Pine baseline.
# -----------------------------------------------------------------------------
def strat_state(o, h, l, c, ph, pl, mintick: float, include_color: bool = True) -> str:
    body = abs(c - o)
    eff_body = max(body, mintick)
    body_top = max(o, c)
    body_bottom = min(o, c)
    upper = h - body_top
    lower = body_bottom - l

    hammer = lower >= 2.0 * eff_body and lower > upper
    shooter = upper >= 2.0 * eff_body and upper > lower
    green = c >= o
    outside = h > ph and l < pl
    two_up = h > ph and not outside
    two_down = l < pl and not outside
    inside = h <= ph and l >= pl

    if inside:
        return "1"
    if outside:
        if hammer:
            return "3-H"
        if shooter:
            return "3-SS"
        return ("3G" if green else "3R") if include_color else "3"
    if two_up:
        if hammer:
            return "2-H"
        if shooter:
            return "2-SS"
        return ("2UG" if green else "2UR") if include_color else "2U"
    if two_down:
        if hammer:
            return "2-H"
        if shooter:
            return "2-SS"
        return ("2DG" if green else "2DR") if include_color else "2D"
    return "N/A"


def compute_states(df: pd.DataFrame, mintick: float, include_color: bool = True) -> np.ndarray:
    n = len(df)
    out = np.full(n, "N/A", dtype=object)
    if n < 2:
        return out
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    for i in range(1, n):
        out[i] = strat_state(o[i], h[i], l[i], c[i], h[i-1], l[i-1], mintick, include_color)
    return out


def find_latest_watch_setup(states: np.ndarray, last_closed: int) -> Optional[Dict]:
    """Check only the latest closed B and, for 2-candle patterns, A/B."""
    if last_closed < 1:
        return None
    b = str(states[last_closed])
    a = str(states[last_closed - 1])
    for pattern in SCAN_PATTERNS:
        if len(pattern) == 1 and b == pattern[0]:
            return {"pattern": pattern, "label": pattern[0], "a_state": a, "b_state": b}
        if len(pattern) == 2 and (a, b) == tuple(pattern):
            return {"pattern": pattern, "label": f"{a} ➔ {b}", "a_state": a, "b_state": b}
    return None


def evaluate_bar_ohlc(b_high, b_low, a_high, a_low, c_high, c_low):
    """Fallback when no lower-TF data is available: v7 semantics, conservative ambiguity."""
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
    if hit_tp and hit_sl:
        return direction, 2
    if hit_tp:
        return direction, 1
    if hit_sl:
        return direction, 2  # entry + SL in same main-TF bar: order unknown
    return direction, 3


def resolve_active_ohlc(direction, sl, tp, h, l):
    hit_tp = h >= tp if direction == 1 else l <= tp
    hit_sl = l <= sl if direction == 1 else h >= sl
    if hit_tp and hit_sl:
        return 2
    if hit_tp:
        return 1
    if hit_sl:
        return -1
    return 0


def actionable_stats_for_pattern(df: pd.DataFrame, states: np.ndarray, pattern: tuple, history_bars: int) -> List[Dict]:
    """
    Replays historical A->B->C using the v7 trade definitions.

    One-candle watch patterns mean "B has this state". A is still the candle before B,
    because v7 requires A's high/low as the target. Two-candle patterns require exact A/B.

    This main-TF implementation is deliberately conservative for bars whose OHLC cannot
    establish intrabar ordering; those are Ambiguous, just like unresolved same-LTF cases
    in Pine. The scanner therefore never invents an ordering from OHLC.
    """
    n = len(df)
    if n < 4:
        return []
    first_c = max(2, n - int(history_bars))
    last_c = n - 1

    stats = {
        "Long": {"wins": 0, "losses": 0, "ambiguous": 0},
        "Short": {"wins": 0, "losses": 0, "ambiguous": 0},
    }
    active = []

    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)

    for c_idx in range(first_c, last_c + 1):
        # Resolve trades opened by previous C candles first, same order as v7.
        keep = []
        for tr in active:
            result = resolve_active_ohlc(tr["dir"], tr["sl"], tr["tp"], highs[c_idx], lows[c_idx])
            bucket = stats[tr["side"]]
            if result == 1:
                bucket["wins"] += 1
            elif result == -1:
                bucket["losses"] += 1
            elif result == 2:
                bucket["ambiguous"] += 1
            else:
                keep.append(tr)
        active = keep

        a_idx, b_idx = c_idx - 2, c_idx - 1
        if a_idx < 1:
            continue
        a_state, b_state = str(states[a_idx]), str(states[b_idx])
        matched = b_state == pattern[0] if len(pattern) == 1 else (a_state, b_state) == tuple(pattern)
        if not matched:
            continue

        direction, outcome = evaluate_bar_ohlc(
            highs[b_idx], lows[b_idx], highs[a_idx], lows[a_idx], highs[c_idx], lows[c_idx]
        )
        if direction == 0:
            # Same as Pine's non-directional ambiguous entry: do not contaminate Long/Short.
            continue
        side = "Long" if direction == 1 else "Short"
        bucket = stats[side]
        if outcome == 1:
            bucket["wins"] += 1
        elif outcome == -1:
            bucket["losses"] += 1
        elif outcome == 2:
            bucket["ambiguous"] += 1
        elif outcome == 3:
            active.append({
                "dir": direction,
                "side": side,
                "sl": lows[b_idx] if direction == 1 else highs[b_idx],
                "tp": highs[a_idx] if direction == 1 else lows[a_idx],
            })
        # outcome 4 = invalid target, ignored exactly as v7.

    rows = []
    for side in ("Long", "Short"):
        s = stats[side]
        resolved = s["wins"] + s["losses"]
        rows.append({
            "Direction": side,
            "Resolved": resolved,
            "Wins": s["wins"],
            "Losses": s["losses"],
            "Ambiguous": s["ambiguous"],
            "Success %": (100.0 * s["wins"] / resolved) if resolved else np.nan,
        })
    return rows


def scan_one_symbol(
    token, account_id, environment, instrument_row, granularity, history_bars,
    include_color, daily_alignment, alignment_timezone, weekly_alignment,
):
    symbol = instrument_row["name"]
    pip_location = int(instrument_row.get("pipLocation", -4))
    mintick = 10 ** pip_location

    base = fetch_candles(
        token, account_id, environment, symbol, granularity, history_bars,
        daily_alignment, alignment_timezone, weekly_alignment, "M"
    ).reset_index(drop=True)

    closed = base.index[base["complete"].astype(bool)].tolist()
    if len(closed) < 3:
        return []
    last_closed = closed[-1]
    # Ignore any current incomplete candle. Historical engine also ends at last closed.
    base = base.iloc[:last_closed + 1].reset_index(drop=True)
    last_closed = len(base) - 1
    states = compute_states(base, mintick, include_color)

    found = find_latest_watch_setup(states, last_closed)
    if not found:
        return []

    stats_rows = actionable_stats_for_pattern(base, states, found["pattern"], history_bars)
    out = []
    for s in stats_rows:
        out.append({
            "Symbol": symbol,
            "Setup": found["label"],
            "A": found["a_state"],
            "B": found["b_state"],
            **s,
            "B Candle Time": base.loc[last_closed, "time"],
        })
    return out


def main():
    st.set_page_config(page_title="OANDA v7 Actionable Setup Scanner", layout="wide")
    st.title("OANDA v7 Actionable Setup Scanner")
    st.caption(
        "Stage 1 checks only the latest two closed candles for the configured watchlist. "
        "Stage 2 replays v7-style actionable A/B/C outcomes only for matching symbols."
    )

    token0, env0 = get_credentials()
    with st.sidebar:
        st.header("OANDA Connection")
        environment = st.selectbox(
            "Environment", ["Practice", "Live"],
            index=0 if env0.lower().startswith("prac") else 1,
        )
        token = st.text_input("OANDA API token", value=token0, type="password")
        account_id = ""
        if token:
            try:
                account_ids = discover_account_ids(token, environment)
                if len(account_ids) == 1:
                    account_id = account_ids[0]
                    st.caption(f"Account: {account_id}")
                else:
                    account_id = st.selectbox("Authorized account", list(account_ids))
            except Exception as exc:
                st.error(str(exc))

        st.divider()
        st.header("Scanner")
        granularity = st.selectbox(
            "LTF / Scanner Timeframe", GRANULARITIES, index=GRANULARITIES.index("H1")
        )
        history_bars = st.slider(
            "OANDA Candles to Load", 200, 5000, 5000, step=100,
            help="Maximum historical bars used for setup discovery and v7-style outcome statistics."
        )
        use_offset = st.checkbox(
            "Use Closed Bars Only", True, disabled=True,
            help="This scanner is intentionally fixed to the last two CLOSED candles."
        )
        lookback_n = st.slider(
            "Pattern Lookback Window (Candles)", 1, 5, 2, disabled=True,
            help="Fixed to the latest A/B pair. One-candle watch items inspect B only."
        )

        st.divider()
        st.header("Context Filters")
        # Kept for configuration compatibility with the supplied scanner. The new v7
        # actionable engine does not apply these legacy next-bar filters.
        filter_htf = st.checkbox("Filter by HTF 1", False, disabled=True)
        htf_input = st.selectbox("HTF 1", GRANULARITIES, index=GRANULARITIES.index("D"), disabled=True)
        filter_htf2 = st.checkbox("Filter by HTF 2", False, disabled=True)
        htf_input2 = st.selectbox("HTF 2", GRANULARITIES, index=GRANULARITIES.index("W"), disabled=True)
        filter_sr = st.checkbox("Filter by Active S/R Context", False, disabled=True)
        filter_dow = st.checkbox("Filter by Day of Week", False, disabled=True)
        dow_shift_days = st.slider("Day of Week Shift (Days)", -2, 2, 0, disabled=True)
        filter_4h = st.checkbox("Filter by 4-Hour Session", False, disabled=True)
        filter_8h = st.checkbox("Filter by 8-Hour Session", False, disabled=True)

        st.divider()
        st.header("S/R Settings")
        min_touches = st.slider("Minimum Touches", 2, 5, 2, disabled=True)
        sr_lookback = st.slider("S/R Lookback", 50, 500, 200, step=10, disabled=True)
        tolerance_mult = st.number_input("ATR Tolerance Multiplier", 0.01, 2.0, 0.20, step=0.01, disabled=True)
        min_wick_mult = st.number_input("Minimum Wick ATR Multiplier", 0.0, 5.0, 0.0, step=0.05, disabled=True)
        invalidation = st.selectbox("S/R Invalidation", ["Close", "Wick"], disabled=True)

        st.divider()
        st.header("OANDA Candle Alignment")
        daily_alignment = st.number_input("Daily alignment hour", 0, 23, 17)
        alignment_timezone = st.text_input("Alignment timezone", "America/New_York")
        weekly_alignment = st.selectbox(
            "Weekly alignment",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=4,
        )

        st.divider()
        max_workers = st.slider(
            "Parallel Symbols", 1, 50, 20,
            help="Lower this if OANDA returns rate-limit/network errors."
        )
        min_sample = st.number_input(
            "Minimum Sample Size to Display", min_value=0, max_value=5000, value=0, step=1
        )
        sort_by = st.selectbox(
            "Sort Results By", ["Symbol", "Resolved", "Success %"]
        )
        include_color = st.checkbox("Use G/R in 2 and 3 vocabulary", True)
        scan = st.button("Scan All OANDA Forex Symbols", type="primary", width="stretch")
        refresh = st.button("Clear OANDA Cache", width="stretch")

    if refresh:
        fetch_candles.clear()
        fetch_instruments.clear()
        discover_account_ids.clear()
        st.rerun()

    if not token:
        st.info("Enter your OANDA API token in the sidebar.")
        st.stop()
    if not account_id:
        st.info("No OANDA account could be discovered from this token.")
        st.stop()

    try:
        instruments_df = fetch_instruments(token, account_id, environment)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    forex_df = instruments_df[
        instruments_df["type"].astype(str).str.upper().eq("CURRENCY")
    ].copy().reset_index(drop=True)
    st.write(f"Forex symbols available from this OANDA account: **{len(forex_df)}**")
    st.caption("Watchlist: " + ", ".join(" → ".join(x) for x in SCAN_PATTERNS))

    if not scan:
        st.info("Configure the scanner, then click **Scan All OANDA Forex Symbols**.")
        st.stop()

    rows, errors = [], []
    progress = st.progress(0, text="Starting scan…")
    total_symbols = len(forex_df)

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        futures = {}
        for _, instrument_row in forex_df.iterrows():
            d = instrument_row.to_dict()
            f = executor.submit(
                scan_one_symbol,
                token, account_id, environment, d, granularity, int(history_bars),
                include_color, int(daily_alignment), alignment_timezone, weekly_alignment,
            )
            futures[f] = d["name"]

        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append({"Symbol": symbol, "Error": str(exc)})
            progress.progress(completed / max(total_symbols, 1), text=f"Scanning {completed}/{total_symbols} — {symbol}")
    progress.empty()

    if not rows:
        st.warning("No latest closed setup matched SCAN_PATTERNS.")
        if errors:
            with st.expander(f"Scan errors ({len(errors)})"):
                st.dataframe(pd.DataFrame(errors).astype(str), hide_index=True, width="stretch")
        st.stop()

    result_df = pd.DataFrame(rows)
    result_df = result_df[result_df["Resolved"] >= int(min_sample)].copy()
    if sort_by == "Symbol":
        result_df = result_df.sort_values(["Symbol", "Direction"])
    elif sort_by == "Resolved":
        result_df = result_df.sort_values(["Resolved", "Success %"], ascending=[False, False], na_position="last")
    else:
        result_df = result_df.sort_values(["Success %", "Resolved"], ascending=[False, False], na_position="last")
    result_df = result_df.reset_index(drop=True)

    display_df = result_df.copy()
    display_df["Success %"] = display_df["Success %"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}%")
    display_df["B Candle Time"] = pd.to_datetime(display_df["B Candle Time"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")

    st.subheader("Latest actionable setups — all OANDA Forex symbols")
    st.dataframe(display_df, hide_index=True, width="stretch", height=min(1000, 38 * (len(display_df) + 1)))
    st.caption(
        "Resolved = Wins + Losses. Ambiguous outcomes are excluded from Success %. "
        "Invalid targets are ignored. Open historical trades remain unresolved."
    )

    if errors:
        with st.expander(f"Scan errors ({len(errors)})"):
            st.dataframe(pd.DataFrame(errors).astype(str), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()

from collections import Counter
from datetime import datetime, timezone
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Strat Candle Probability Tracker", layout="wide"
)

# ---------------------------------------------------------
# OANDA TICKER & TIME FRAME MAPPING
# ---------------------------------------------------------
TICKER_MAPPING = {
    "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD",
    "AUD/USD": "AUD_USD",
    "NZD/USD": "NZD_USD",
    "USD/CAD": "USD_CAD",
    "USD/CHF": "USD_CHF",
    "USD/JPY": "USD_JPY",
    "USD/SGD": "USD_SGD",
    "Gold (XAU/USD)": "XAU_USD",
    "Silver (XAG/USD)": "XAG_USD",
    "Crude Oil (BCO/USD)": "BCO_USD",
    "Bitcoin (BTC/USD)": "BTC_USD",
    "EUR/GBP": "EUR_GBP",
    "EUR/JPY": "EUR_JPY",
    "GBP/JPY": "GBP_JPY",
    "AUD/JPY": "AUD_JPY",
}

GRANULARITY_MAP = {
    "15 Minutes": "M15",
    "1 Hour": "H1",
    "4 Hours": "H4",
    "1 Day": "D",
    "1 Week": "W",
    "1 Month": "M",
}

HTF_OPTIONS_MAP = {
    "15 Minutes": ["1 Hour", "4 Hours", "1 Day", "1 Week"],
    "1 Hour": ["4 Hours", "1 Day", "1 Week", "1 Month"],
    "4 Hours": ["1 Day", "1 Week", "1 Month"],
    "1 Day": ["1 Week", "1 Month"],
    "1 Week": ["1 Month"],
}

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("OANDA API Credentials")

# Attempt secret retrieval if available
secret_token = st.secrets.get("oanda_api_token", "")
secret_env = st.secrets.get("oanda_env", "Practice")

if secret_token:
    st.sidebar.success("🔒 API Token loaded from Streamlit Secrets")
    api_token = secret_token
    oanda_env = secret_env
else:
    api_token = st.sidebar.text_input(
        "OANDA API Token", type="password", value=""
    )
    oanda_env = st.sidebar.selectbox(
        "Environment", ["Practice", "Live"], index=0
    )

st.sidebar.markdown("---")
st.sidebar.header("Data & Timeframe Settings")

selected_symbol = st.sidebar.selectbox(
    "Symbol", options=list(TICKER_MAPPING.keys()), index=0
)
execution_tf = st.sidebar.selectbox(
    "Selected Execution Timeframe",
    options=list(GRANULARITY_MAP.keys()),
    index=3,
)

available_htfs = HTF_OPTIONS_MAP.get(execution_tf, ["1 Month"])
htf_context_tf = st.sidebar.selectbox(
    "Higher Timeframe Context (HTF)", options=available_htfs, index=0
)

current_year = datetime.now().year
start_year = st.sidebar.selectbox(
    "Historical Start Year",
    options=list(range(2005, current_year + 1)),
    index=15,  # Defaults to 2020
    help="Fetches data from Jan 1 of this year through current date.",
)

st.sidebar.markdown("---")
st.sidebar.header("Pattern Analysis Parameters")

lookback_n = st.sidebar.slider(
    "Pattern Window (N Candles)",
    min_value=1,
    max_value=5,
    value=3,
    help="Number of consecutive prior candles matched in historical sequence.",
)
include_live_bar = st.sidebar.checkbox(
    "Include Unclosed (Live) Candle", value=False
)
filter_by_htf = st.sidebar.checkbox(
    f"Filter by HTF Context ({htf_context_tf})", value=True
)


# ---------------------------------------------------------
# THREAD-SAFE PAGINATED OANDA FETCHER
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_oanda_data_paginated(symbol_key, timeframe_key, start_yr, token, env):
    """Fetches candles directly for the user-selected timeframe from OANDA

    using forward-chunked pagination (up to 5,000 candles per batch).
    """
    if not token:
        return None, "OANDA API Token is missing."

    inst = TICKER_MAPPING.get(symbol_key, "EUR_USD")
    gran = GRANULARITY_MAP.get(timeframe_key, "D")
    domain = (
        "api-fxtrade.oanda.com" if env == "Live" else "api-fxpractice.oanda.com"
    )
    url = f"https://{domain}/v3/instruments/{inst}/candles"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    from_time = f"{start_yr}-01-01T00:00:00Z"
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_candles = []
    max_batches = 30  # Safety cap (up to 150,000 candles per timeframe)
    batch_count = 0

    while batch_count < max_batches:
        params = {
            "price": "M",
            "granularity": gran,
            "count": 5000,
            "from": from_time,
        }
        try:
            res = requests.get(
                url, headers=headers, params=params, timeout=20
            )
            if res.status_code != 200:
                return (
                    None,
                    f"OANDA API returned HTTP {res.status_code}: {res.text}",
                )

            data = res.json()
            candles = data.get("candles", [])
            if not candles:
                break

            all_candles.extend(candles)

            # Advance timestamp by 1 second to prevent fetching duplicate boundary bar
            last_time_str = candles[-1]["time"]
            last_dt = pd.to_datetime(last_time_str)
            next_from_time = (last_dt + pd.Timedelta(seconds=1)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

            if (
                len(candles) < 5000
                or next_from_time >= now_iso
                or next_from_time <= from_time
            ):
                break

            from_time = next_from_time
            batch_count += 1
        except Exception as e:
            return None, f"Connection error: {str(e)}"

    if not all_candles:
        return None, f"No candles found for {symbol_key} ({timeframe_key})."

    rows = [
        {
            "Date": pd.to_datetime(c["time"]),
            "Open": float(c["mid"]["o"]),
            "High": float(c["mid"]["h"]),
            "Low": float(c["mid"]["l"]),
            "Close": float(c["mid"]["c"]),
            "Complete": c.get("complete", True),
        }
        for c in all_candles
    ]

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["Date"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    return df, None


# ---------------------------------------------------------
# PINE SCRIPT STRAT ENGINE
# ---------------------------------------------------------
def classify_strat_candles(df):
    """Classifies OHLC candles into pure Pine Script Strat types:

    Type 1  (Inside):  High <= Prev High and Low >= Prev Low
    Type 2U (Up):      High >  Prev High and Low >= Prev Low
    Type 2D (Down):    High <= Prev High and Low <  Prev Low
    Type 3  (Outside): High >  Prev High and Low <  Prev Low
    Arrows: Close >= Open -> ↑, Close < Open -> ↓
    """
    if df is None or len(df) < 2:
        return pd.DataFrame()

    states = []
    dates = []

    for i in range(1, len(df)):
        h_prev, l_prev = df["High"].iloc[i - 1], df["Low"].iloc[i - 1]
        h_curr, l_curr = df["High"].iloc[i], df["Low"].iloc[i]
        c_curr, o_curr = df["Close"].iloc[i], df["Open"].iloc[i]

        hh = h_curr > h_prev
        ll = l_curr < l_prev

        if hh and ll:
            num = "3"
        elif hh and not ll:
            num = "2U"
        elif not hh and ll:
            num = "2D"
        else:
            num = "1"

        arrow = "↑" if c_curr >= o_curr else "↓"
        states.append(f"{num} {arrow}")
        dates.append(df.index[i])

    return pd.DataFrame({"Date": pd.to_datetime(dates), "State": states})


def calculate_probabilities(
    exec_states, htf_states, n_back, filter_htf_enabled
):
    if exec_states.empty or len(exec_states) <= n_back:
        return [], {}, 0, "N/A"

    merged = exec_states.sort_values("Date").copy()
    current_htf_state = "N/A"

    if filter_htf_enabled and not htf_states.empty:
        merged = pd.merge_asof(
            merged,
            htf_states.sort_values("Date"),
            on="Date",
            direction="backward",
            suffixes=("", "_HTF"),
        )
        if "State_HTF" in merged.columns:
            current_htf_state = merged["State_HTF"].iloc[-1]

    states = merged["State"].tolist()
    htf_list = (
        merged["State_HTF"].tolist() if "State_HTF" in merged.columns else []
    )

    current_pattern = states[-n_back:]
    next_states = []

    for i in range(len(states) - n_back):
        window = states[i : i + n_back]
        pattern_match = window == current_pattern

        context_match = True
        if filter_htf_enabled and current_htf_state != "N/A" and htf_list:
            context_match = htf_list[i + n_back - 1] == current_htf_state

        if pattern_match and context_match:
            if i + n_back < len(states):
                next_states.append(states[i + n_back])

    if not next_states:
        return current_pattern, {}, 0, current_htf_state

    total = len(next_states)
    counts = Counter(next_states)
    probs = {
        k: round((v / total) * 100, 2)
        for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)
    }

    return current_pattern, probs, total, current_htf_state


# ---------------------------------------------------------
# MAIN APPLICATION INTERFACE
# ---------------------------------------------------------
st.title("📈 Strat Candle Probability Tracker")

if not api_token:
    st.info(
        "👋 Please enter your **OANDA API Token** in the sidebar to load market data."
    )
    st.stop()

# Fetch selected timeframe and HTF context
with st.spinner(
    f"Fetching historical OANDA data for {selected_symbol} ({execution_tf}) since {start_year}..."
):
    df_exec, err_exec = fetch_oanda_data_paginated(
        selected_symbol, execution_tf, start_year, api_token, oanda_env
    )
    df_htf, err_htf = fetch_oanda_data_paginated(
        selected_symbol, htf_context_tf, start_year, api_token, oanda_env
    )

if err_exec:
    st.error(f"Error fetching execution timeframe data: {err_exec}")
    st.stop()

if err_htf:
    st.warning(
        f"Warning regarding HTF context data: {err_htf}. Proceeding with execution TF only."
    )

# Filter out live/incomplete bars if unselected
if not include_live_bar:
    if df_exec is not None and "Complete" in df_exec.columns:
        df_exec = df_exec[df_exec["Complete"] == True]
    if df_htf is not None and "Complete" in df_htf.columns:
        df_htf = df_htf[df_htf["Complete"] == True]

# Classify structures
exec_classified = classify_strat_candles(df_exec)
htf_classified = (
    classify_strat_candles(df_htf) if df_htf is not None else pd.DataFrame()
)

if exec_classified.empty:
    st.error("Insufficient candles to evaluate Strat structures.")
    st.stop()

# Calculate probabilities
current_pattern, probabilities, total_matches, active_htf = (
    calculate_probabilities(
        exec_classified, htf_classified, lookback_n, filter_by_htf
    )
)

# Display Summary Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Loaded Candles", f"{len(exec_classified):,}")
m2.metric("Current Pattern Sequence", " ➔ ".join(current_pattern))
m3.metric(f"Active HTF ({htf_context_tf})", active_htf)
m4.metric("Matching Sequence Occurrences", total_matches)

st.markdown("---")

# Display Probabilities
if total_matches > 0:
    st.subheader("📊 Probabilities for Next Candle")

    prob_df = pd.DataFrame(
        list(probabilities.items()),
        columns=["Next Candle Structure", "Probability (%)"],
    )

    col_tbl, col_cht = st.columns([1, 1])

    with col_tbl:
        st.markdown("##### Detailed Breakdown")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with col_cht:
        st.markdown("##### Structure Distribution")
        st.bar_chart(
            prob_df.set_index("Next Candle Structure")["Probability (%)"]
        )

    # Aggregates (Up vs Down, Structure Breakdown)
    up_prob = sum(
        v
        for k, v in probabilities.items()
        if "↑" in k or k.startswith("2U")
    )
    down_prob = sum(
        v
        for k, v in probabilities.items()
        if "↓" in k or k.startswith("2D")
    )

    struct_totals = {"2U (Up)": 0.0, "2D (Down)": 0.0, "1 (Inside)": 0.0, "3 (Outside)": 0.0}
    for k, v in probabilities.items():
        if k.startswith("2U"):
            struct_totals["2U (Up)"] += v
        elif k.startswith("2D"):
            struct_totals["2D (Down)"] += v
        elif k.startswith("1"):
            struct_totals["1 (Inside)"] += v
        elif k.startswith("3"):
            struct_totals["3 (Outside)"] += v

    dir_df = pd.DataFrame(
        [
            {"Direction": "Bullish / Up (↑)", "Probability (%)": round(up_prob, 2)},
            {"Direction": "Bearish / Down (↓)", "Probability (%)": round(down_prob, 2)},
        ]
    )

    struct_df = pd.DataFrame(
        [
            {"Structure": k, "Probability (%)": round(v, 2)}
            for k, v in struct_totals.items()
        ]
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Directional Probability (Up vs Down)")
        st.dataframe(dir_df, use_container_width=True, hide_index=True)
        st.bar_chart(dir_df.set_index("Direction")["Probability (%)"])

    with c2:
        st.markdown("##### Strat Structure Type Probability")
        st.dataframe(struct_df, use_container_width=True, hide_index=True)
        st.bar_chart(struct_df.set_index("Structure")["Probability (%)"])
else:
    st.warning(
        f"No historical matches found for the sequence {' ➔ '.join(current_pattern)} under active HTF context '{active_htf}'."
    )

with st.expander("🔍 View Recent Classified Candle History"):
    st.dataframe(
        exec_classified.tail(50).sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

from collections import Counter, defaultdict
from datetime import datetime, timezone
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pine Script Strat Probability Tracker", layout="wide"
)

# ---------------------------------------------------------
# OANDA TICKER & TIMEFRAME MAPPING
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

HTF_OPTIONS = ["None", "1 Hour", "4 Hours", "1 Day", "1 Week", "1 Month"]

# ---------------------------------------------------------
# SIDEBAR CONTROLS & PINE SCRIPT FILTERS
# ---------------------------------------------------------
st.sidebar.header("OANDA API Credentials")

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
    "Execution Timeframe", options=list(GRANULARITY_MAP.keys()), index=3
)

current_year = datetime.now().year
start_year = st.sidebar.selectbox(
    "Historical Start Year",
    options=list(range(2005, current_year + 1)),
    index=15,  # Defaults to ~2020
    help="Fetches OANDA data from present day backward until Jan 1 of this year.",
)

st.sidebar.markdown("---")
st.sidebar.header("🌲 Pine Script Strat Filters")

match_mode = st.sidebar.radio(
    "Pattern Match Mode",
    ["Exact (Number + Arrow Direction)", "Structure Only (Numbers Only)"],
    index=0,
    help="Exact matches '2U ↑', whereas Structure Only matches '2U' regardless of candle color.",
)

setup_filter = st.sidebar.selectbox(
    "Strat Setup Filter",
    ["All Patterns", "Reversals Only (e.g. 2-2 / 1-2 / 3-2)", "Continuations Only (e.g. 2U-2U / 2D-2D)"],
    index=0,
)

st.sidebar.subheader("Full Timeframe Continuity (FTFC)")
htf_1 = st.sidebar.selectbox("Primary HTF Context", HTF_OPTIONS, index=4)  # 1 Week
htf_2 = st.sidebar.selectbox("Secondary HTF Context", HTF_OPTIONS, index=5)  # 1 Month

lookback_n = st.sidebar.slider(
    "Pattern Window (N Candles)", min_value=1, max_value=5, value=3
)
min_sample_cutoff = st.sidebar.slider(
    "Min Historical Occurrences Cutoff", min_value=1, max_value=20, value=1
)
include_live_bar = st.sidebar.checkbox(
    "Include Unclosed (Live) Candle", value=False
)


# ---------------------------------------------------------
# BACKWARD PAGINATED OANDA DATA ENGINE
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_oanda_data_paginated(symbol_key, timeframe_key, start_yr, token, env):
    """Fetches complete data from present day back to start_yr by looping backwards

    using clean RFC3339 timestamps.
    """
    if not token or timeframe_key == "None":
        return None, None

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

    target_start_iso = f"{start_yr}-01-01T00:00:00Z"
    to_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_candles = []
    max_batches = 60
    batch_count = 0

    while batch_count < max_batches:
        params = {
            "price": "M",
            "granularity": gran,
            "count": 5000,
            "to": to_time,
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

            # Prepend chronologically
            all_candles = candles + all_candles

            oldest_time_str = candles[0]["time"]
            oldest_dt = pd.to_datetime(oldest_time_str)
            oldest_iso = oldest_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            if oldest_iso <= target_start_iso or len(candles) < 5000:
                break

            to_time = (oldest_dt - pd.Timedelta(seconds=1)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
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
    """Classifies OHLC candles into pure Pine Script Strat types."""
    if df is None or len(df) < 2:
        return pd.DataFrame()

    states, struct_nums, dates = [], [], []

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
        struct_nums.append(num)
        dates.append(df.index[i])

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "State": states,
            "StructNum": struct_nums,
        }
    )


def is_reversal(state1, state2):
    s1 = state1.split()[0]
    s2 = state2.split()[0]
    if s1 in ["2D", "1", "3"] and s2 == "2U":
        return True
    if s1 in ["2U", "1", "3"] and s2 == "2D":
        return True
    return False


def is_continuation(state1, state2):
    s1 = state1.split()[0]
    s2 = state2.split()[0]
    return (s1 == "2U" and s2 == "2U") or (s1 == "2D" and s2 == "2D")


def calculate_strat_probabilities(
    exec_df,
    htf1_df,
    htf2_df,
    n_back,
    match_mode_str,
    setup_type_str,
    min_cutoff,
):
    if exec_df.empty or len(exec_df) <= n_back:
        return [], {}, 0, "N/A", "N/A"

    merged = exec_df.sort_values("Date").copy()

    # Merge Primary HTF Context
    curr_htf1 = "N/A"
    if htf1_df is not None and not htf1_df.empty:
        merged = pd.merge_asof(
            merged,
            htf1_df.sort_values("Date"),
            on="Date",
            direction="backward",
            suffixes=("", "_HTF1"),
        )
        if "State_HTF1" in merged.columns:
            curr_htf1 = merged["State_HTF1"].iloc[-1]

    # Merge Secondary HTF Context
    curr_htf2 = "N/A"
    if htf2_df is not None and not htf2_df.empty:
        merged = pd.merge_asof(
            merged,
            htf2_df.sort_values("Date"),
            on="Date",
            direction="backward",
            suffixes=("", "_HTF2"),
        )
        if "State_HTF2" in merged.columns:
            curr_htf2 = merged["State_HTF2"].iloc[-1]

    use_exact = "Exact" in match_mode_str
    target_series = merged["State"] if use_exact else merged["StructNum"]
    states_list = target_series.tolist()

    current_pattern = states_list[-n_back:]
    next_states = []

    htf1_list = (
        merged["State_HTF1"].tolist() if "State_HTF1" in merged.columns else []
    )
    htf2_list = (
        merged["State_HTF2"].tolist() if "State_HTF2" in merged.columns else []
    )

    for i in range(len(states_list) - n_back):
        window = states_list[i : i + n_back]
        if window != current_pattern:
            continue

        # Check Setup Filter (Reversal vs Continuation)
        if setup_type_str.startswith("Reversals"):
            if not is_reversal(merged["State"].iloc[i + n_back - 2], merged["State"].iloc[i + n_back - 1]):
                continue
        elif setup_type_str.startswith("Continuations"):
            if not is_continuation(merged["State"].iloc[i + n_back - 2], merged["State"].iloc[i + n_back - 1]):
                continue

        # Check FTFC Constraints
        if curr_htf1 != "N/A" and htf1_list:
            if htf1_list[i + n_back - 1] != curr_htf1:
                continue
        if curr_htf2 != "N/A" and htf2_list:
            if htf2_list[i + n_back - 1] != curr_htf2:
                continue

        if i + n_back < len(states_list):
            next_states.append(merged["State"].iloc[i + n_back])

    if not next_states or len(next_states) < min_cutoff:
        return current_pattern, {}, len(next_states), curr_htf1, curr_htf2

    total = len(next_states)
    counts = Counter(next_states)
    probs = {
        k: round((v / total) * 100, 2)
        for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)
    }

    return current_pattern, probs, total, curr_htf1, curr_htf2


# ---------------------------------------------------------
# MAIN DASHBOARD UI
# ---------------------------------------------------------
st.title("🌲 Pine Script Strat Probability Tracker")

if not api_token:
    st.info("👋 Enter your **OANDA API Token** in the sidebar to start.")
    st.stop()

with st.spinner(
    f"Loading OANDA data for {selected_symbol} ({execution_tf}) back to {start_year}..."
):
    df_exec, err_exec = fetch_oanda_data_paginated(
        selected_symbol, execution_tf, start_year, api_token, oanda_env
    )
    df_htf1, _ = fetch_oanda_data_paginated(
        selected_symbol, htf_1, start_year, api_token, oanda_env
    )
    df_htf2, _ = fetch_oanda_data_paginated(
        selected_symbol, htf_2, start_year, api_token, oanda_env
    )

if err_exec or df_exec is None:
    st.error(f"Error loading execution data: {err_exec}")
    st.stop()

if not include_live_bar:
    if "Complete" in df_exec.columns:
        df_exec = df_exec[df_exec["Complete"] == True]
    if df_htf1 is not None and "Complete" in df_htf1.columns:
        df_htf1 = df_htf1[df_htf1["Complete"] == True]
    if df_htf2 is not None and "Complete" in df_htf2.columns:
        df_htf2 = df_htf2[df_htf2["Complete"] == True]

exec_classified = classify_strat_candles(df_exec)
htf1_classified = (
    classify_strat_candles(df_htf1) if df_htf1 is not None else pd.DataFrame()
)
htf2_classified = (
    classify_strat_candles(df_htf2) if df_htf2 is not None else pd.DataFrame()
)

current_pattern, probabilities, total_matches, active_htf1, active_htf2 = (
    calculate_strat_probabilities(
        exec_classified,
        htf1_classified,
        htf2_classified,
        lookback_n,
        match_mode,
        setup_filter,
        min_sample_cutoff,
    )
)

# Date Verification Banner
earliest_loaded = df_exec.index[0].strftime("%Y-%m-%d")
latest_loaded = df_exec.index[-1].strftime("%Y-%m-%d")
st.success(
    f"Loaded **{len(df_exec):,} candles** spanning from **{earliest_loaded}** to **{latest_loaded}**."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Pattern Sequence", " ➔ ".join(current_pattern))
m2.metric(f"HTF 1 ({htf_1}) Context", active_htf1)
m3.metric(f"HTF 2 ({htf_2}) Context", active_htf2)
m4.metric("Matching Sequence Samples", total_matches)

st.markdown("---")

if total_matches >= min_sample_cutoff and probabilities:
    st.subheader("📊 Probabilities for Next Strat Candle")

    prob_df = pd.DataFrame(
        list(probabilities.items()),
        columns=["Next Candle Structure", "Probability (%)"],
    )

    c_tbl, c_cht = st.columns([1, 1])
    with c_tbl:
        st.markdown("##### Detailed Breakdown")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
    with c_cht:
        st.markdown("##### Structure Distribution")
        st.bar_chart(
            prob_df.set_index("Next Candle Structure")["Probability (%)"]
        )

    # Aggregates
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
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Directional Probability (Up vs Down)")
        st.dataframe(dir_df, use_container_width=True, hide_index=True)
        st.bar_chart(dir_df.set_index("Direction")["Probability (%)"])
    with col2:
        st.markdown("##### Strat Structure Breakdown")
        st.dataframe(struct_df, use_container_width=True, hide_index=True)
        st.bar_chart(struct_df.set_index("Structure")["Probability (%)"])

else:
    st.warning(
        f"No historical matches found meeting all active Pine Script filters (Sample cutoff: {min_sample_cutoff}). Try broadening your filters."
    )

with st.expander("🔍 View Recent Classified Candle History"):
    st.dataframe(
        exec_classified.tail(50).sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

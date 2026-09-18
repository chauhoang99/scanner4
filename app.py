from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="Strat Candle Probability Tracker", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# LOAD OANDA SECRETS (IF AVAILABLE)
# ---------------------------------------------------------
try:
    secret_token = st.secrets.get("oanda_api_token", "")
    secret_account = st.secrets.get("oanda_account_id", "")
    secret_env = st.secrets.get("oanda_env", "Practice")
except Exception:
    secret_token, secret_account, secret_env = "", "", "Practice"

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.header("Data Provider & Settings")
data_source = st.sidebar.radio("Data Source", ["OANDA API", "Yahoo Finance (yfinance)"], index=0)

if data_source == "OANDA API":
    if secret_token:
        st.sidebar.success("🔒 Oanda Token loaded from Streamlit Secrets")
        api_token = secret_token
        oanda_env = secret_env
    else:
        api_token = st.sidebar.text_input("Oanda API Token", type="password", value="")
        oanda_env = st.sidebar.selectbox("Environment", ["Practice", "Live"], index=0)
else:
    api_token, oanda_env = "", "Practice"

st.sidebar.markdown("---")
st.sidebar.header("Structure Tracker Settings")

# Ticker List with OANDA Mapping
ticker_mapping = {
    "EURUSD=X": "EUR_USD", "GBPUSD=X": "GBP_USD", "AUDUSD=X": "AUD_USD",
    "NZDUSD=X": "NZD_USD", "USDCAD=X": "USD_CAD", "USDCHF=X": "USD_CHF",
    "USDJPY=X": "USD_JPY", "USDSGD=X": "USD_SGD", "GC=F": "XAU_USD",
    "BZ=F": "BCO_USD", "ZB=F": "USB30Y_USD", "BTC-USD": "BTC_USD",
    "EURGBP=X": "EUR_GBP", "EURAUD=X": "EUR_AUD", "EURNZD=X": "EUR_NZD",
    "EURCAD=X": "EUR_CAD", "EURCHF=X": "EUR_CHF", "EURJPY=X": "EUR_JPY",
    "EURSGD=X": "EUR_SGD", "XAUEUR=X": "XAU_EUR", "GBPAUD=X": "GBP_AUD",
    "GBPNZD=X": "GBP_NZD", "GBPCAD=X": "GBP_CAD", "GBPCHF=X": "GBP_CHF",
    "GBPJPY=X": "GBP_JPY", "GBPSGD=X": "GBP_SGD", "AUDNZD=X": "AUD_NZD",
    "AUDCAD=X": "AUD_CAD", "AUDCHF=X": "AUD_CHF", "AUDJPY=X": "AUD_JPY",
    "AUDSGD=X": "AUD_SGD", "AAPL": "AAPL", "MSFT": "MSFT", "SPY": "SPY", "QQQ": "QQQ"
}

ticker_options = list(ticker_mapping.keys())
symbol = st.sidebar.selectbox("Ticker Symbol", options=ticker_options, index=0, help="Select a ticker symbol from the list.")

st.sidebar.subheader("Timeframes & History")

granularity_map = {
    "60m": "H1", "1d": "D", "1wk": "W", "1mo": "M"
}

# Configurable Execution Timeframe
timeframe = st.sidebar.selectbox("Execution Timeframe", ["60m", "1d", "1wk"], index=1)

# Dynamically map valid Higher Timeframes based on selected Execution Timeframe
htf_options_map = {
    "60m": ["1d", "1wk", "1mo"],
    "1d": ["1wk", "1mo"],
    "1wk": ["1mo"]
}
htf_timeframe = st.sidebar.selectbox("Higher Timeframe (Context)", htf_options_map[timeframe], index=0)

if data_source == "OANDA API":
    sample_count = st.sidebar.slider("Historical Candle Count", 100, 4000, 1000, step=100)
    history_period = "2y"
else:
    history_period = st.sidebar.selectbox("History Range", ["1y", "2y", "5y", "10y", "max"], index=1)
    sample_count = 1000

lookback_n = st.sidebar.slider("Pattern Lookback Window (Candles)", min_value=1, max_value=5, value=3, help="Number of past consecutive candle structures to match historically.")
include_live_bar = st.sidebar.checkbox("Include Live (Unclosed) Candle", value=False, help="When unchecked, current forming bar is excluded.")
filter_by_htf = st.sidebar.checkbox(f"Filter Probabilities by Active HTF ({htf_timeframe}) Context", value=True, help="Only match patterns that occurred under the same Higher Timeframe Strat candle state.")
show_all_patterns = st.sidebar.checkbox("Show All Historical Patterns Summary", value=False)

if st.sidebar.button("🔄 Run Analysis"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_oanda_data(symbol_name, gran, count, token, env):
    inst = ticker_mapping.get(symbol_name, symbol_name.replace("=X", "").replace("=", "_"))
    domain = "api-fxtrade.oanda.com" if env == "Live" else "api-fxpractice.oanda.com"
    url = f"https://{domain}/v3/instruments/{inst}/candles"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    params = {"price": "M", "granularity": gran, "count": min(count, 5000)}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=15)
        if res.status_code == 200:
            candles = res.json().get("candles", [])
            rows = [{
                "Date": pd.to_datetime(c["time"]),
                "Open": float(c["mid"]["o"]),
                "High": float(c["mid"]["h"]),
                "Low": float(c["mid"]["l"]),
                "Close": float(c["mid"]["c"]),
                "Complete": c.get("complete", True)
            } for c in candles]
            df = pd.DataFrame(rows)
            if not df.empty:
                df.set_index("Date", inplace=True)
            return df
    except Exception:
        return None
    return None

@st.cache_data(ttl=300)
def fetch_yf_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return None

# ---------------------------------------------------------
# UNIFIED STRAT CANDLE CLASSIFICATION ENGINE
# ---------------------------------------------------------
def get_candle_structure_series(df):
    if df is None or len(df) < 2:
        return pd.DataFrame()

    states = []
    dates = []

    for i in range(1, len(df)):
        h_prev = df["High"].iloc[i-1]
        l_prev = df["Low"].iloc[i-1]
        h_curr = df["High"].iloc[i]
        l_curr = df["Low"].iloc[i]
        c_curr = df["Close"].iloc[i]
        o_curr = df["Open"].iloc[i]

        higher_high = h_curr > h_prev
        lower_low = l_curr < l_prev

        # 1. Strat Candle Structure Type
        if higher_high and lower_low:
            num = "3"
        elif higher_high and not lower_low:
            num = "2U"
        elif not higher_high and lower_low:
            num = "2D"
        else:
            num = "1"

        # 2. Candle Color Direction (Close vs Open)
        arrow = "↑" if c_curr >= o_curr else "↓"

        # 3. Solid Triangle (In-Force Holding Condition)
        if c_curr > h_prev:
            triangle = "▲"
        elif c_curr < l_prev:
            triangle = "▼"
        else:
            triangle = ""

        # Universal Output Formatting: [Type] [Arrow] [Triangle]
        state = f"{num} {arrow} {triangle}".strip()

        states.append(state)
        dates.append(df.index[i])

    return pd.DataFrame({'Date': pd.to_datetime(dates), 'State': states})

def calculate_next_state_probabilities(state_df, htf_df=None, n_back=3, use_htf_context=True):
    if state_df.empty or len(state_df) <= n_back:
        return [], {}, 0, "N/A"

    df_merged = state_df.sort_values("Date").copy()

    current_htf_context = "N/A"
    if use_htf_context and htf_df is not None and not htf_df.empty:
        df_merged = pd.merge_asof(
            df_merged,
            htf_df.sort_values("Date"),
            on="Date",
            direction="backward",
            suffixes=("", "_HTF")
        )
        if "State_HTF" in df_merged.columns:
            current_htf_context = df_merged["State_HTF"].iloc[-1]

    states = df_merged['State'].tolist()
    htf_states = df_merged['State_HTF'].tolist() if "State_HTF" in df_merged.columns else []
    
    current_pattern = states[-n_back:]

    next_states = []
    for i in range(len(states) - n_back):
        window = states[i:i+n_back]
        
        # Condition check: Match LTF pattern AND matching HTF Context (if enabled)
        pattern_match = (window == current_pattern)
        context_match = True
        
        if use_htf_context and current_htf_context != "N/A" and htf_states:
            context_at_time = htf_states[i + n_back - 1]
            context_match = (context_at_time == current_htf_context)

        if pattern_match and context_match:
            if i + n_back < len(states):
                next_states.append(states[i+n_back])

    if not next_states:
        return current_pattern, {}, 0, current_htf_context

    total_matches = len(next_states)
    counts = Counter(next_states)
    probabilities = {s: (c / total_matches) * 100 for s, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)}

    return current_pattern, probabilities, total_matches, current_htf_context

def get_all_patterns_summary(state_df, n_back=3):
    if state_df.empty or len(state_df) <= n_back:
        return pd.DataFrame()

    states = state_df['State'].tolist()
    pattern_transitions = defaultdict(list)

    for i in range(len(states) - n_back):
        window = tuple(states[i:i+n_back])
        pattern_transitions[window].append(states[i+n_back])

    summary_data = []
    for pattern, subsequent in pattern_transitions.items():
        total_occurrences = len(subsequent)
        counts = Counter(subsequent)
        most_common_next, most_common_count = counts.most_common(1)[0]
        most_common_pct = (most_common_count / total_occurrences) * 100

        summary_data.append({
            "Pattern Sequence": " ➔ ".join(pattern),
            "Total Sample Occurrences": total_occurrences,
            "Most Common Next Structure": most_common_next,
            "Probability": f"{most_common_pct:.1f}%"
        })

    return pd.DataFrame(summary_data).sort_values(by="Total Sample Occurrences", ascending=False)

# ---------------------------------------------------------
# MAIN DASHBOARD UI
# ---------------------------------------------------------
st.title("📈 Strat Candle Probability Tracker")

if data_source == "OANDA API":
    if not api_token:
        st.warning("⚠️ OANDA API token missing. Please enter it in the sidebar or switch data source to Yahoo Finance.")
        st.stop()
    raw_df = fetch_oanda_data(symbol, granularity_map.get(timeframe, "D"), sample_count, api_token, oanda_env)
    raw_htf_df = fetch_oanda_data(symbol, granularity_map.get(htf_timeframe, "W"), max(100, int(sample_count / 5)), api_token, oanda_env)
else:
    raw_df = fetch_yf_data(symbol, history_period, timeframe)
    raw_htf_df = fetch_yf_data(symbol, history_period, htf_timeframe)

if raw_df is None or raw_df.empty:
    st.error(f"Could not retrieve data for '{symbol}'. Please check your token or symbol configuration.")
else:
    df = raw_df.copy()
    htf_df = raw_htf_df.copy() if raw_htf_df is not None else None

    if not include_live_bar:
        if len(df) > 1:
            if "Complete" in df.columns:
                df = df[df["Complete"] == True]
            else:
                df = df.iloc[:-1]
        if htf_df is not None and len(htf_df) > 1:
            if "Complete" in htf_df.columns:
                htf_df = htf_df[htf_df["Complete"] == True]
            else:
                htf_df = htf_df.iloc[:-1]

    state_history = get_candle_structure_series(df)
    htf_state_history = get_candle_structure_series(htf_df) if htf_df is not None else pd.DataFrame()

    if state_history.empty:
        st.warning("Not enough historical data points to generate candle structures.")
    else:
        current_pattern, probabilities, total_matches, htf_context = calculate_next_state_probabilities(
            state_history, 
            htf_state_history, 
            lookback_n, 
            use_htf_context=filter_by_htf
        )

        status_label = "Live Candle Included" if include_live_bar else "Closed Candles Only"
        st.markdown(f"Tracking Strat candle structure transitions for **{symbol}** on timeframe **{timeframe}** (`{status_label}`).")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Dataset Candles", len(state_history))
        with col2:
            st.metric("Current Pattern Sequence", " ➔ ".join(current_pattern))
        with col3:
            st.metric(f"Active HTF ({htf_timeframe}) Context", htf_context)
        with col4:
            st.metric("Sample Size (Matching Occurrences)", total_matches)

        st.markdown("---")
        
        context_msg = f" (Filtered by Active HTF [{htf_timeframe}]: **{htf_context}**)" if (filter_by_htf and htf_context != "N/A") else ""
        st.subheader(f"📊 Next Candle Structure Probability Breakdown{context_msg}")

        if total_matches > 0:
            prob_df = pd.DataFrame(list(probabilities.items()), columns=["Next Candle Structure", "Probability (%)"])
            prob_df["Probability (%)"] = prob_df["Probability (%)"].round(2)
            prob_df["Probability (Fraction)"] = prob_df["Probability (%)"].apply(lambda x: f"{x}%")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("##### Probability Table")
                st.dataframe(prob_df[["Next Candle Structure", "Probability (Fraction)"]], use_container_width=True, hide_index=True)
            with c2:
                st.markdown("##### Visual Distribution")
                st.bar_chart(prob_df.set_index("Next Candle Structure")["Probability (%)"])
        else:
            st.warning(f"⚠️ No historical matches found for this exact structural sequence under the active {htf_timeframe} context.")

        if show_all_patterns:
            st.markdown("---")
            st.subheader("📋 All Historical Structure Patterns Summary")
            all_summary_df = get_all_patterns_summary(state_history, lookback_n)
            if not all_summary_df.empty:
                st.dataframe(all_summary_df, use_container_width=True, hide_index=True)

        with st.expander("🔍 View Full Historical Candle Structure Series"):
            st.dataframe(state_history.tail(100).sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)

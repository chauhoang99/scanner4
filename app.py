from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="", layout="wide")

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
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.header("Structure Tracker Settings")

ticker_options = [
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X",
    "USDCHF=X", "USDJPY=X", "USDSGD=X", "GC=F", "BZ=F", "ZB=F",
    "BTC-USD", "EURGBP=X", "EURAUD=X", "EURNZD=X", "EURCAD=X",
    "EURCHF=X", "EURJPY=X", "EURSGD=X", "XAUEUR=X", "GBPAUD=X",
    "GBPNZD=X", "GBPCAD=X", "GBPCHF=X", "GBPJPY=X", "GBPSGD=X",
    "AUDNZD=X", "AUDCAD=X", "AUDCHF=X", "AUDJPY=X", "AUDSGD=X",
    "AAPL", "MSFT", "SPY", "QQQ"
]
symbol = st.sidebar.selectbox("Ticker Symbol", options=ticker_options, index=0, help="Select a ticker symbol from the list.")

st.sidebar.subheader("Timeframe & History")
timeframe = st.sidebar.selectbox("Timeframe", ["60m", "1d", "1wk", "1mo", "3mo"], index=1)
history_period = st.sidebar.selectbox("History Range", ["1y", "2y", "5y", "10y", "max"], index=1)

lookback_n = st.sidebar.slider("Pattern Lookback Window (Candles)", min_value=1, max_value=5, value=3, help="Number of past consecutive candle structures to match historically.")

show_all_patterns = st.sidebar.checkbox("Show All Historical Patterns Summary", value=False)

if st.sidebar.button("🔄 Run Analysis"):
    st.rerun()

# ---------------------------------------------------------
# CORE LOGIC: STRAT CANDLE STRUCTURE STATES
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        return None


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
        
        higher_high = h_curr > h_prev
        higher_low = l_curr > l_prev
        
        # The Strat Candle Classification:
        # 1  = Inside Bar
        # 2U = Directional Up Bar
        # 2D = Directional Down Bar
        # 3  = Outside Bar
        if higher_high and higher_low:
            state = "2U"
        elif not higher_high and not higher_low:
            state = "2D"
        elif not higher_high and higher_low:
            state = "1"
        else:  # higher_high and not higher_low
            state = "3"
            
        states.append(state)
        dates.append(df.index[i])
        
    return pd.DataFrame({'Date': dates, 'State': states})


def calculate_next_state_probabilities(state_df, n_back=3):
    if state_df.empty or len(state_df) <= n_back:
        return [], {}, 0
    
    states = state_df['State'].tolist()
    current_pattern = states[-n_back:]
    
    next_states = []
    for i in range(len(states) - n_back):
        window = states[i:i+n_back]
        if window == current_pattern:
            if i + n_back < len(states):
                next_states.append(states[i+n_back])
                
    if not next_states:
        return current_pattern, {}, 0
        
    total_matches = len(next_states)
    counts = Counter(next_states)
    
    probabilities = {state: (count / total_matches) * 100 for state, count in counts.items()}
    probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    
    return current_pattern, probabilities, total_matches


def get_all_patterns_summary(state_df, n_back=3):
    if state_df.empty or len(state_df) <= n_back:
        return pd.DataFrame()
    
    states = state_df['State'].tolist()
    pattern_transitions = defaultdict(list)
    
    for i in range(len(states) - n_back):
        window = tuple(states[i:i+n_back])
        next_val = states[i+n_back]
        pattern_transitions[window].append(next_val)
        
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
        
    summary_df = pd.DataFrame(summary_data)
    return summary_df.sort_values(by="Total Sample Occurrences", ascending=False)


# ---------------------------------------------------------
# MAIN DASHBOARD UI
# ---------------------------------------------------------
st.title("📈 Candle Structure Probability Tracker")
st.markdown(f"Tracking Strat candle structure transitions (1, 2U, 2D, 3) for **{symbol}** on timeframe **{timeframe}**.")

# Fetch Data
df = fetch_data(symbol, history_period, timeframe)

if df is None or df.empty:
    st.error(f"Could not retrieve data for ticker '{symbol}'. Please check the symbol and try again.")
else:
    state_history = get_candle_structure_series(df)
    
    if state_history.empty:
        st.warning("Not enough historical data points to generate candle structures.")
    else:
        current_pattern, probabilities, total_matches = calculate_next_state_probabilities(state_history, lookback_n)
        
        # Display Overview Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Dataset Candles", len(state_history))
        with col2:
            st.metric("Current Pattern Sequence", " ➔ ".join(current_pattern))
        with col3:
            st.metric("Sample Size (Matching Occurrences)", total_matches)
            
        st.markdown("---")
        
        # Results Section for Current Pattern
        st.subheader("📊 Next Candle Structure Probability Breakdown")
        st.info(f"ℹ️ This statistic is based on **{total_matches}** historical sample occurrences where the structure sequence matching your recent candles appeared.")
        
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
                chart_data = prob_df.set_index("Next Candle Structure")["Probability (%)"]
                st.bar_chart(chart_data)
        else:
            st.warning("⚠️ No historical matches found for this exact structural pattern in the selected range. Try expanding the history range.")
            
        # Show all patterns summary option if toggled
        if show_all_patterns:
            st.markdown("---")
            st.subheader("📋 All Historical Structure Patterns Summary")
            st.markdown(f"Overview of all unique structural sequences of length **{lookback_n}** found in history and their sample sizes:")
            all_summary_df = get_all_patterns_summary(state_history, lookback_n)
            if not all_summary_df.empty:
                st.dataframe(all_summary_df, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough data to generate pattern summaries.")

        with st.expander("🔍 View Full Historical Candle Structure Series"):
            st.dataframe(state_history.tail(100).sort_values(by="Date", ascending=False), use_container_width=True, hide_index=True)

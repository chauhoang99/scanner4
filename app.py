from collections import Counter
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# STRAT CANDLE CLASSIFIER
# ---------------------------------------------------------
def get_candle_structure_series(df):
    if df is None or len(df) < 2:
        return pd.DataFrame()

    states, dates = [], []
    for i in range(1, len(df)):
        h_prev, l_prev = df["High"].iloc[i - 1], df["Low"].iloc[i - 1]
        h_curr, l_curr = df["High"].iloc[i], df["Low"].iloc[i]
        c_curr, o_curr = df["Close"].iloc[i], df["Open"].iloc[i]

        higher_high, lower_low = h_curr > h_prev, l_curr < l_prev

        if higher_high and lower_low:
            num = "3"
        elif higher_high:
            num = "2U"
        elif lower_low:
            num = "2D"
        else:
            num = "1"

        arrow = "↑" if c_curr >= o_curr else "↓"

        if c_curr > h_prev:
            triangle = "▲"
        elif c_curr < l_prev:
            triangle = "▼"
        else:
            triangle = ""

        state = f"{num} {arrow} {triangle}".strip()
        states.append(state)
        dates.append(df.index[i])

    return pd.DataFrame({"Date": dates, "State": states})


# ---------------------------------------------------------
# CONTEXT-AWARE PROBABILITY ENGINE
# ---------------------------------------------------------
def calculate_contextual_probabilities(ltf_df, mtf_df, n_back=3):
    """Aligns Lower Timeframe (LTF) candles with Higher Timeframe (MTF Monthly) context

    and calculates probabilities filtered by matching Monthly state.
    """
    ltf_states = get_candle_structure_series(ltf_df)
    mtf_states = get_candle_structure_series(mtf_df)

    if ltf_states.empty or mtf_states.empty:
        return None, {}, 0, ""

    # Ensure Datetime index alignment
    ltf_states["Date"] = pd.to_datetime(ltf_states["Date"])
    mtf_states["Date"] = pd.to_datetime(mtf_states["Date"])

    ltf_states = ltf_states.sort_values("Date")
    mtf_states = mtf_states.sort_values("Date")

    # Align each LTF candle with the active Monthly state at that point in time
    merged = pd.merge_asof(
        ltf_states,
        mtf_states,
        on="Date",
        direction="backward",
        suffixes=("_LTF", "_MTF"),
    )

    if len(merged) <= n_back:
        return None, {}, 0, ""

    # Current Market Context
    current_mtf_context = merged["State_MTF"].iloc[-1]
    current_ltf_pattern = merged["State_LTF"].iloc[-n_back:].tolist()

    next_states = []
    # Search history for identical LTF patterns occurring UNDER THE SAME MONTHLY CONTEXT
    for i in range(len(merged) - n_back):
        window_ltf = merged["State_LTF"].iloc[i : i + n_back].tolist()
        context_at_time = merged["State_MTF"].iloc[i + n_back - 1]

        if (
            window_ltf == current_ltf_pattern
            and context_at_time == current_mtf_context
        ):
            if i + n_back < len(merged):
                next_states.append(merged["State_LTF"].iloc[i + n_back])

    total_matches = len(next_states)
    if total_matches == 0:
        return current_ltf_pattern, {}, 0, current_mtf_context

    counts = Counter(next_states)
    probs = {
        s: round((c / total_matches) * 100, 2)
        for s, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)
    }

    return current_ltf_pattern, probs, total_matches, current_mtf_context


# ---------------------------------------------------------
# STREAMLIT IMPLEMENTATION EXAMPLE
# ---------------------------------------------------------
st.title("🎯 Strat Contextual Probability Engine")

symbol = st.text_input("Ticker Symbol", "EURUSD=X")
ltf_timeframe = st.selectbox(
    "Execution Timeframe", ["1d", "1wk", "60m"], index=0
)
lookback_n = st.slider("LTF Pattern Lookback (Bars)", 1, 5, 3)

if st.button("Run Contextual Analysis"):
    # Fetch Lower Timeframe and Monthly Higher Timeframe Data
    ltf_data = yf.download(
        symbol, period="5y", interval=ltf_timeframe, progress=False
    )
    mtf_data = yf.download(
        symbol, period="5y", interval="1mo", progress=False
    )

    if isinstance(ltf_data.columns, pd.MultiIndex):
        ltf_data.columns = ltf_data.columns.get_level_values(0)
    if isinstance(mtf_data.columns, pd.MultiIndex):
        mtf_data.columns = mtf_data.columns.get_level_values(0)

    pattern, probabilities, matches, monthly_context = (
        calculate_contextual_probabilities(ltf_data, mtf_data, lookback_n)
    )

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Monthly Context", monthly_context)
    c2.metric("LTF Sequence", " ➔ ".join(pattern) if pattern else "N/A")
    c3.metric("Matching Historical Scenarios", matches)

    if matches > 0:
        st.subheader(
            f"Probabilities for Next Bar (Filtered by Monthly '{monthly_context}')"
        )
        prob_df = pd.DataFrame(
            list(probabilities.items()),
            columns=["Next Structure", "Probability (%)"],
        )
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        st.bar_chart(prob_df.set_index("Next Structure"))
    else:
        st.warning(
            f"No historical matches found for sequence {' ➔ '.join(pattern)} while Monthly was in state '{monthly_context}'."
        )

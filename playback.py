import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="OANDA Candle Playback", layout="wide")
st.title("OANDA Candle Playback")


def api_base(environment: str) -> str:
    return (
        "https://api-fxpractice.oanda.com"
        if environment == "Practice"
        else "https://api-fxtrade.oanda.com"
    )


def oanda_get(path: str, token: str, environment: str, params=None) -> dict:
    response = requests.get(
        api_base(environment) + path,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=params,
        timeout=30,
    )
    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(f"OANDA API {response.status_code}: {detail}")
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def discover_accounts(token: str, environment: str):
    data = oanda_get("/v3/accounts", token, environment)
    return [x["id"] for x in data.get("accounts", []) if x.get("id")]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_instruments(token: str, account_id: str, environment: str):
    data = oanda_get(
        f"/v3/accounts/{account_id}/instruments",
        token,
        environment,
    )
    return sorted(
        x["name"] for x in data.get("instruments", []) if x.get("name")
    )


@st.cache_data(ttl=5, show_spinner=False)
def fetch_candles(
    token: str,
    environment: str,
    instrument: str,
    granularity: str,
    count: int,
):
    # Candle data itself does not require the account id.
    params = {
        "price": "M",
        "granularity": granularity,
        "count": min(int(count), 5000),
        "smooth": "false",
    }

    data = oanda_get(
        f"/v3/instruments/{instrument}/candles",
        token,
        environment,
        params,
    )

    rows = []
    for candle in data.get("candles", []):
        mid = candle.get("mid")
        if not mid:
            continue

        rows.append(
            {
                "time": pd.to_datetime(candle["time"], utc=True),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "complete": bool(candle.get("complete", False)),
            }
        )

    if not rows:
        raise RuntimeError("OANDA returned no candle data.")

    return (
        pd.DataFrame(rows)
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


def reset_playback():
    st.session_state.playback_index = None
    st.session_state.playback_key = None


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("OANDA")

    environment = st.selectbox("Environment", ["Practice", "Live"])

    default_token = ""
    try:
        default_token = st.secrets.get("OANDA_API_KEY", "")
    except Exception:
        pass

    token = st.text_input(
        "API token",
        value=default_token,
        type="password",
    )

    if not token:
        st.info("Enter your OANDA API token.")
        st.stop()

    try:
        accounts = discover_accounts(token, environment)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if not accounts:
        st.error("No OANDA accounts are authorized for this API token.")
        st.stop()

    account_id = (
        accounts[0]
        if len(accounts) == 1
        else st.selectbox("Account", accounts)
    )

    try:
        instruments = fetch_instruments(token, account_id, environment)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    instrument = st.selectbox(
        "Instrument",
        instruments,
        index=instruments.index("EUR_USD") if "EUR_USD" in instruments else 0,
        on_change=reset_playback,
    )

    granularities = [
        "S5", "S10", "S15", "S30",
        "M1", "M2", "M4", "M5", "M10", "M15", "M30",
        "H1", "H2", "H3", "H4", "H6", "H8", "H12",
        "D", "W", "M",
    ]

    granularity = st.selectbox(
        "Timeframe",
        granularities,
        index=granularities.index("H1"),
        on_change=reset_playback,
    )

    history_bars = st.slider(
        "Candles to load",
        100,
        5000,
        1000,
        step=100,
        on_change=reset_playback,
    )

    chart_bars = st.slider(
        "Visible candles",
        25,
        500,
        150,
        step=25,
    )

    st.divider()
    st.header("Playback")

    bars_back = st.number_input(
        "Start bars back",
        min_value=1,
        max_value=max(1, int(history_bars) - 1),
        value=min(100, max(1, int(history_bars) - 1)),
        step=1,
    )

    go_to_bar = st.button(
        "⏮ Go to bar",
        use_container_width=True,
    )

    previous_bar = st.button(
        "◀ Previous bar",
        use_container_width=True,
    )

    next_bar = st.button(
        "Next bar ▶",
        type="primary",
        use_container_width=True,
    )

    latest_bar = st.button(
        "⏭ Latest",
        use_container_width=True,
    )

    refresh = st.button(
        "Refresh OANDA data",
        use_container_width=True,
    )


if refresh:
    fetch_candles.clear()


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

try:
    df = fetch_candles(
        token,
        environment,
        instrument,
        granularity,
        history_bars,
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

latest_index = len(df) - 1

playback_key = (
    environment,
    account_id,
    instrument,
    granularity,
    history_bars,
)

if (
    st.session_state.get("playback_key") != playback_key
    or st.session_state.get("playback_index") is None
):
    st.session_state.playback_key = playback_key
    st.session_state.playback_index = max(
        0,
        latest_index - int(bars_back),
    )

if go_to_bar:
    st.session_state.playback_index = max(
        0,
        latest_index - int(bars_back),
    )

if previous_bar:
    st.session_state.playback_index = max(
        0,
        int(st.session_state.playback_index) - 1,
    )

if next_bar:
    st.session_state.playback_index = min(
        latest_index,
        int(st.session_state.playback_index) + 1,
    )

if latest_bar:
    st.session_state.playback_index = latest_index

cursor = max(
    0,
    min(int(st.session_state.playback_index), latest_index),
)

# Future candles are completely hidden.
visible_history = df.iloc[: cursor + 1].copy()

# Keep only the requested number of candles on screen.
chart_start = max(0, len(visible_history) - int(chart_bars))
chart_df = visible_history.iloc[chart_start:].copy()

current = df.iloc[cursor]
remaining = latest_index - cursor


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------

c1, c2, c3 = st.columns(3)

c1.metric(
    "Playback candle",
    current["time"].strftime("%Y-%m-%d %H:%M UTC"),
)

c2.metric(
    "Bars from latest",
    remaining,
)

c3.metric(
    "Position",
    f"{cursor + 1} / {len(df)}",
)


# ----------------------------------------------------------------------
# Candlestick chart only
# ----------------------------------------------------------------------

fig = go.Figure(
    data=[
        go.Candlestick(
            x=chart_df["time"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name=instrument,
        )
    ]
)

fig.update_layout(
    title=f"{instrument} · {granularity}",
    xaxis_title=None,
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    height=720,
    margin=dict(l=20, r=20, t=55, b=20),
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
    },
)

if remaining == 0:
    st.caption("Playback is at the latest available OANDA candle.")
else:
    st.caption(
        f"{remaining} candle{'s' if remaining != 1 else ''} remain hidden. "
        "Press Next bar to reveal one candle."
    )

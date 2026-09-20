from datetime import datetime, timezone
import pandas as pd
import requests
import streamlit as st


@st.cache_data(ttl=3600)
def fetch_oanda_data_paginated(symbol_name, gran, start_yr, token, env):
    """Fetches complete historical candles starting from start_yr up to the present day

    by forward-chunking in 5,000 bar batches with clean RFC3339 timestamps.
    """
    inst = ticker_mapping.get(
        symbol_name, symbol_name.replace("=X", "").replace("=", "_")
    )
    domain = (
        "api-fxtrade.oanda.com" if env == "Live" else "api-fxpractice.oanda.com"
    )
    url = f"https://{domain}/v3/instruments/{inst}/candles"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Start timestamp in clean RFC3339 format
    from_time = f"{start_yr}-01-01T00:00:00Z"
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_candles = []
    max_requests = (
        100  # Allows up to 500,000 bars (covers 15m data back to 2020)
    )
    req_count = 0

    while req_count < max_requests:
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
                st.error(
                    f"⚠️ OANDA API Error on batch #{req_count + 1} (Status {res.status_code}): {res.text}"
                )
                break

            candles = res.json().get("candles", [])
            if not candles:
                break

            all_candles.extend(candles)

            # Reformat the timestamp to RFC3339 without nanoseconds
            last_time_str = candles[-1]["time"]
            last_dt = pd.to_datetime(last_time_str)
            next_from_time = last_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Break loop if end of stream reached, current time met, or timestamp didn't advance
            if (
                len(candles) < 5000
                or next_from_time >= now_iso
                or next_from_time == from_time
            ):
                break

            from_time = next_from_time
            req_count += 1

        except Exception as e:
            st.error(f"⚠️ Network error while fetching OANDA data: {e}")
            break

    if not all_candles:
        return None

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
    if not df.empty:
        df.drop_duplicates(subset=["Date"], inplace=True)
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)

    return df

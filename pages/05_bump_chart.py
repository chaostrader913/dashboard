import streamlit as st
import yfinance as yf
from lightweight_charts_v5 import lightweight_charts_v5_component

st.set_page_config(layout="wide")
st.title("📈 TradingView Dashboard")

# 1. Sidebar for User Input
ticker = st.sidebar.text_input("Enter Ticker", value="AAPL").upper()
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "15m", "5m"])

# 2. Fetch Data
data = yf.download(ticker, period="1y", interval=timeframe)

if not data.empty:
    # 3. Format data for the chart component
    # The component expects a list of dicts with 'time' and 'value' (or OHLC keys)
    chart_data = [
        {
            "time": str(date.date()) if timeframe == "1d" else int(date.timestamp()),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        }
        for date, row in data.iterrows()
    ]

    # 4. Render the Chart
    chart_options = {
        "layout": {"background": {"color": "#131722"}, "textColor": "#d1d4dc"},
        "grid": {"vertLines": {"color": "#242733"}, "horzLines": {"color": "#242733"}},
        "crosshair": {"mode": 0},
        "priceScale": {"borderColor": "#485c7b"},
        "timeScale": {"borderColor": "#485c7b", "timeVisible": True},
    }

    # Display the component
    lightweight_charts_v5_component(
        charts=[{
            "chart": chart_options,
            "series": [{
                "type": "Candlestick",
                "data": chart_data,
                "options": {
                    "upColor": "#26a69a",
                    "downColor": "#ef5350",
                    "borderVisible": False,
                    "wickUpColor": "#26a69a",
                    "wickDownColor": "#ef5350",
                }
            }]
        }],
        key="tradingview_chart"
    )
else:
    st.error("No data found for this ticker.")

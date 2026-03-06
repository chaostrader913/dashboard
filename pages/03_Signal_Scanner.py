import streamlit as st
import pandas as pd
import numpy as np
from streamlit_lightweight_charts import renderLightweightCharts

# Import from your utils folder
from utils.data_loader import fetch_data
from utils.indicators import (
    apply_td_sequential, 
    apply_rsi_divergence, 
    apply_macd, 
    apply_bollinger_bands,
    apply_advanced_trendlines
)

# --- 1. Terminal UI Styling ---
st.markdown("### 📡 MODULE: ADVANCED SIGNAL SCANNER (LW-CHARTS)")
st.divider()

# --- 2. Command Row (User Inputs) ---
col1, col2, col3, col4, col5 = st.columns([1.5, 1, 2, 2, 1])
with col1:
    ticker = st.text_input("TARGET ASSET", value="BTC-USD").upper()
with col2:
    timeframe = st.selectbox("TIMEFRAME", options=["1d", "1h", "15m"], index=0)
with col3:
    overlays = st.multiselect(
        "PRICE OVERLAYS", 
        options=["TD Sequential", "Auto Trendlines", "Bollinger Bands"],
        default=["TD Sequential", "Auto Trendlines"]
    )
with col4:
    oscillators = st.multiselect(
        "OSCILLATORS", 
        options=["RSI Divergence", "MACD"],
        default=["RSI Divergence"]
    )
with col5:
    max_tl = st.slider("MAX TRENDLINES", min_value=1, max_value=10, value=3) if "Auto Trendlines" in overlays else 3
        
# --- 3. Data Processing & Indicator Application ---
with st.spinner(f"EXECUTING ALGORITHMS FOR {ticker}..."):
    data = fetch_data(ticker, interval=timeframe, period="1y" if timeframe == "1d" else "60d")

if data is None or data.empty:
    st.error(f"ERR: NO DATA FOUND FOR {ticker}.")
    st.stop()

# Apply selected mathematical models
if "TD Sequential" in overlays: data = apply_td_sequential(data)
if "Bollinger Bands" in overlays: data = apply_bollinger_bands(data)
if "RSI Divergence" in oscillators: data = apply_rsi_divergence(data)
if "MACD" in oscillators: data = apply_macd(data)

# --- HELPER: Time Conversion ---
def get_time(idx):
    return int(pd.Timestamp(idx).timestamp())

# --- 4. Dynamic Charting Engine (Lightweight Charts) ---
charts = []

# Global Chart Theme
chart_layout = {
    "layout": { "textColor": '#E0E6ED', "background": { "type": 'solid', "color": '#0E1117' } },
    "grid": { "vertLines": { "color": '#2B3040' }, "horzLines": { "color": '#2B3040' } },
    "crosshair": { "mode": 0 }, 
    "timeScale": { "timeVisible": True, "secondsVisible": False }
}

# --- PANE 1: PRICE & OVERLAYS ---
main_series = []
candles = [{"time": get_time(idx), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for idx, r in data.iterrows()]

# Generate Markers for TD Sequential
markers = []
if "TD Sequential" in overlays:
    if 'Setup_Signal' in data.columns:
        for idx, r in data[data['Setup_Signal'] == 1].iterrows():
            markers.append({"time": get_time(idx), "position": "belowBar", "color": "#00FFAA", "shape": "arrowUp", "text": "9"})
        for idx, r in data[data['Setup_Signal'] == -1].iterrows():
            markers.append({"time": get_time(idx), "position": "aboveBar", "color": "#FF4B4B", "shape": "arrowDown", "text": "9"})
            
    if 'Countdown_Signal' in data.columns:
        for idx, r in data[data['Countdown_Signal'] == 1].iterrows():
            markers.append({"time": get_time(idx), "position": "belowBar", "color": "#00AAFF", "shape": "arrowUp", "text": "13", "size": 2})
        for idx, r in data[data['Countdown_Signal'] == -1].iterrows():
            markers.append({"time": get_time(idx), "position": "aboveBar", "color": "#FFAA00", "shape": "arrowDown", "text": "13", "size": 2})

# Append Candlesticks
main_series.append({
    "type": "Candlestick",
    "data": candles,
    "options": {
        "upColor": '#00FFAA', "downColor": '#FF4B4B',
        "borderVisible": False, "wickUpColor": '#00FFAA', "wickDownColor": '#FF4B4B'
    },
    "markers": markers if markers else []
})

# Overlay: Volume Profile (Ghosted at the bottom)
if "Volume" in data.columns:
    vol_data = [{"time": get_time(idx), "value": r['Volume'], "color": "rgba(0, 255, 170, 0.2)" if r['Close'] >= r['Open'] else "rgba(255, 75, 75, 0.2)"} for idx, r in data.iterrows()]
    main_series.append({
        "type": "Histogram",
        "data": vol_data,
        "options": {
            "priceFormat": {"type": 'volume'},
            "priceScaleId": "", # Keeps volume off the main price axis
            "scaleMargins": {"top": 0.8, "bottom": 0} # Pins to the bottom 20%
        }
    })

# Overlay: Bollinger Bands
if "Bollinger Bands" in overlays and 'BB_Upper' in data.columns:
    bb_up = [{"time": get_time(idx), "value": r['BB_Upper']} for idx, r in data.iterrows() if pd.notna(r['BB_Upper'])]
    bb_low = [{"time": get_time(idx), "value": r['BB_Lower']} for idx, r in data.iterrows() if pd.notna(r['BB_Lower'])]
    main_series.append({"type": "Line", "data": bb_up, "options": {"color": "#4B4BFF", "lineWidth": 1}})
    main_series.append({"type": "Line", "data": bb_low, "options": {"color": "#4B4BFF", "lineWidth": 1}})

# Overlay: Trendlines
if "Auto Trendlines" in overlays:
    upper_lines, lower_lines = apply_advanced_trendlines(data, window=5, pct_limit=10.0, breaks_limit=2)
    
    # Lightweight charts treats each trendline as an independent LineSeries with 2 data points
    for line in upper_lines:
        t1, t2 = get_time(line[0][0]), get_time(line[1][0])
        main_series.append({
            "type": "Line",
            "data": [{"time": t1, "value": line[0][1]}, {"time": t2, "value": line[1][1]}],
            "options": {"color": "#FF4B4B", "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}
        })
    for line in lower_lines:
        t1, t2 = get_time(line[0][0]), get_time(line[1][0])
        main_series.append({
            "type": "Line",
            "data": [{"time": t1, "value": line[0][1]}, {"time": t2, "value": line[1][1]}],
            "options": {"color": "#00FFAA", "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}
        })

# Register Main Chart
charts.append({
    "chartOptions": {**chart_layout, "height": 500},
    "series": main_series
})

# --- PANE 2/3: OSCILLATORS ---
if "RSI Divergence" in oscillators and 'RSI' in data.columns:
    rsi_data = [{"time": get_time(idx), "value": r['RSI']} for idx, r in data.iterrows() if pd.notna(r['RSI'])]
    rsi_markers = []
    
    if 'Signal' in data.columns:
        for idx, r in data[data['Signal'] == 1].iterrows():
            rsi_markers.append({"time": get_time(idx), "position": "belowBar", "color": "#00FFAA", "shape": "arrowUp", "text": "DIV"})

    charts.append({
        "chartOptions": {**chart_layout, "height": 200},
        "series": [{
            "type": "Line", 
            "data": rsi_data, 
            "options": {
                "color": "#00AAFF", "lineWidth": 1.5,
                "priceLines": [
                    {"price": 70, "color": "#FF4B4B", "lineStyle": 2, "lineWidth": 1},
                    {"price": 30, "color": "#00FFAA", "lineStyle": 2, "lineWidth": 1}
                ]
            },
            "markers": rsi_markers
        }]
    })

if "MACD" in oscillators and 'MACD' in data.columns:
    macd_line = [{"time": get_time(idx), "value": r['MACD']} for idx, r in data.iterrows() if pd.notna(r['MACD'])]
    macd_signal = [{"time": get_time(idx), "value": r['MACD_Signal']} for idx, r in data.iterrows() if pd.notna(r['MACD_Signal'])]
    macd_hist = [{"time": get_time(idx), "value": r['MACD_Hist'], "color": "#00FFAA" if r['MACD_Hist'] >= 0 else "#FF4B4B"} for idx, r in data.iterrows() if pd.notna(r['MACD_Hist'])]

    charts.append({
        "chartOptions": {**chart_layout, "height": 200},
        "series": [
            {"type": "Histogram", "data": macd_hist, "options": {"priceScaleId": ""}},
            {"type": "Line", "data": macd_line, "options": {"color": "#00AAFF", "lineWidth": 1.5}},
            {"type": "Line", "data": macd_signal, "options": {"color": "#FFBB00", "lineWidth": 1.5}}
        ]
    })

# --- 5. Render to Streamlit ---
with st.container(border=True):
    # This single call natively stacks and syncs all the charts in the array!
    renderLightweightCharts(charts)

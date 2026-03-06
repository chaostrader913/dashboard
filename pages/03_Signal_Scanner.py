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
col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1, 1.2, 2, 2, 1])
with col1:
    ticker = st.text_input("TARGET ASSET", value="BTC-USD").upper()
with col2:
    timeframe = st.selectbox("TIMEFRAME", options=["1d", "1h", "15m"], index=0)
with col3:
    # 🔥 NEW: Theme Selector
    theme_sel = st.selectbox("THEME", ["mike", "nightclouds", "yahoo", "blueskies"], index=0)
with col4:
    overlays = st.multiselect(
        "PRICE OVERLAYS", 
        options=["TD Sequential", "Auto Trendlines", "Bollinger Bands"],
        default=["TD Sequential", "Auto Trendlines"]
    )
with col5:
    oscillators = st.multiselect(
        "OSCILLATORS", 
        options=["RSI Divergence", "MACD"],
        default=["RSI Divergence"]
    )
with col6:
    max_tl = st.slider("MAX TRENDLINES", min_value=1, max_value=10, value=3) if "Auto Trendlines" in overlays else 3
        
# --- 3. Data Processing & Indicator Application ---
with st.spinner(f"EXECUTING ALGORITHMS FOR {ticker}..."):
    # 🔥 FIXED: Pulling 5 years of daily data to allow the MACD EMAs to properly "warm up"
    data = fetch_data(ticker, interval=timeframe, period="2y" if timeframe == "1d" else "60d")

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

# 🔥 NEW: Theme Dictionary mapped to Lightweight Charts API
themes = {
    "mike": {
        "bg": "#0E1117", "text": "#E0E6ED", "grid": "#2B3040", 
        "up": "#00FFAA", "down": "#FF4B4B", "vol_up": "rgba(0, 255, 170, 0.2)", "vol_down": "rgba(255, 75, 75, 0.2)"
    },
    "nightclouds": {
        "bg": "#131722", "text": "#D1D4DC", "grid": "#363C4E", 
        "up": "#26A69A", "down": "#EF5350", "vol_up": "rgba(38, 166, 154, 0.2)", "vol_down": "rgba(239, 83, 80, 0.2)"
    },
    "yahoo": {
        "bg": "#FFFFFF", "text": "#111111", "grid": "#E1E5EA", 
        "up": "#0081F2", "down": "#FF333A", "vol_up": "rgba(0, 129, 242, 0.2)", "vol_down": "rgba(255, 51, 58, 0.2)"
    },
    "blueskies": {
        "bg": "#F4F8FB", "text": "#000000", "grid": "#DDE6ED", 
        "up": "#089981", "down": "#F23645", "vol_up": "rgba(8, 153, 129, 0.2)", "vol_down": "rgba(242, 54, 69, 0.2)"
    }
}
c_theme = themes[theme_sel]

# Apply Selected Theme to Global Layout
chart_layout = {
    "layout": { "textColor": c_theme["text"], "background": { "type": 'solid', "color": c_theme["bg"] } },
    "grid": { "vertLines": { "color": c_theme["grid"] }, "horzLines": { "color": c_theme["grid"] } },
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
            markers.append({"time": get_time(idx), "position": "belowBar", "color": c_theme["up"], "shape": "arrowUp", "text": "9"})
        for idx, r in data[data['Setup_Signal'] == -1].iterrows():
            markers.append({"time": get_time(idx), "position": "aboveBar", "color": c_theme["down"], "shape": "arrowDown", "text": "9"})
            
    if 'Countdown_Signal' in data.columns:
        for idx, r in data[data['Countdown_Signal'] == 1].iterrows():
            markers.append({"time": get_time(idx), "position": "belowBar", "color": "#00AAFF", "shape": "arrowUp", "text": "13", "size": 2})
        for idx, r in data[data['Countdown_Signal'] == -1].iterrows():
            markers.append({"time": get_time(idx), "position": "aboveBar", "color": "#FFAA00", "shape": "arrowDown", "text": "13", "size": 2})

# Append Candlesticks (Themed)
main_series.append({
    "type": "Candlestick",
    "data": candles,
    "options": {
        "upColor": c_theme["up"], "downColor": c_theme["down"],
        "borderVisible": False, "wickUpColor": c_theme["up"], "wickDownColor": c_theme["down"]
    },
    "markers": markers if markers else []
})

# Overlay: Volume Profile (Themed)
if "Volume" in data.columns:
    vol_data = [{"time": get_time(idx), "value": r['Volume'], "color": c_theme["vol_up"] if r['Close'] >= r['Open'] else c_theme["vol_down"]} for idx, r in data.iterrows()]
    main_series.append({
        "type": "Histogram",
        "data": vol_data,
        "options": {
            "priceFormat": {"type": 'volume'},
            "priceScaleId": "", 
            "scaleMargins": {"top": 0.8, "bottom": 0} 
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
    
    for line in upper_lines:
        t1, t2 = get_time(line[0][0]), get_time(line[1][0])
        main_series.append({
            "type": "Line",
            "data": [{"time": t1, "value": line[0][1]}, {"time": t2, "value": line[1][1]}],
            "options": {"color": c_theme["down"], "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}
        })
    for line in lower_lines:
        t1, t2 = get_time(line[0][0]), get_time(line[1][0])
        main_series.append({
            "type": "Line",
            "data": [{"time": t1, "value": line[0][1]}, {"time": t2, "value": line[1][1]}],
            "options": {"color": c_theme["up"], "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}
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
            rsi_markers.append({"time": get_time(idx), "position": "belowBar", "color": c_theme["up"], "shape": "arrowUp", "text": "DIV"})

    charts.append({
        "chartOptions": {**chart_layout, "height": 200},
        "series": [{
            "type": "Line", 
            "data": rsi_data, 
            "options": {
                "color": "#00AAFF", "lineWidth": 1.5,
                "priceLines": [
                    {"price": 70, "color": c_theme["down"], "lineStyle": 2, "lineWidth": 1},
                    {"price": 30, "color": c_theme["up"], "lineStyle": 2, "lineWidth": 1}
                ]
            },
            "markers": rsi_markers
        }]
    })

if "MACD" in oscillators and 'MACD' in data.columns:
    macd_line = [{"time": get_time(idx), "value": r['MACD']} for idx, r in data.iterrows() if pd.notna(r['MACD'])]
    macd_signal = [{"time": get_time(idx), "value": r['MACD_Signal']} for idx, r in data.iterrows() if pd.notna(r['MACD_Signal'])]
    macd_hist = [{"time": get_time(idx), "value": r['MACD_Hist'], "color": c_theme["up"] if r['MACD_Hist'] >= 0 else c_theme["down"]} for idx, r in data.iterrows() if pd.notna(r['MACD_Hist'])]

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
    renderLightweightCharts(charts)

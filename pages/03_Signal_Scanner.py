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
    apply_corrected_qwma, # 🔥 SWAPPED: Bollinger Bands out, QWMA in
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
    theme_sel = st.selectbox("THEME", ["mike", "nightclouds", "yahoo", "blueskies"], index=0)
with col4:
    overlays = st.multiselect(
        "PRICE OVERLAYS", 
        options=["TD Sequential", "Auto Trendlines", "Corrected QWMA"], # 🔥 UI Updated
        default=["TD Sequential", "Corrected QWMA"]
    )
with col5:
    oscillators = st.multiselect(
        "OSCILLATORS", 
        options=["RSI Divergence", "MACD"],
        default=["RSI Divergence"]
    )
with col6:
    max_tl = st.slider("MAX TRENDLINES", min_value=1, max_value=10, value=3) if "Auto Trendlines" in overlays else 3
    use_log = st.checkbox("Logarithmic Scale", value=True) if timeframe == "1d" else False
        
# --- 3. Data Processing & Indicator Application ---
with st.spinner(f"EXECUTING ALGORITHMS FOR {ticker}..."):
    data = fetch_data(ticker, interval=timeframe, period="5y" if timeframe == "1d" else "60d")

if data is None or data.empty:
    st.error(f"ERR: NO DATA FOUND FOR {ticker}.")
    st.stop()

# Apply selected mathematical models
if "TD Sequential" in overlays: data = apply_td_sequential(data)
if "Corrected QWMA" in overlays: data = apply_corrected_qwma(data) # 🔥 Logic Updated
if "RSI Divergence" in oscillators: data = apply_rsi_divergence(data)
if "MACD" in oscillators: data = apply_macd(data)

def get_time(idx):
    return int(pd.Timestamp(idx).timestamp())

# --- 4. Dynamic Charting Engine (Lightweight Charts) ---
charts = []

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

# --- 4. Dynamic Charting Engine (Lightweight Charts) ---

# [Keep your themes and c_theme setup here...]

# Remove timeScale from the global layout so we can assign it individually
chart_layout = {
    "layout": { "textColor": c_theme["text"], "background": { "type": 'solid', "color": c_theme["bg"] } },
    "grid": { "vertLines": { "color": c_theme["grid"] }, "horzLines": { "color": c_theme["grid"] } },
    "crosshair": { "mode": 0 }, 
    "rightPriceScale": {
        "mode": 1 if use_log else 0,
        "autoScale": True,
        "alignLabels": True,
        "borderVisible": False,
    }
}

# --- DYNAMIC AXIS LOGIC ---
# Figure out which chart is at the very bottom so we only show the X-Axis once
has_rsi = "RSI Divergence" in oscillators
has_macd = "MACD" in oscillators

price_is_bottom = not (has_rsi or has_macd)
rsi_is_bottom = has_rsi and not has_macd

# --- PANE 1: PRICE & OVERLAYS ---
# [Keep your main_series, markers, volume, QWMA, and trendline logic here...]

# Register Main Chart
charts.append({
    "chartOptions": {
        **chart_layout, 
        "height": 600, # Enlarged Price Panel
        # 🔥 Hides the X-axis unless this is the only chart on the screen
        "timeScale": { "visible": price_is_bottom, "timeVisible": True, "secondsVisible": False } 
    },
    "series": main_series
})

# --- PANE 2: RSI ---
if has_rsi and 'RSI' in data.columns:
    rsi_data = [{"time": get_time(idx), "value": r['RSI']} for idx, r in data.iterrows() if pd.notna(r['RSI'])]
    rsi_markers = []
    
    if 'Signal' in data.columns:
        for idx, r in data[data['Signal'] == 1].iterrows():
            rsi_markers.append({"time": get_time(idx), "position": "belowBar", "color": c_theme["up"], "shape": "arrowUp", "text": "DIV"})

    charts.append({
        "chartOptions": {
            **chart_layout, 
            "height": 180,
            # 🔥 Hides the X-axis unless MACD is turned off
            "timeScale": { "visible": rsi_is_bottom, "timeVisible": True, "secondsVisible": False }
        },
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

# --- PANE 3: MACD ---
if has_macd and 'MACD' in data.columns:
    macd_line = [{"time": get_time(idx), "value": r['MACD']} for idx, r in data.iterrows() if pd.notna(r['MACD'])]
    macd_signal = [{"time": get_time(idx), "value": r['MACD_Signal']} for idx, r in data.iterrows() if pd.notna(r['MACD_Signal'])]
    macd_hist = [{"time": get_time(idx), "value": r['MACD_Hist'], "color": c_theme["up"] if r['MACD_Hist'] >= 0 else c_theme["down"]} for idx, r in data.iterrows() if pd.notna(r['MACD_Hist'])]
    
    # The Dummy Line Trick to fix MACD time-axis misalignment 
    dummy_timeline = [{"time": get_time(idx), "value": 0} for idx, r in data.iterrows()]

    charts.append({
        "chartOptions": {
            **chart_layout, 
            "height": 180,
            # 🔥 MACD is always at the bottom, so we force the X-Axis to be visible
            "timeScale": { "visible": True, "timeVisible": True, "secondsVisible": False }
        },
        "series": [
            {"type": "Line", "data": dummy_timeline, "options": {"color": "transparent", "priceLineVisible": False, "crosshairMarkerVisible": False, "lastValueVisible": False}},
            {"type": "Histogram", "data": macd_hist, "options": {"priceScaleId": ""}},
            {"type": "Line", "data": macd_line, "options": {"color": "#00AAFF", "lineWidth": 1.5}},
            {"type": "Line", "data": macd_signal, "options": {"color": "#FFBB00", "lineWidth": 1.5}}
        ]
    })

# --- 5. Render to Streamlit ---
with st.container(border=True):
    renderLightweightCharts(charts)

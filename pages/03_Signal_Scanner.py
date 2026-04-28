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
    apply_corrected_qwma, 
    apply_jma,
    apply_natural_moving_average,
    apply_natural_market_channel,
    apply_dma,
    apply_dma_bands
)

# --- 1. Terminal UI Styling ---
st.markdown("### 🎛️ MODULE: ADVANCED SIGNAL SCANNER (LW-CHARTS)")
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
        options=["TD Sequential", "Corrected QWMA", "Jurik MA", "Natural MA", "Natural Channel"], 
        default=["TD Sequential", "Corrected QWMA"]
    )
with col5:
    oscillators = st.multiselect(
        "OSCILLATORS", 
        options=["RSI Divergence", "MACD"],
        default=["RSI Divergence"]
    )
with col6:
    use_log = st.checkbox("Logarithmic", value=True) if timeframe == "1d" else False

# --- 3. Indicator Parameter Tuning ---
with st.expander("⚙️ INDICATOR PARAMETERS", expanded=False):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        jma_len = st.number_input("Jurik MA Length", 1, 100, 7)
        jma_phase = st.number_input("Jurik MA Phase", -100, 100, 0)
    with p2:
        nma_len = st.number_input("Natural MA Length", 1, 200, 40)
        nmc_mult = st.number_input("Natural Channel Mult", 0.1, 5.0, 1.5)
    with p3:
        qwma_len = st.number_input("QWMA Period", 1, 100, 25)
        macd_fast = st.number_input("MACD Fast", 1, 50, 12)
        macd_slow = st.number_input("MACD Slow", 1, 100, 26)
    with p4:
        rsi_len = st.number_input("RSI Period", 1, 50, 14)
        rsi_lookback = st.number_input("RSI Lookback", 5, 100, 20)

# --- 4. Backtesting Config Row ---
st.markdown("##### 🧪 BACKTEST CONFIGURATION")
bt_col1, bt_col2, bt_col3 = st.columns([1.5, 3, 6])
with bt_col1:
    run_backtest = st.checkbox("Enable Backtest Engine", value=False)
with bt_col2:
    available_strats = []
    if "Corrected QWMA" in overlays: available_strats.append("CQWMA Trend Color")
    if "Jurik MA" in overlays: available_strats.append("Price vs Jurik MA")
    if "Natural MA" in overlays or "Natural Channel" in overlays: available_strats.append("Price vs Natural MA")
    if "TD Sequential" in overlays: available_strats.append("TD Sequential (Buy 9/Sell 9)")
    if "MACD" in oscillators: available_strats.append("MACD Crossover")
    
    bt_source = None
    if run_backtest:
        if not available_strats:
            st.warning("⚠️ Select an indicator above to unlock backtesting.")
        else:
            bt_source = st.selectbox("Strategy Logic", available_strats, label_visibility="collapsed")

# --- 5. Data Processing & Indicator Application ---
with st.spinner(f"EXECUTING ALGORITHMS FOR {ticker}..."):
    data = fetch_data(ticker, interval=timeframe, period="5y" if timeframe == "1d" else "60d")

if data is None or data.empty:
    st.error(f"ERR: NO DATA FOUND FOR {ticker}.")
    st.stop()

# Apply selected mathematical models with user parameters
if "TD Sequential" in overlays: data = apply_td_sequential(data)
if "RSI Divergence" in oscillators: data = apply_rsi_divergence(data, rsi_period=rsi_len, lookback=rsi_lookback)
if "MACD" in oscillators: data = apply_macd(data, fast=macd_fast, slow=macd_slow)
if "Jurik MA" in overlays: data = apply_jma(data, length=jma_len, phase=jma_phase)
if "Natural MA" in overlays: data = apply_natural_moving_average(data, length=nma_len)
if "Natural Channel" in overlays: data = apply_natural_market_channel(data, nma_length=nma_len, multiplier=nmc_mult)
if "Corrected QWMA" in overlays: data = apply_corrected_qwma(data, ma_period=qwma_len)

# --- 6. Dynamic Backtest Engine Logic ---
if run_backtest and bt_source:
    pos = pd.Series(0, index=data.index)
    
    # Generate Position Vectors (1 = Long, -1 = Short)
    if bt_source == "CQWMA Trend Color" and 'CQWMA_Color' in data.columns:
        pos = data['CQWMA_Color'].replace({0: np.nan, 2: -1}).ffill().fillna(0)
        
    elif bt_source == "Price vs Jurik MA" and 'JMA' in data.columns:
        pos = pd.Series(np.where(data['Close'] > data['JMA'], 1, -1), index=data.index)
        
    elif bt_source == "Price vs Natural MA" and 'NMA' in data.columns:
        pos = pd.Series(np.where(data['Close'] > data['NMA'], 1, -1), index=data.index)
        
    elif bt_source == "MACD Crossover" and 'MACD' in data.columns:
        pos = pd.Series(np.where(data['MACD'] > data['MACD_Signal'], 1, -1), index=data.index)
        
    elif bt_source == "TD Sequential (Buy 9/Sell 9)" and 'Setup_Signal' in data.columns:
        pos = data['Setup_Signal'].replace(0, np.nan).ffill().fillna(0)

    # Shift by 1 to prevent lookahead bias (trade executes on the open of the next candle)
    data['Position'] = pos.shift(1)
    
    # Calculate daily strategy returns
    data['Strat_Return'] = data['Position'] * data['Close'].pct_change()
    
    # Calculate Equity Curve (Starting capital: $10,000)
    data['Equity'] = (1 + data['Strat_Return'].fillna(0)).cumprod() * 10000


def get_time(idx):
    return int(pd.Timestamp(idx).timestamp())

# --- 7. Dynamic Charting Engine (Lightweight Charts) ---
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
has_rsi = "RSI Divergence" in oscillators
has_macd = "MACD" in oscillators
has_bt = run_backtest and 'Equity' in data.columns

price_is_bottom = not (has_rsi or has_macd or has_bt)
rsi_is_bottom = has_rsi and not (has_macd or has_bt)
macd_is_bottom = has_macd and not has_bt
bt_is_bottom = has_bt

# --- PANE 1: PRICE & OVERLAYS ---
main_series = []
candles = [{"time": get_time(idx), "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for idx, r in data.iterrows()]

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

main_series.append({
    "type": "Candlestick",
    "data": candles,
    "options": {
        "upColor": c_theme["up"], "downColor": c_theme["down"],
        "borderVisible": False, "wickUpColor": c_theme["up"], "wickDownColor": c_theme["down"]
    },
    "markers": markers if markers else []
})

if "Volume" in data.columns:
    vol_data = [{"time": get_time(idx), "value": r['Volume'], "color": c_theme["vol_up"] if r['Close'] >= r['Open'] else c_theme["vol_down"]} for idx, r in data.iterrows()]
    main_series.append({
        "type": "Histogram",
        "data": vol_data,
        "options": {
            "priceFormat": {"type": 'volume'},
            "priceScaleId": "", 
            "scaleMargins": {"top": 0.85, "bottom": 0} 
        }
    })

# Custom Moving Averages
if "Jurik MA" in overlays and 'JMA' in data.columns:
    jma_data = [{"time": get_time(idx), "value": r['JMA']} for idx, r in data.iterrows() if pd.notna(r['JMA'])]
    main_series.append({"type": "Line", "data": jma_data, "options": {"color": "#E040FB", "lineWidth": 2, "crosshairMarkerVisible": False}})

if "Natural MA" in overlays and 'NMA' in data.columns:
    nma_data = [{"time": get_time(idx), "value": r['NMA']} for idx, r in data.iterrows() if pd.notna(r['NMA'])]
    main_series.append({"type": "Line", "data": nma_data, "options": {"color": "#00E676", "lineWidth": 2, "crosshairMarkerVisible": False}})

if "Natural Channel" in overlays and 'NMC_Upper' in data.columns:
    nmc_up = [{"time": get_time(idx), "value": r['NMC_Upper']} for idx, r in data.iterrows() if pd.notna(r['NMC_Upper'])]
    nmc_dn = [{"time": get_time(idx), "value": r['NMC_Lower']} for idx, r in data.iterrows() if pd.notna(r['NMC_Lower'])]
    main_series.append({"type": "Line", "data": nmc_up, "options": {"color": "#FF9100", "lineWidth": 1, "lineStyle": 2, "crosshairMarkerVisible": False}})
    main_series.append({"type": "Line", "data": nmc_dn, "options": {"color": "#FF9100", "lineWidth": 1, "lineStyle": 2, "crosshairMarkerVisible": False}})

if "Corrected QWMA" in overlays and 'CQWMA' in data.columns:
    cqwma_up = [{"time": get_time(idx), "value": r['CQWMA_Up']} for idx, r in data.iterrows() if pd.notna(r['CQWMA_Up'])]
    cqwma_dn = [{"time": get_time(idx), "value": r['CQWMA_Down']} for idx, r in data.iterrows() if pd.notna(r['CQWMA_Down'])]
    cqwma_mid = [{"time": get_time(idx), "value": r['CQWMA_Mid']} for idx, r in data.iterrows() if pd.notna(r['CQWMA_Mid'])]
    
    cqwma_data = []
    for idx, r in data.iterrows():
        val = r['CQWMA']
        if pd.notna(val):
            c_val = r['CQWMA_Color']
            color = c_theme["up"] if c_val == 1 else (c_theme["down"] if c_val == 2 else "#888888")
            cqwma_data.append({"time": get_time(idx), "value": val, "color": color})

    main_series.append({"type": "Line", "data": cqwma_up, "options": {"color": c_theme["up"], "lineWidth": 1, "lineStyle": 2, "crosshairMarkerVisible": False}})
    main_series.append({"type": "Line", "data": cqwma_mid, "options": {"color": "#888888", "lineWidth": 1, "lineStyle": 3, "crosshairMarkerVisible": False}})
    main_series.append({"type": "Line", "data": cqwma_dn, "options": {"color": c_theme["down"], "lineWidth": 1, "lineStyle": 2, "crosshairMarkerVisible": False}})
    main_series.append({"type": "Line", "data": cqwma_data, "options": {"lineWidth": 3}})

charts.append({
    "chart": {
        **chart_layout, 
        "height": 600, 
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

    rsi_dummy_timeline = [{"time": get_time(idx), "value": 50} for idx, r in data.iterrows()]

    charts.append({
        "chart": {
            **chart_layout, 
            "height": 180,
            "timeScale": { "visible": rsi_is_bottom, "timeVisible": True, "secondsVisible": False }
        },
        "series": [
            {"type": "Line", "data": rsi_dummy_timeline, "options": {"color": "transparent", "priceLineVisible": False, "crosshairMarkerVisible": False, "lastValueVisible": False}},
            {
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
            }
        ]
    })

# --- PANE 3: MACD ---
if has_macd and 'MACD' in data.columns:
    macd_line = [{"time": get_time(idx), "value": r['MACD']} for idx, r in data.iterrows() if pd.notna(r['MACD'])]
    macd_signal = [{"time": get_time(idx), "value": r['MACD_Signal']} for idx, r in data.iterrows() if pd.notna(r['MACD_Signal'])]
    macd_hist = [{"time": get_time(idx), "value": r['MACD_Hist'], "color": c_theme["up"] if r['MACD_Hist'] >= 0 else c_theme["down"]} for idx, r in data.iterrows() if pd.notna(r['MACD_Hist'])]
    
    dummy_timeline = [{"time": get_time(idx), "value": 0} for idx, r in data.iterrows()]

    charts.append({
        "chart": {
            **chart_layout, 
            "height": 180,
            "timeScale": { "visible": macd_is_bottom, "timeVisible": True, "secondsVisible": False }
        },
        "series": [
            {"type": "Line", "data": dummy_timeline, "options": {"color": "transparent", "priceLineVisible": False, "crosshairMarkerVisible": False, "lastValueVisible": False}},
            {"type": "Histogram", "data": macd_hist},
            {"type": "Line", "data": macd_line, "options": {"color": "#00AAFF", "lineWidth": 1.5}},
            {"type": "Line", "data": macd_signal, "options": {"color": "#FFBB00", "lineWidth": 1.5}}
        ]
    })

# --- PANE 4: BACKTEST EQUITY CURVE ---
if has_bt:
    baseline_data = [{"time": get_time(idx), "value": r['Equity']} for idx, r in data.iterrows() if pd.notna(r['Equity'])]
    bt_dummy = [{"time": get_time(idx), "value": 10000} for idx, r in data.iterrows()]

    charts.append({
        "chart": {
            **chart_layout, 
            "height": 200,
            "timeScale": { "visible": bt_is_bottom, "timeVisible": True, "secondsVisible": False }
        },
        "series": [
            {"type": "Line", "data": bt_dummy, "options": {"color": "transparent", "priceLineVisible": False, "crosshairMarkerVisible": False, "lastValueVisible": False}},
            {
                "type": "Baseline", 
                "data": baseline_data, 
                "options": {
                    "baseValue": {"type": "price", "price": 10000}, # Anchors chart at initial capital
                    "topLineColor": c_theme["up"],
                    "topFillColor1": c_theme["vol_up"],
                    "topFillColor2": "rgba(0, 0, 0, 0)",
                    "bottomLineColor": c_theme["down"],
                    "bottomFillColor1": "rgba(0, 0, 0, 0)",
                    "bottomFillColor2": c_theme["vol_down"],
                    "lineWidth": 2,
                }
            }
        ]
    })

# --- 8. Render to Streamlit ---
with st.container(border=True):
    renderLightweightCharts(charts)

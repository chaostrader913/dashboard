import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import io
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

try:
    from utils.data_loader import fetch_data
    from utils.indicators import apply_td_sequential, apply_rsi_divergence, apply_corrected_qwma
except ImportError:
    st.error("Missing utility files. Ensure 'utils/data_loader.py' and 'utils/indicators.py' exist.")
    st.stop()

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Multi-Timeframe Analyzer")

# --- 2. Logic Engines ---

def get_signal_status(df):
    """Determines the status including CQWMA trend states."""
    if df is None or df.empty: return "⚪ OFFLINE"
    recent = df.tail(3)
    
    # 1. High Priority: TD13
    if 'Countdown_Signal' in recent.columns:
        last_val = recent['Countdown_Signal'].replace(0, np.nan).ffill().iloc[-1]
        if last_val == 1: return "🔥 TD13 BUY"
        if last_val == -1: return "💀 TD13 SELL"

    # 2. Medium Priority: TD9
    if 'Setup_Signal' in recent.columns:
        last_val = recent['Setup_Signal'].replace(0, np.nan).ffill().iloc[-1]
        if last_val == 1: return "🟢 TD9 BUY"
        if last_val == -1: return "🔴 TD9 SELL"
        
    # 3. RSI Divergence
    if 'Signal' in recent.columns and recent['Signal'].any():
        return "🔵 RSI DIV"
    
    # 4. Trend Filter: CQWMA Color (1=Green, 2=Red, 0=Gray)
    if 'CQWMA_Color' in df.columns:
        trend = df['CQWMA_Color'].iloc[-1]
        if trend == 1: return "📈 TREND UP"
        if trend == 2: return "📉 TREND DOWN"
        
    return "⚪ NEUTRAL"

def calculate_confluence(sync_report):
    """Calculates score with weighted CQWMA trends."""
    weights = { "30M": 1, "1H": 2, "4H": 5, "D": 10, "W": 20, "M": 30 }
    total_score = 0
    max_active_weight = 0
    
    for label, (_, status) in sync_report.items():
        w = weights.get(label, 5)
        max_active_weight += w
        if "BUY" in status or "UP" in status: 
            total_score += w if "BUY" in status else (w * 0.5)
        elif "SELL" in status or "DOWN" in status: 
            total_score -= w if "SELL" in status else (w * 0.5)
            
    if max_active_weight == 0: return 0
    return np.clip((total_score / max_active_weight) * 100, -100, 100)

# --- 3. Configuration ---
TIMEFRAMES = [
    {"interval": "30m", "period": "5d",   "label": "30M"},
    {"interval": "1h",  "period": "1mo",  "label": "1H"},
    {"interval": "4h",  "period": "3mo",  "label": "4H"},
    {"interval": "1d",  "period": "1y",   "label": "D"},
    {"interval": "1wk", "period": "3y",   "label": "W"},
    {"interval": "1mo", "period": "5y",   "label": "M"},
]

# --- 4. Sidebar ---
with st.sidebar:
    st.header("🔎 ASSET SYNC")
    ticker = st.text_input("SYMBOL", value="BTC-USD").upper()
    chart_sel = st.selectbox("CHART TYPE", ['Candlestick', 'Point & Figure', 'Renko'], index=0)
    style_sel = st.selectbox("THEME", ['nightclouds', 'yahoo', 'mike', 'blueskies'], index=2)
    show_vol = st.checkbox("Show Volume", value=False)
    
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    if auto_refresh:
        st_autorefresh(interval=60000, limit=None, key="mtf_refresh")

chart_type_map = {'Candlestick': 'candle', 'Point & Figure': 'pnf', 'Renko': 'renko'}
selected_type = chart_type_map[chart_sel]

# --- 5. Main Content ---
if ticker:
    st.markdown(f"### 🧭 {ticker} Fractal Sync")
    
    sync_report = {}
    with st.spinner("Crunching data..."):
        for tf in TIMEFRAMES:
            data = fetch_data(ticker, tf['interval'], tf['period'])
            if data is not None and not data.empty:
                df = data.copy()
                df = apply_td_sequential(df)
                df = apply_rsi_divergence(df)
                df = apply_corrected_qwma(df)
                sync_report[tf['label']] = (df, get_signal_status(df))
            else:
                sync_report[tf['label']] = (None, "⚪ OFFLINE")

    st.divider()
    score = calculate_confluence(sync_report)
    st.metric("CONFLUENCE SCORE", f"{score:+.1f}%")

    # --- 6. The Grid ---
    rows = [TIMEFRAMES[i:i+3] for i in range(0, len(TIMEFRAMES), 3)]
    for row_tfs in rows:
        cols = st.columns(3)
        for i, tf_info in enumerate(row_tfs):
            label = tf_info['label']
            data, status = sync_report[label]
            
            with cols[i]:
                if data is not None:
                    apds = []                    
                    if selected_type == 'candle' and 'CQWMA' in data.columns:
                        # FIX: Use a single color for the base line to prevent ValueError
                        apds.append(mpf.make_addplot(data['CQWMA'], color='#888888', width=1.0, alpha=0.5))
                        
                        # Use Scatter plots to represent the Trend Color states on top of the line
                        bull_qwma = np.where(data['CQWMA_Color'] == 1, data['CQWMA'], np.nan)
                        bear_qwma = np.where(data['CQWMA_Color'] == 2, data['CQWMA'], np.nan)
                        
                        if not np.all(np.isnan(bull_qwma)):
                            apds.append(mpf.make_addplot(bull_qwma, type='scatter', marker='.', color='#00FFAA', markersize=10))
                        if not np.all(np.isnan(bear_qwma)):
                            apds.append(mpf.make_addplot(bear_qwma, type='scatter', marker='.', color='#FF4B4B', markersize=10))
                        
                        # Floating Levels
                        apds.append(mpf.make_addplot(data['CQWMA_Up'], color='#00FFAA', linestyle='--', width=0.5, alpha=0.3))
                        apds.append(mpf.make_addplot(data['CQWMA_Down'], color='#FF4B4B', linestyle='--', width=0.5, alpha=0.3))

                        # Indicators: TD9/TD13
                        if 'Setup_Signal' in data.columns:
                            b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.99, np.nan)
                            s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.01, np.nan)
                            if not np.all(np.isnan(b9)): apds.append(mpf.make_addplot(b9, type='scatter', marker='^', color='#00FFAA', markersize=20))
                            if not np.all(np.isnan(s9)): apds.append(mpf.make_addplot(s9, type='scatter', marker='v', color='#FF4B4B', markersize=20))

                    fig, axlist = mpf.plot(data, type=selected_type, style=style_sel, volume=show_vol, 
                                           addplot=apds, figsize=(5, 3.5), tight_layout=True, returnfig=True)
                    
                    # Watermark
                    axlist[0].text(0.5, 0.5, label, transform=axlist[0].transAxes, fontsize=50, 
                                 fontweight='bold', color='white', alpha=0.1, ha='center', va='center')
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', pad_inches=0)
                    st.image(buf, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning(f"{label} No Data")

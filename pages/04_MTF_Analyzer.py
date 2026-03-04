import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import io
import matplotlib.pyplot as plt

# --- IMPORT GLOBALS ---
from utils.data_loader import fetch_data
from utils.indicators import apply_td_sequential, apply_rsi_divergence

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Multi-Timeframe Analyzer")

# Custom CSS for high-density layout
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem; max-width: 100%; }
    div[data-testid="stMetric"] { background-color: #1a1c24; padding: 10px; border-radius: 5px; border: 1px solid #2b3040; }
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. Logic Engines ---

def get_signal_status(df):
    """Detects the most recent active signal in a dataframe."""
    if df is None or df.empty: return "⚪ OFFLINE"
    
    # Check the most recent 3 bars for activity
    recent = df.tail(3)
    
    if 'Countdown_Signal' in recent.columns and recent['Countdown_Signal'].any():
        val = recent['Countdown_Signal'].iloc[-1]
        return "🔥 TD13 BUY" if val == 1 else "💀 TD13 SELL"
    
    if 'Setup_Signal' in recent.columns and recent['Setup_Signal'].any():
        val = recent['Setup_Signal'].iloc[-1]
        return "🟢 TD9 BUY" if val == 1 else "🔴 TD9 SELL"
        
    if 'Signal' in recent.columns and recent['Signal'].any():
        return "🔵 RSI DIV"
        
    return "⚪ NEUTRAL"

def calculate_confluence(sync_report):
    """Calculates a weighted sentiment score across resolutions."""
    weights = {
        "1M": 1, "5M": 2, "15M": 4, "30M": 6, 
        "1H": 10, "4H": 15, "D": 25, 
        "W": 40, "M": 50
    }
    total_score = 0
    max_possible = sum(weights.values())
    
    for label, (_, status) in sync_report.items():
        w = weights.get(label, 5)
        if "BUY" in status: total_score += w
        elif "SELL" in status: total_score -= w
            
    return np.clip((total_score / max_possible) * 100, -100, 100)

def generate_trade_summary(score, sync_report):
    """Generates an institutional-grade summary based on fractal alignment."""
    macro_signals = [s for l, (_, s) in sync_report.items() if l in ["D", "W", "M"]]
    micro_signals = [s for l, (_, s) in sync_report.items() if l in ["1M", "5M", "15M"]]
    
    macro_bull = any("BUY" in s for s in macro_signals)
    macro_bear = any("SELL" in s for s in macro_signals)
    micro_bull = any("BUY" in s for s in micro_signals)
    micro_bear = any("SELL" in s for s in micro_signals)
    
    if score > 65: return "🚀 CONFLUENT UPTREND: Trend stacking detected. High probability of continuation."
    if score < -65: return "⚠️ SYSTEMIC WEAKNESS: Selling pressure across all resolutions. Avoid longs."
    if macro_bull and micro_bear: return "⚖️ MEAN REVERSION: Macro trend is Bullish, but Micro is overextended. Buy the dip."
    if macro_bear and micro_bull: return "🩸 DEAD CAT BOUNCE: Macro trend is Bearish. Short-term strength is likely a trap."
    if abs(score) < 15: return "🌀 COMPRESSION: Market is in a fractal squeeze. Wait for 4H/Daily direction."
    return "🔎 MONITORING: Mixed alignment. Look for the 1H 'Anchor' to flip direction."

# --- 3. Configuration ---
TIMEFRAMES = [
    {"interval": "1m",  "period": "1d",   "label": "1M"},
    {"interval": "5m",  "period": "1d",   "label": "5M"},
    {"interval": "15m", "period": "2d",   "label": "15M"},
    {"interval": "30m", "period": "5d",   "label": "30M"},
    {"interval": "1h",  "period": "1wk",  "label": "1H"},
    {"interval": "2h",  "period": "1mo",  "label": "2H"},
    {"interval": "4h",  "period": "3mo",  "label": "4H"},
    {"interval": "1d",  "period": "6mo",  "label": "D"},
    {"interval": "1d",  "period": "1y",   "label": "D-LONG"},
    {"interval": "1wk", "period": "1y",   "label": "W"},
    {"interval": "1wk", "period": "3y",   "label": "W-LONG"},
    {"interval": "1mo", "period": "5y",   "label": "M"},
]

# --- 4. Sidebar ---
with st.sidebar:
    st.header("🔎 ASSET SYNC")
    ticker = st.text_input("SYMBOL", value="NVDA").upper()
    style_sel = st.selectbox("THEME", ['nightclouds', 'yahoo', 'mike', 'blueskies'], index=0)
    st.divider()
    show_vol = st.checkbox("Show Volume", value=False)
    st.caption("Lower timeframes (1M-5M) may have limited history depending on ticker type.")

# --- 5. Main Content Execution ---
if ticker:
    st.markdown(f"### 🧭 MTF FRACTAL SYNC: {ticker}")
    
    sync_report = {}
    with st.spinner(f"Synchronizing 12 resolutions for {ticker}..."):
        for tf in TIMEFRAMES:
            data = fetch_data(ticker, tf['interval'], tf['period'])
            if data is not None and not data.empty:
                # We use .copy() to prevent SettingWithCopy warnings across resolutions
                df = data.copy()
                df = apply_td_sequential(df)
                df = apply_rsi_divergence(df)
                sync_report[tf['label']] = (df, get_signal_status(df))
            else:
                sync_report[tf['label']] = (None, "⚪ OFFLINE")

    # --- TOP SYNC STRIP (HEATMAP) ---
    h_cols = st.columns(len(TIMEFRAMES))
    for i, (label, (df, status)) in enumerate(sync_report.items()):
        color = "#888888" # Default Gray
        if "BUY" in status: color = "#00FFAA"
        elif "SELL" in status: color = "#FF4B4B"
        elif "DIV" in status: color = "#00AAFF"
        
        with h_cols[i]:
            st.markdown(f"<div style='text-align:center;'><small><b>{label}</b></small><br><span style='color:{color}; font-size:9px;'>{status}</span></div>", unsafe_allow_html=True)

    st.divider()

    # --- SENTIMENT & STRATEGY DASHBOARD ---
    score = calculate_confluence(sync_report)
    summary = generate_trade_summary(score, sync_report)
    
    # Gauge Color Logic
    gauge_color = "#00FFAA" if score > 20 else "#FF4B4B" if score < -20 else "#888888"
    
    m_left, m_right = st.columns([1, 2])
    with m_left:
        st.markdown(f"""
        <div style="background-color: #1a1c24; padding: 15px; border-radius: 8px; border-left: 5px solid {gauge_color};">
            <h4 style="margin:0; color:{gauge_color};">FRACTAL CONFLUENCE</h4>
            <p style="margin:0; font-family:monospace; font-size: 28px; font-weight:bold;">{score:+.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m_right:
        st.markdown(f"""
        <div style="background-color: #0e1117; padding: 15px; border: 1px solid #2b3040; border-radius: 8px; height: 75px; display: flex; align-items: center;">
            <p style="margin: 0; font-size: 15px;">{summary}</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # Spacing

    # --- 6. THE GRID (4x3 Layout) ---
    for row in range(3):
        grid_cols = st.columns(4)
        for col in range(4):
            idx = row * 4 + col
            if idx < len(TIMEFRAMES):
                tf_label = TIMEFRAMES[idx]['label']
                data, status = sync_report[tf_label]
                
                with grid_cols[col]:
                    if data is not None:
                        # Prepare Addplots
                        apds = []
                        if 'Setup_Signal' in data.columns:
                            b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.98, np.nan)
                            s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.02, np.nan)
                            if not np.all(np.isnan(b9)): apds.append(mpf.make_addplot(b9, type='scatter', marker=r'$9$', color='#00FFAA', markersize=25))
                            if not np.all(np.isnan(s9)): apds.append(mpf.make_addplot(s9, type='scatter', marker=r'$9$', color='#FF4B4B', markersize=25))
                        
                        # Plot Generation
                        fig, axlist = mpf.plot(
                            data, type='candle', style=style_sel, volume=show_vol,
                            returnfig=True, figsize=(4, 2.5), tight_layout=True,
                            addplot=apds if apds else None, xrotation=0,
                            axisoff=True, # Minimalist look for grid
                        )
                        
                        # Title inside the chart
                        axlist[0].set_title(tf_label, fontsize=12, color='white', loc='left', pad=-15)
                        
                        # Border highlight for active signals
                        if "BUY" in status or "SELL" in status:
                            border_color = "#00FFAA" if "BUY" in status else "#FF4B4B"
                            rect = plt.Rectangle((0,0), 1, 1, fill=False, color=border_color, lw=3, transform=fig.transFigure)
                            fig.patches.append(rect)

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=85, bbox_inches='tight', facecolor=fig.get_facecolor())
                        st.image(buf, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.error(f"ERR: {tf_label}")

else:
    st.info("👈 Enter a ticker symbol in the sidebar to synchronize resolutions.")

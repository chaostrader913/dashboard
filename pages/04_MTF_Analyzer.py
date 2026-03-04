import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Essential for web-server stability

# --- IMPORT GLOBALS (Ensure these files exist in your /utils folder) ---
try:
    from utils.data_loader import fetch_data
    from utils.indicators import apply_td_sequential, apply_rsi_divergence
except ImportError:
    st.error("Missing utility files. Ensure 'utils/data_loader.py' and 'utils/indicators.py' exist.")
    st.stop()

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Multi-Timeframe Analyzer")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem; max-width: 100%; }
    div[data-testid="stMetric"] { background-color: #1a1c24; padding: 10px; border-radius: 5px; border: 1px solid #2b3040; }
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. Logic Engines ---

def get_signal_status(df):
    if df is None or df.empty: return "⚪ OFFLINE"
    
    recent = df.tail(3)
    
    # Check TD Countdown (Prioritize the most powerful signal)
    if 'Countdown_Signal' in recent.columns:
        last_val = recent['Countdown_Signal'].replace(0, np.nan).ffill().iloc[-1]
        if last_val == 1: return "🔥 TD13 BUY"
        if last_val == -1: return "💀 TD13 SELL"

    # Check TD Setup
    if 'Setup_Signal' in recent.columns:
        last_val = recent['Setup_Signal'].replace(0, np.nan).ffill().iloc[-1]
        if last_val == 1: return "🟢 TD9 BUY"
        if last_val == -1: return "🔴 TD9 SELL"
        
    if 'Signal' in recent.columns and recent['Signal'].any():
        return "🔵 RSI DIV"
        
    return "⚪ NEUTRAL"

def calculate_confluence(sync_report):
    weights = {
        "1M": 1, "5M": 2, "15M": 4, "30M": 6, 
        "1H": 10, "4H": 15, "D": 25, "D-L": 30,
        "W": 40, "W-L": 45, "M": 50
    }
    total_score = 0
    max_active_weight = 0
    
    for label, (_, status) in sync_report.items():
        w = weights.get(label, 5)
        max_active_weight += w
        if "BUY" in status: total_score += w
        elif "SELL" in status: total_score -= w
            
    if max_active_weight == 0: return 0
    return np.clip((total_score / max_active_weight) * 100, -100, 100)

def generate_trade_summary(score, sync_report):
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

# --- 3. Configuration (Fixed duplicate Labels) ---
TIMEFRAMES = [
    {"interval": "1m",  "period": "1d",   "label": "1M"},
    {"interval": "5m",  "period": "1d",   "label": "5M"},
    {"interval": "15m", "period": "2d",   "label": "15M"},
    {"interval": "30m", "period": "5d",   "label": "30M"},
    {"interval": "1h",  "period": "1wk",  "label": "1H"},
    {"interval": "4h",  "period": "3mo",  "label": "4H"},
    {"interval": "1d",  "period": "6mo",  "label": "D"},
    {"interval": "1d",  "period": "2y",   "label": "D-L"},
    {"interval": "1wk", "period": "1y",   "label": "W"},
    {"interval": "1wk", "period": "3y",   "label": "W-L"},
    {"interval": "1mo", "period": "2y",   "label": "M"},
    {"interval": "1mo", "period": "max",  "label": "MAX"},
]

# --- 4. Sidebar ---
with st.sidebar:
    st.header("🔎 ASSET SYNC")
    ticker = st.text_input("SYMBOL", value="NVDA").upper()
    style_sel = st.selectbox("THEME", ['nightclouds', 'yahoo', 'mike', 'blueskies'], index=0)
    st.divider()
    show_vol = st.checkbox("Show Volume", value=False)

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
                sync_report[tf['label']] = (df, get_signal_status(df))
            else:
                sync_report[tf['label']] = (None, "⚪ OFFLINE")

    # Heatmap Strip
    h_cols = st.columns(len(TIMEFRAMES))
    for i, (label, (df, status)) in enumerate(sync_report.items()):
        color = "#888888"
        if "BUY" in status: color = "#00FFAA"
        elif "SELL" in status: color = "#FF4B4B"
        elif "DIV" in status: color = "#00AAFF"
        with h_cols[i]:
            st.markdown(f"<div style='text-align:center;'><small><b>{label}</b></small><br><span style='color:{color}; font-size:10px;'>{status.split()[-1]}</span></div>", unsafe_allow_html=True)

    st.divider()

    # Sentiment Score
    score = calculate_confluence(sync_report)
    summary = generate_trade_summary(score, sync_report)
    gauge_color = "#00FFAA" if score > 20 else "#FF4B4B" if score < -20 else "#888888"
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("CONFLUENCE", f"{score:+.1f}%", delta_color="normal")
    with c2:
        st.info(summary)

    # --- 6. The Grid ---
    rows = [TIMEFRAMES[i:i+4] for i in range(0, len(TIMEFRAMES), 4)]
    for row_tfs in rows:
        cols = st.columns(4)
        for i, tf_info in enumerate(row_tfs):
            label = tf_info['label']
            data, status = sync_report[label]
            with cols[i]:
                if data is not None:
                    apds = []
                    # Plot logic for TD 9s
                    if 'Setup_Signal' in data.columns:
                        b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.99, np.nan)
                        s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.01, np.nan)
                        if not np.all(np.isnan(b9)): apds.append(mpf.make_addplot(b9, type='scatter', marker='^', color='#00FFAA', markersize=20))
                        if not np.all(np.isnan(s9)): apds.append(mpf.make_addplot(s9, type='scatter', marker='v', color='#FF4B4B', markersize=20))

                    fig, axlist = mpf.plot(
                        data, type='candle', style=style_sel, volume=show_vol,
                        figsize=(5, 3.5), tight_layout=True, returnfig=True,
                        addplot=apds if apds else None, axisoff=True
                    )
                    axlist[0].set_title(label, fontsize=14, color='white', loc='left')
                    
                    # Highlight Active signals with a border
                    if "BUY" in status or "SELL" in status:
                        rect_color = "#00FFAA" if "BUY" in status else "#FF4B4B"
                        fig.patch.set_linewidth(4)
                        fig.patch.set_edgecolor(rect_color)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
                    st.image(buf)
                    plt.close(fig) # Prevent memory leaks
                else:
                    st.warning(f"{label} No Data")

else:
    st.info("👈 Enter a ticker symbol in the sidebar.")

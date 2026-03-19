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
    # Added apply_corrected_qwma to imports
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
    
    # 4. Trend Filter: CQWMA Color
    # 1 = Green (Bullish), 2 = Red (Bearish), 0 = Gray (Neutral)
    if 'CQWMA_Color' in recent.columns:
        trend = recent['CQWMA_Color'].iloc[-1]
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
            # Signals (BUY) get full weight, Trend (UP) gets 50% weight for stability
            total_score += w if "BUY" in status else (w * 0.5)
        elif "SELL" in status or "DOWN" in status: 
            total_score -= w if "SELL" in status else (w * 0.5)
            
    if max_active_weight == 0: return 0
    return np.clip((total_score / max_active_weight) * 100, -100, 100)

def generate_trade_summary(score, sync_report):
    macro_signals = [s for l, (_, s) in sync_report.items() if l in ["D", "W", "M"]]
    micro_signals = [s for l, (_, s) in sync_report.items() if l in ["30M", "1H", "4H"]]
    
    macro_bull = any(x in s for s in macro_signals for x in ["BUY", "UP"])
    macro_bear = any(x in s for s in macro_signals for x in ["SELL", "DOWN"])
    micro_bull = any(x in s for s in micro_signals for x in ["BUY", "UP"])
    micro_bear = any(x in s for s in micro_signals for x in ["SELL", "DOWN"])
    
    if score > 65: return "🚀 CONFLUENT UPTREND: Trend stacking detected. High probability of continuation."
    if score < -65: return "⚠️ SYSTEMIC WEAKNESS: Selling pressure across all resolutions. Avoid longs."
    if macro_bull and micro_bear: return "⚖️ MEAN REVERSION: Macro trend is Bullish, but Micro is overextended. Buy the dip."
    if macro_bear and micro_bull: return "🩸 DEAD CAT BOUNCE: Macro trend is Bearish. Short-term strength is likely a trap."
    if abs(score) < 15: return "🌀 COMPRESSION: Market is in a fractal squeeze. Wait for 4H/Daily direction."
    return "🔎 MONITORING: Mixed alignment. Look for the 1H/4H 'Anchor' to flip direction."

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
    st.divider()
    show_vol = st.checkbox("Show Volume", value=False)
    st.subheader("⏱️ Live Sync")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    refresh_rate = st.slider("Refresh Interval (Seconds)", min_value=30, max_value=300, value=60, step=30)
    
    if auto_refresh:
        st_autorefresh(interval=refresh_rate * 1000, limit=None, key="mtf_refresh")
        st.caption(f"🟢 Polling API every {refresh_rate}s")
    else:
        st.caption("🔴 Live Sync Paused")

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
                df = apply_corrected_qwma(df) # NEW: Integrated CQWMA
                sync_report[tf['label']] = (df, get_signal_status(df))
            else:
                sync_report[tf['label']] = (None, "⚪ OFFLINE")

    # Heatmap Strip
    h_cols = st.columns(len(TIMEFRAMES))
    for i, (label, (df, status)) in enumerate(sync_report.items()):
        color = "#888888"
        if "BUY" in status or "UP" in status: color = "#00FFAA"
        elif "SELL" in status or "DOWN" in status: color = "#FF4B4B"
        elif "DIV" in status: color = "#00AAFF"
        with h_cols[i]:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span style='font-size:16px; font-weight:bold;'>{label}</span><br>"
                f"<span style='color:{color}; font-size:12px; font-weight:bold;'>{status}</span>"
                f"</div>", 
                unsafe_allow_html=True
            )

    st.divider()

    score = calculate_confluence(sync_report)
    summary = generate_trade_summary(score, sync_report)
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("CONFLUENCE", f"{score:+.1f}%", delta_color="normal")
    with c2:
        st.info(summary)

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
                    if selected_type == 'candle':
                        # NEW: Plot CQWMA and Floating Levels
                        if 'CQWMA' in data.columns:
                            # Color logic for CQWMA line: Green if Trend Up, Red if Trend Down, else Gray
                            c_color = '#888888' #'#00FFAA' if data['CQWMA_Color'].iloc[-1] == 1 else ('#FF4B4B' if data['CQWMA_Color'].iloc[-1] == 2 else '#888888')
                            apds.append(mpf.make_addplot(data['CQWMA'], color=c_color, width=1.5, alpha=0.9))
                            apds.append(mpf.make_addplot(data['CQWMA_Up'], color='#00FFAA', linestyle='--', width=0.7, alpha=0.3))
                            apds.append(mpf.make_addplot(data['CQWMA_Down'], color='#FF4B4B', linestyle='--', width=0.7, alpha=0.3))

                        # TD Setup (9)
                        if 'Setup_Signal' in data.columns:
                            b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.99, np.nan)
                            s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.01, np.nan)
                            if not np.all(np.isnan(b9)): apds.append(mpf.make_addplot(b9, type='scatter', marker='^', color='#00FFAA', markersize=30))
                            if not np.all(np.isnan(s9)): apds.append(mpf.make_addplot(s9, type='scatter', marker='v', color='#FF4B4B', markersize=30))

                        # TD Countdown (13)
                        if 'Countdown_Signal' in data.columns:
                            b13 = np.where(data['Countdown_Signal'] == 1, data['Low'] * 0.975, np.nan)
                            s13 = np.where(data['Countdown_Signal'] == -1, data['High'] * 1.025, np.nan)
                            if not np.all(np.isnan(b13)): apds.append(mpf.make_addplot(b13, type='scatter', marker=r'$13$', color='#00AAFF', markersize=80))
                            if not np.all(np.isnan(s13)): apds.append(mpf.make_addplot(s13, type='scatter', marker=r'$13$', color='#FFAA00', markersize=80))

                        # RSI Divergence
                        if 'Signal' in data.columns:
                            div_bull = np.where(data['Signal'] == 1, data['Low'] * 0.96, np.nan)
                            if not np.all(np.isnan(div_bull)): apds.append(mpf.make_addplot(div_bull, type='scatter', marker='*', color='#00FFAA', markersize=60))

                    plot_kwargs = {
                        "type": selected_type, "style": style_sel, "volume": show_vol,
                        "figsize": (5, 3.5), "tight_layout": True, "returnfig": True, "xrotation": 45
                    }
                    if apds: plot_kwargs["addplot"] = apds

                    # Dynamic Support/Resistance
                    recent_closed = data[:-1]
                    resistance = recent_closed['High'].rolling(window=20).max().iloc[-1]
                    support = recent_closed['Low'].rolling(window=20).min().iloc[-1]
                    current_price = data['Close'].iloc[-1]

                    fig, axlist = mpf.plot(
                        data, **plot_kwargs,
                        hlines=dict(hlines=[current_price, resistance, support], 
                                   colors=['#888888', '#FF4B4B', '#00FFAA'], 
                                   linestyle=['dotted', 'dashed', 'dashed'], 
                                   linewidths=[1.5, 1, 1], alpha=0.7)
                    )
                    
                    for ax in axlist: ax.grid(False)
                    xmin, xmax = axlist[0].get_xlim()
                    axlist[0].set_xlim(xmin, xmax + 8)
                    
                    axlist[0].text(0.5, 0.5, label, transform=axlist[0].transAxes, 
                                 fontsize=70, fontweight='bold', color='white', 
                                 alpha=0.15, ha='center', va='center', zorder=0)
                    
                    if "BUY" in status or "SELL" in status:
                        rect_color = "#00FFAA" if "BUY" in status else "#FF4B4B"
                        fig.patch.set_linewidth(4)
                        fig.patch.set_edgecolor(rect_color)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
                    st.image(buf, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning(f"{label} No Data")
else:
    st.info("👈 Enter a ticker symbol in the sidebar.")

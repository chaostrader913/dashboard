import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import io
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Essential for web-server stability

try:
    from utils.data_loader import fetch_data
    from utils.indicators import apply_td_sequential, apply_rsi_divergence
except ImportError:
    st.error("Missing utility files. Ensure 'utils/data_loader.py' and 'utils/indicators.py' exist.")
    st.stop()

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Multi-Timeframe Analyzer")

# st.markdown("""
# <style>
#     .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem; max-width: 100%; }
    
#     /* 1. Remove horizontal space between columns */
#     div[data-testid="stHorizontalBlock"] { gap: 0.10rem !important; }
    
#     /* 2. Remove padding inside the columns */
#     div[data-testid="column"] { padding: 0.1rem !important; }
    
#     /* 3. Pull the rows closer together vertically */
#     div[data-testid="stImage"] { margin-bottom: -1.2rem !important; }
# </style>
# """, unsafe_allow_html=True)

# --- 2. Logic Engines ---

def get_signal_status(df):
    if df is None or df.empty: return "⚪ OFFLINE"
    recent = df.tail(3)
    
    if 'Countdown_Signal' in recent.columns:
        last_val = recent['Countdown_Signal'].replace(0, np.nan).ffill().iloc[-1]
        if last_val == 1: return "🔥 TD13 BUY"
        if last_val == -1: return "💀 TD13 SELL"

    if 'Setup_Signal' in recent.columns:
        last_val = recent['Setup_Signal'].replace(0, np.nan).ffill().iloc[-1]
        if last_val == 1: return "🟢 TD9 BUY"
        if last_val == -1: return "🔴 TD9 SELL"
        
    if 'Signal' in recent.columns and recent['Signal'].any():
        return "🔵 RSI DIV"
        
    return "⚪ NEUTRAL"

def calculate_confluence(sync_report):
    weights = { "30M": 1, "1H": 2, "4H": 5, "D": 10, "W": 20, "M": 30 }
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
    micro_signals = [s for l, (_, s) in sync_report.items() if l in ["30M", "1H", "4H"]]
    
    macro_bull = any("BUY" in s for s in macro_signals)
    macro_bear = any("SELL" in s for s in macro_signals)
    micro_bull = any("BUY" in s for s in micro_signals)
    micro_bear = any("SELL" in s for s in micro_signals)
    
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
    
    # Chart Type Selector
    chart_sel = st.selectbox("CHART TYPE", ['Candlestick', 'Point & Figure', 'Renko'], index=0)
    
    # Theme Selector (Index 2 sets 'mike' as the default)
    style_sel = st.selectbox("THEME", ['nightclouds', 'yahoo', 'mike', 'blueskies'], index=2)
    
    st.divider()
    show_vol = st.checkbox("Show Volume", value=False)
    # Auto-Refresh Toggle
    st.subheader("⏱️ Live Sync")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    refresh_rate = st.slider("Refresh Interval (Seconds)", min_value=30, max_value=300, value=60, step=30)
    
    if auto_refresh:
        # interval takes milliseconds, so we multiply the seconds by 1000
        st_autorefresh(interval=refresh_rate * 1000, limit=None, key="mtf_refresh")
        st.caption(f"🟢 Polling API every {refresh_rate}s")
    else:
        st.caption("🔴 Live Sync Paused")

# Map UI selection to mplfinance kwargs
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
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span style='font-size:16px; font-weight:bold;'>{label}</span><br>"
                f"<span style='color:{color}; font-size:14px; font-weight:bold;'>{status.split()[-1]}</span>"
                f"</div>", 
                unsafe_allow_html=True
            )

    st.divider()

    # Sentiment Score
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
                    # SAFETY CHECK: Only plot overlays on Time-Based Candlestick charts
                    if selected_type == 'candle':
                        
                        # 1. TD Setup (9) - Closest to the candle
                        if 'Setup_Signal' in data.columns:
                            b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.99, np.nan)
                            s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.01, np.nan)
                            if not np.all(np.isnan(b9)): apds.append(mpf.make_addplot(b9, type='scatter', marker='^', color='#00FFAA', markersize=30))
                            if not np.all(np.isnan(s9)): apds.append(mpf.make_addplot(s9, type='scatter', marker='v', color='#FF4B4B', markersize=30))

                        # 2. TD Countdown (13) - Pushed slightly further out
                        if 'Countdown_Signal' in data.columns:
                            b13 = np.where(data['Countdown_Signal'] == 1, data['Low'] * 0.975, np.nan)
                            s13 = np.where(data['Countdown_Signal'] == -1, data['High'] * 1.025, np.nan)
                            if not np.all(np.isnan(b13)): apds.append(mpf.make_addplot(b13, type='scatter', marker=r'$13$', color='#00AAFF', markersize=80))
                            if not np.all(np.isnan(s13)): apds.append(mpf.make_addplot(s13, type='scatter', marker=r'$13$', color='#FFAA00', markersize=80))

                        # 3. RSI Divergence - Pushed furthest out, marked with a Star
                        if 'Signal' in data.columns:
                            div_bull = np.where(data['Signal'] == 1, data['Low'] * 0.96, np.nan)
                            if not np.all(np.isnan(div_bull)): apds.append(mpf.make_addplot(div_bull, type='scatter', marker='*', color='#00FFAA', markersize=60))

                    # Combine args
                    plot_kwargs = {
                        "type": selected_type, 
                        "style": style_sel, 
                        "volume": show_vol,
                        "figsize": (5, 3.5),
                        "tight_layout": True, 
                        "returnfig": True,
                        "xrotation": 45
                        #"axisoff": True
                    }
                    if apds: plot_kwargs["addplot"] = apds

                    # Generate the Chart
                    # Get the absolute latest closing price
                    current_price = data['Close'].iloc[-1]

                    # Generate the Chart with the price line added
                    fig, axlist = mpf.plot(
                        data, 
                        **plot_kwargs,
                        hlines=dict(hlines=[current_price], colors=['#888888'], linestyle='dotted', linewidths=1.5, alpha=0.7)
                    )                    

                    
                    # Kill empty space before/after data
                    # axlist[0].margins(x=0) 
                    
                    # Custom inner title block
                    # 🔥 THE WATERMARK
                    axlist[0].text(
                        0.5, 0.5, label,                  # X=0.5, Y=0.5 is the exact center
                        transform=axlist[0].transAxes,    # Use relative axis coordinates
                        fontsize=70,                      # Massive font size
                        fontweight='bold',
                        color='white',
                        alpha=0.20,                       # Make it 94% transparent
                        ha='center',                      # Horizontally center the text
                        va='center',                      # Vertically center the text
                        zorder=0                          # CRITICAL: Pushes the text behind the candles
                    )
                    
                    # Highlight Active signals with a border
                    if "BUY" in status or "SELL" in status:
                        rect_color = "#00FFAA" if "BUY" in status else "#FF4B4B"
                        fig.patch.set_linewidth(4)
                        fig.patch.set_edgecolor(rect_color)

                    # Aggressively crop invisible chart margins
                    buf = io.BytesIO()
                    fig.savefig(
                        buf, 
                        format="png", 
                        dpi=100, 
                        facecolor=fig.get_facecolor(), 
                        bbox_inches='tight', 
                        pad_inches=0
                    )
                    st.image(buf, use_container_width=True)
                    plt.close(fig)
                    
                else:
                    st.warning(f"{label} No Data")

else:
    st.info("👈 Enter a ticker symbol in the sidebar.")

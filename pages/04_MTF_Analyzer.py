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

# Custom CSS - Updated for Theme Compatibility
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 1.5rem; padding-right: 1.5rem; max-width: 100%; }
    
    /* Adaptable Metric Box */
    .metric-container { 
        background-color: rgba(128, 128, 128, 0.1); 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Ensure text is legible on both light/dark backgrounds */
    .sync-label { font-weight: bold; color: inherit; }
    .status-text { font-size: 0.75rem; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# --- 2. Logic Engines ---

def get_signal_status(df):
    """Detects the most recent active signal in a dataframe."""
    if df is None or df.empty: return "⚪ OFFLINE"
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
    """Calculates a weighted sentiment score for the 6 core timeframes."""
    weights = {
        "30M": 5, "1H": 15, "4H": 25, "D": 40, "W": 60, "M": 80
    }
    total_score = 0
    max_possible = sum(weights.values())
    
    for label, (df, status) in sync_report.items():
        if df is None: continue
        w = weights.get(label, 10)
        if "BUY" in status: total_score += w
        elif "SELL" in status: total_score -= w
            
    return np.clip((total_score / max_possible) * 100, -100, 100)

def generate_trade_summary(score, sync_report):
    macro_signals = [s for l, (_, s) in sync_report.items() if l in ["D", "W", "M"]]
    macro_bull = any("BUY" in s for s in macro_signals)
    
    if score > 65: return "🚀 CONFLUENT UPTREND: Strong trend stacking. High probability of continuation."
    if score < -65: return "⚠️ SYSTEMIC WEAKNESS: Heavy selling pressure. Avoid long entries."
    if macro_bull and score < 0: return "⚖️ MEAN REVERSION: Macro trend is Bullish. Current micro weakness is a pullback."
    if abs(score) < 15: return "🌀 COMPRESSION: Market is in a squeeze. Look for the 1H/4H breakout."
    return "🔎 MONITORING: Mixed alignment. Watch the 1H 'Anchor' for trend confirmation."

# --- 3. Configuration (6 Core Resolutions) ---
TIMEFRAMES = [
    {"interval": "30m", "period": "5d",   "label": "30M"},
    {"interval": "60m", "period": "1wk",  "label": "1H"},
    {"interval": "1h",  "period": "1mo",  "label": "4H"}, 
    {"interval": "1d",  "period": "6mo",  "label": "D"},
    {"interval": "1wk", "period": "2y",   "label": "W"}, 
    {"interval": "1mo", "period": "5y",   "label": "M"}, 
]

# --- 4. Sidebar ---
with st.sidebar:
    st.header("🔎 ASSET SYNC")
    ticker = st.text_input("SYMBOL", value="NVDA").upper()
    style_sel = st.selectbox("THEME", ['nightclouds', 'yahoo', 'mike', 'blueskies'], index=0)
    st.divider()
    show_vol = st.checkbox("Show Volume", value=False)
    st.caption("Core Multi-Timeframe Analysis (30M - Monthly).")

# --- 5. Main Content Execution ---
if ticker:
    st.markdown(f"### 🧭 MTF FRACTAL SYNC: {ticker}")
    
    sync_report = {}
    with st.spinner(f"Synchronizing 6 resolutions for {ticker}..."):
        for tf in TIMEFRAMES:
            raw_data = fetch_data(ticker, tf['interval'], tf['period'])
            
            if raw_data is not None and not raw_data.empty:
                data = raw_data.copy()
                
                # Surgical Sanitization
                if isinstance(data.columns, pd.MultiIndex):
                    try: data = data.xs(ticker, axis=1, level=1)
                    except: data.columns = data.columns.get_level_values(0)
                
                # Duplicate & Timezone Cleaning
                data.index = pd.to_datetime(data.index).tz_localize(None)
                if tf['interval'] in ['1d', '1wk', '1mo']:
                    data.index = data.index.normalize()
                data = data[~data.index.duplicated(keep='last')]
                
                # Numeric Enforcement
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col].squeeze(), errors='coerce')
                
                data = data.dropna(subset=['Close'])
                
                if len(data) > 15:
                    data = apply_td_sequential(data)
                    data = apply_rsi_divergence(data)
                    sync_report[tf['label']] = (data, get_signal_status(data))
                else:
                    sync_report[tf['label']] = (None, "⚪ DATA SHORT")
            else:
                sync_report[tf['label']] = (None, "⚪ OFFLINE")

    # --- TOP SYNC STRIP (HEATMAP) ---
    # Using columns with standard markdown to let Streamlit handle text color
    h_cols = st.columns(len(TIMEFRAMES))
    for i, (label, (df, status)) in enumerate(sync_report.items()):
        # Determine color for status only, let label stay default theme color
        color = "#888888" 
        if "BUY" in status: color = "#00B37E" # Slightly darker green for light mode
        elif "SELL" in status: color = "#E91E63" # Slightly darker red for light mode
        elif "DIV" in status: color = "#2196F3" # Standard blue
        
        with h_cols[i]:
            st.markdown(f"**{label}**")
            st.markdown(f"<span style='color:{color}; font-size:10px; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)

    st.divider()

    # --- SCOREBOARD ---
    score = calculate_confluence(sync_report)
    summary = generate_trade_summary(score, sync_report)
    
    # Gauge Color Logic
    gauge_color = "#00B37E" if score > 20 else "#E91E63" if score < -20 else "#888888"
    
    m_left, m_right = st.columns([1, 2])
    with m_left:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 5px solid {gauge_color};">
            <p style="margin:0; font-size: 14px; opacity: 0.8;">MACRO CONFLUENCE</p>
            <p style="margin:0; font-family:monospace; font-size: 32px; font-weight:bold; color: {gauge_color};">
                {score:+.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with m_right:
        st.markdown(f"""
        <div class="metric-container">
            <p style="margin: 0; font-size: 15px; font-weight: 500;">{summary}</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("") 

    # --- 6. THE GRID (2x3 Layout) ---
    cols_per_row = 3
    num_rows = 2

    for r in range(num_rows):
        grid_cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < len(TIMEFRAMES):
                tf_label = TIMEFRAMES[idx]['label']
                data, status = sync_report[tf_label]
                
                with grid_cols[c]:
                    if data is not None and len(data) > 5:
                        apds = []
                        if 'Setup_Signal' in data.columns:
                            b9 = np.full(len(data), np.nan)
                            s9 = np.full(len(data), np.nan)
                            b_mask = (data['Setup_Signal'] == 1).values
                            s_mask = (data['Setup_Signal'] == -1).values
                            b9[b_mask] = data['Low'].values[b_mask] * 0.98
                            s9[s_mask] = data['High'].values[s_mask] * 1.02
                            if not np.all(np.isnan(b9)): apds.append(mpf.make_addplot(b9, type='scatter', marker=r'$9$', color='#00B37E', markersize=30))
                            if not np.all(np.isnan(s9)): apds.append(mpf.make_addplot(s9, type='scatter', marker=r'$9$', color='#E91E63', markersize=30))
                        
                        valid_styles = ['binance', 'blueskies', 'brasil', 'charles', 'checkers', 'classic', 'default', 'mike', 'nightclouds', 'sas', 'starsandstripes', 'yahoo']
                        
                        # Use 'blueskies' or 'yahoo' if the user is in light mode for better chart visibility
                        safe_style = style_sel if style_sel in valid_styles else 'charles'

                        try:
                            v_on = show_vol if 'Volume' in data.columns and not data['Volume'].isna().all() else False
                            fig, axlist = mpf.plot(
                                data, type='candle', style=safe_style, volume=v_on,
                                returnfig=True, figsize=(5, 3.5), tight_layout=True,
                                addplot=apds if apds else None, xrotation=0, axisoff=True
                            )
                            # Title inside chart - using darker colors for labels
                            axlist[0].set_title(tf_label, fontsize=14, color='gray', loc='left', pad=-20)
                            
                            if "BUY" in status or "SELL" in status:
                                b_color = "#00B37E" if "BUY" in status else "#E91E63"
                                rect = plt.Rectangle((0,0), 1, 1, fill=False, color=b_color, lw=3, transform=fig.transFigure)
                                fig.patches.append(rect)

                            buf = io.BytesIO()
                            # facecolor='none' allows the chart to blend with the app background
                            fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', facecolor='none')
                            st.image(buf, use_container_width=True)
                            plt.close(fig)
                        except:
                            st.error(f"Render Error {tf_label}")
                    else:
                        st.error(f"OFFLINE: {tf_label}")
else:
    st.info("👈 Enter a ticker symbol in the sidebar to begin analysis.")

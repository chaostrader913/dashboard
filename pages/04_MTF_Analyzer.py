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

# Custom CSS for a clean, modern look on both themes
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 0rem; max-width: 95%; }
    
    /* Modern Metric Containers */
    .info-box { 
        background-color: rgba(128, 128, 128, 0.05); 
        padding: 1rem; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Sync Strip Styling */
    .sync-card {
        text-align: center;
        padding: 8px;
        border-radius: 6px;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* Chart Container Styling */
    .chart-header {
        font-size: 1.2rem;
        font-weight: 800;
        color: #888;
        margin-bottom: -10px;
        margin-left: 5px;
    }
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
    weights = {"30M": 5, "1H": 15, "4H": 25, "D": 40, "W": 60, "M": 80}
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
    
    if score > 65: return "🚀 CONFLUENT UPTREND: Trend stacking detected. High probability of continuation."
    if score < -65: return "⚠️ SYSTEMIC WEAKNESS: Heavy selling pressure. Avoid long entries."
    if macro_bull and score < 0: return "⚖️ MEAN REVERSION: Macro trend is Bullish. Current micro weakness is a pullback."
    if abs(score) < 15: return "🌀 COMPRESSION: Market is in a squeeze. Look for the 1H/4H breakout."
    return "🔎 MONITORING: Mixed alignment. Watch the 1H 'Anchor' for trend confirmation."

# --- 3. Configuration ---
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
    style_sel = st.selectbox("CHART THEME", ['nightclouds', 'yahoo', 'mike', 'blueskies'], index=0)
    st.divider()
    show_vol = st.checkbox("Show Volume", value=False)
    st.caption("Core Multi-Timeframe Analysis.")

# --- 5. Main Content Execution ---
if ticker:
    st.subheader(f"🧭 MTF FRACTAL SYNC: {ticker}")
    
    sync_report = {}
    with st.spinner(f"Synchronizing..."):
        for tf in TIMEFRAMES:
            raw_data = fetch_data(ticker, tf['interval'], tf['period'])
            
            if raw_data is not None and not raw_data.empty:
                data = raw_data.copy()
                
                # MultiIndex & Timezone Handling
                if isinstance(data.columns, pd.MultiIndex):
                    try: data = data.xs(ticker, axis=1, level=1)
                    except: data.columns = data.columns.get_level_values(0)
                
                # Fix for "Render Error D": Force naive datetime and drop duplicates
                data.index = pd.to_datetime(data.index).tz_localize(None)
                if tf['interval'] in ['1d', '1wk', '1mo']:
                    data.index = data.index.normalize()
                data = data[~data.index.duplicated(keep='last')]
                
                # Numeric Enforcement
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col].squeeze(), errors='coerce')
                
                data = data.dropna(subset=['Close'])
                
                if len(data) > 10:
                    data = apply_td_sequential(data)
                    data = apply_rsi_divergence(data)
                    sync_report[tf['label']] = (data, get_signal_status(data))
                else:
                    sync_report[tf['label']] = (None, "⚪ DATA SHORT")
            else:
                sync_report[tf['label']] = (None, "⚪ OFFLINE")

    # --- TOP SYNC STRIP ---
    h_cols = st.columns(len(TIMEFRAMES))
    for i, (label, (df, status)) in enumerate(sync_report.items()):
        color = "#888888" 
        if "BUY" in status: color = "#00B37E" 
        elif "SELL" in status: color = "#E91E63" 
        elif "DIV" in status: color = "#2196F3" 
        
        with h_cols[i]:
            st.markdown(f"""
            <div class="sync-card">
                <div style="font-weight:bold; font-size:12px;">{label}</div>
                <div style="color:{color}; font-size:10px; font-weight:bold;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # --- SCOREBOARD ---
    score = calculate_confluence(sync_report)
    summary = generate_trade_summary(score, sync_report)
    gauge_color = "#00B37E" if score > 20 else "#E91E63" if score < -20 else "#888888"
    
    m_left, m_right = st.columns([1, 2])
    with m_left:
        st.markdown(f"""
        <div class="info-box" style="border-left: 5px solid {gauge_color};">
            <div style="font-size: 0.8rem; opacity: 0.7;">MACRO CONFLUENCE</div>
            <div style="font-family:monospace; font-size: 1.8rem; font-weight:900; color:{gauge_color};">{score:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_right:
        st.markdown(f"""
        <div class="info-box">
            <div style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 4px;">STRATEGY ADVISOR</div>
            <div style="font-weight: 500;">{summary}</div>
        </div>
        """, unsafe_allow_html=True)

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
                    # Move title to Streamlit for sharpness
                    st.markdown(f'<div class="chart-header">{tf_label}</div>', unsafe_allow_html=True)
                    
                    if data is not None and len(data) > 5:
                        apds = []
                        if 'Setup_Signal' in data.columns:
                            b9, s9 = np.full(len(data), np.nan), np.full(len(data), np.nan)
                            b_mask, s_mask = (data['Setup_Signal'] == 1).values, (data['Setup_Signal'] == -1).values
                            b9[b_mask], s9[s_mask] = data['Low'].values[b_mask] * 0.98, data['High'].values[s_mask] * 1.02
                            if not np.all(np.isnan(b9)): apds.append(mpf.make_addplot(b9, type='scatter', marker=r'$9$', color='#00B37E', markersize=35))
                            if not np.all(np.isnan(s9)): apds.append(mpf.make_addplot(s9, type='scatter', marker=r'$9$', color='#E91E63', markersize=35))
                        
                        try:
                            v_on = show_vol if 'Volume' in data.columns and not data['Volume'].isna().all() else False
                            fig, axlist = mpf.plot(
                                data, type='candle', style=style_sel, volume=v_on,
                                returnfig=True, figsize=(6, 4.2), tight_layout=True,
                                addplot=apds if apds else None, xrotation=0, axisoff=True
                            )
                            
                            # Signal Border
                            if "BUY" in status or "SELL" in status:
                                b_color = "#00B37E" if "BUY" in status else "#E91E63"
                                rect = plt.Rectangle((0,0), 1, 1, fill=False, color=b_color, lw=4, transform=fig.transFigure)
                                fig.patches.append(rect)

                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=110, bbox_inches='tight', facecolor='none')
                            st.image(buf, use_container_width=True)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Render Error {tf_label}")
                    else:
                        st.error(f"OFFLINE: {tf_label}")
else:
    st.info("👈 Enter a ticker symbol in the sidebar to begin analysis.")

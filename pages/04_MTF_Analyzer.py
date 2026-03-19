import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import io
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

# --- IMPORT GLOBALS ---
try:
    from utils.data_loader import fetch_data
    from utils.indicators import apply_td_sequential, apply_rsi_divergence, apply_corrected_qwma
except ImportError:
    st.error("Missing utility files in 'utils/'.")
    st.stop()

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="MTF Fractal Sync")

# Custom CSS for High-Contrast White Theme
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; max-width: 98%; background-color: white; }
    .info-box { 
        background-color: #fcfcfc; 
        padding: 1.2rem; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .sync-card {
        text-align: center;
        padding: 10px;
        border-radius: 6px;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Logic Engines ---

def get_signal_status(df):
    """Detects active signals in the tail of the data."""
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
    weights = {"30M": 5, "1H": 15, "4H": 25, "D": 40, "W": 60, "M": 80}
    total_score = 0
    max_active = 0
    for label, (df, status) in sync_report.items():
        if df is None: continue
        w = weights.get(label, 10)
        max_active += w
        if "BUY" in status: total_score += w
        elif "SELL" in status: total_score -= w
    return np.clip((total_score / max_active) * 100, -100, 100) if max_active > 0 else 0

def generate_trade_summary(score, sync_report):
    macro_signals = [s for l, (_, s) in sync_report.items() if l in ["D", "W", "M"]]
    macro_bull = any("BUY" in s for s in macro_signals)
    if score > 65: return "🚀 CONFLUENT UPTREND: Trend stacking detected. High continuation probability."
    if score < -65: return "⚠️ SYSTEMIC WEAKNESS: Heavy selling pressure. Avoid long entries."
    if macro_bull and score < 0: return "⚖️ MEAN REVERSION: Macro trend is Bullish. Current dip is a pullback."
    return "🔎 MONITORING: Mixed alignment. Watch the 1H Anchor for direction."

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
    chart_sel = st.selectbox("CHART TYPE", ['Candlestick', 'Point & Figure', 'Renko'], index=0)
    st.divider()
    show_vol = st.checkbox("Show Volume", value=False)
    show_qwma = st.checkbox("Corrected QWMA", value=True)
    
    st.subheader("⏱️ Live Sync")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    if auto_refresh:
        refresh_rate = st.slider("Seconds", 30, 300, 60, 30)
        st_autorefresh(interval=refresh_rate * 1000, key="mtf_sync")

# Mapping
chart_type_map = {'Candlestick': 'candle', 'Point & Figure': 'pnf', 'Renko': 'renko'}
selected_type = chart_type_map[chart_sel]

# --- 5. Execution ---
if ticker:
    st.subheader(f"🧭 {ticker} Fractal Sync")
    
    sync_report = {}
    with st.spinner("Synchronizing..."):
        for tf in TIMEFRAMES:
            raw_data = fetch_data(ticker, tf['interval'], tf['period'])
            if raw_data is not None and not raw_data.empty:
                data = raw_data.copy()
                if isinstance(data.columns, pd.MultiIndex):
                    try: data = data.xs(ticker, axis=1, level=1)
                    except: data.columns = data.columns.get_level_values(0)
                
                # Sanitization for mpf
                data.index = pd.to_datetime(data.index).tz_localize(None)
                if tf['interval'] in ['1d', '1wk', '1mo']: data.index = data.index.normalize()
                data = data[~data.index.duplicated(keep='last')].sort_index()
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns: data[col] = pd.to_numeric(data[col].squeeze(), errors='coerce')
                data = data.dropna(subset=['Close'])

                if len(data) > 20:
                    data = apply_td_sequential(data)
                    data = apply_rsi_divergence(data)
                    if show_qwma: data = apply_corrected_qwma(data)
                    sync_report[tf['label']] = (data, get_signal_status(data))
                else:
                    sync_report[tf['label']] = (None, "⚪ DATA SHORT")
            else:
                sync_report[tf['label']] = (None, "⚪ OFFLINE")

    # Heatmap Strip
    h_cols = st.columns(len(TIMEFRAMES))
    for i, (label, (df, status)) in enumerate(sync_report.items()):
        color = "#495057"
        if "BUY" in status: color = "#008a5d"
        elif "SELL" in status: color = "#c92a2a"
        elif "DIV" in status: color = "#1c7ed6"
        with h_cols[i]:
            st.markdown(f"<div class='sync-card'><div style='font-weight:900;'>{label}</div><div style='color:{color}; font-size:11px; font-weight:800;'>{status}</div></div>", unsafe_allow_html=True)

    st.divider()

    # Scoreboard
    score = calculate_confluence(sync_report)
    summary = generate_trade_summary(score, sync_report)
    gauge_color = "#008a5d" if score > 20 else "#c92a2a" if score < -20 else "#495057"
    
    m1, m2 = st.columns([1, 2])
    with m1:
        st.markdown(f"<div class='info-box' style='border-left: 6px solid {gauge_color};'><div style='font-size: 0.85rem; font-weight: 700; color: #adb5bd;'>CONFLUENCE</div><div style='font-family:monospace; font-size: 2.2rem; font-weight:900; color:{gauge_color};'>{score:+.1f}%</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='info-box'><div style='font-size: 0.85rem; font-weight: 700; color: #adb5bd;'>STRATEGY</div><div style='font-weight:600;'>{summary}</div></div>", unsafe_allow_html=True)

    # --- 6. The Grid ---
    rows = [list(sync_report.items())[i:i+3] for i in range(0, len(sync_report), 3)]
    for row_items in rows:
        cols = st.columns(3)
        for i, (label, (data, status)) in enumerate(row_items):
            with cols[i]:
                if data is not None:
                    apds = []
                    if selected_type == 'candle':
                        # QWMA
                        if show_qwma and 'CQWMA' in data.columns:
                            qwma_g = np.where(data['CQWMA_Color'] == 1, data['CQWMA'], np.nan)
                            qwma_r = np.where(data['CQWMA_Color'] == 2, data['CQWMA'], np.nan)
                            apds.append(mpf.make_addplot(qwma_g, color='#008a5d', width=1.8))
                            apds.append(mpf.make_addplot(qwma_r, color='#c92a2a', width=1.8))
                            apds.append(mpf.make_addplot(data['CQWMA_Up'], color='#008a5d', width=0.7, linestyle='dashed', alpha=0.2))
                            apds.append(mpf.make_addplot(data['CQWMA_Down'], color='#c92a2a', width=0.7, linestyle='dashed', alpha=0.2))

                        # TD Signals
                        if 'Setup_Signal' in data.columns:
                            b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.985, np.nan)
                            s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.015, np.nan)
                            if not np.all(np.isnan(b9)): apds.append(mpf.make_addplot(b9, type='scatter', marker='^', color='#008a5d', markersize=40))
                            if not np.all(np.isnan(s9)): apds.append(mpf.make_addplot(s9, type='scatter', marker='v', color='#c92a2a', markersize=40))
                        
                        if 'Countdown_Signal' in data.columns:
                            b13 = np.where(data['Countdown_Signal'] == 1, data['Low'] * 0.97, np.nan)
                            s13 = np.where(data['Countdown_Signal'] == -1, data['High'] * 1.03, np.nan)
                            if not np.all(np.isnan(b13)): apds.append(mpf.make_addplot(b13, type='scatter', marker=r'$13$', color='#1c7ed6', markersize=70))
                            if not np.all(np.isnan(s13)): apds.append(mpf.make_addplot(s13, type='scatter', marker=r'$13$', color='#fd7e14', markersize=70))

                    # S/R Calc
                    res = data['High'].iloc[:-1].rolling(20).max().iloc[-1]
                    sup = data['Low'].iloc[:-1].rolling(20).min().iloc[-1]
                    curr = data['Close'].iloc[-1]

                    fig, axlist = mpf.plot(
                        data, type=selected_type, style='mike', volume=show_vol and label not in ['W', 'M'],
                        addplot=apds if apds else None, returnfig=True, figsize=(8, 5.5),
                        tight_layout=True, axisoff=True,
                        hlines=dict(hlines=[curr, res, sup], colors=['#adb5bd', '#c92a2a', '#008a5d'], linestyle=['dotted', 'dashed', 'dashed'], linewidths=[1, 0.8, 0.8], alpha=0.5),
                        scale_padding=dict(left=0.1, right=0.1, top=1.2, bottom=1.2)
                    )
                    
                    # Watermark & Padding
                    xmin, xmax = axlist[0].get_xlim()
                    axlist[0].set_xlim(xmin, xmax + 6)
                    axlist[0].text(0.5, 0.5, label, transform=axlist[0].transAxes, fontsize=80, fontweight='black', color='#dee2e6', alpha=0.15, ha='center', va='center', zorder=0)
                    
                    if "BUY" in status or "SELL" in status:
                        rect_color = "#008a5d" if "BUY" in status else "#c92a2a"
                        fig.patch.set_linewidth(6)
                        fig.patch.set_edgecolor(rect_color)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=130, facecolor='white', bbox_inches='tight', pad_inches=0)
                    st.image(buf, use_container_width=True)
                    plt.close(fig)
                else:
                    st.error(f"{label} Offline")

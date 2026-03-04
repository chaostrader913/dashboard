import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- IMPORT GLOBALS ---
from utils.data_loader import fetch_data
from utils.indicators import apply_td_sequential, apply_rsi_divergence

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Multi-Timeframe Analyzer")

# Custom CSS for Plotly Grid
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 1.5rem; padding-right: 1.5rem; max-width: 100%; }
    .stPlotlyChart { border: 1px solid #2b3040; border-radius: 8px; overflow: hidden; }
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
    """Calculates weighted sentiment for Macro/Mid timeframes."""
    weights = {
        "30M": 5, "1H": 10, "90M": 12, "4H": 18, 
        "D": 30, "2D": 35, "W": 50, "W-L": 55, "M": 70
    }
    total_score = 0
    max_possible = sum(weights.values())
    for label, (df, status) in sync_report.items():
        if df is None: continue
        w = weights.get(label, 5)
        if "BUY" in status: total_score += w
        elif "SELL" in status: total_score -= w
    return np.clip((total_score / max_possible) * 100, -100, 100)

def generate_trade_summary(score):
    if score > 65: return "🚀 MACRO TREND OVERLOAD: Heavy buying confluence on institutional timeframes."
    if score < -65: return "⚠️ SYSTEMIC LIQUIDATION: Multi-day selling pressure remains dominant."
    if score > 20: return "📈 ACCUMULATION: Mid-curve timeframes flipping bullish."
    if score < -20: return "📉 DISTRIBUTION: Macro supply hitting the tape."
    return "🌀 EQUILIBRIUM: Price is oscillating in a fractal range. No clear dominance."

# --- 3. Configuration (3x3 Grid - Shorter than 29m Removed) ---
TIMEFRAMES = [
    {"interval": "30m", "period": "5d",   "label": "30M"},
    {"interval": "60m", "period": "1wk",  "label": "1H"},
    {"interval": "90m", "period": "2wk",  "label": "90M"},
    {"interval": "1h",  "period": "1mo",  "label": "4H"}, # 1H proxy for 4H
    {"interval": "1d",  "period": "6mo",  "label": "D"},
    {"interval": "1d",  "period": "1y",   "label": "2D"}, # 1D proxy for 2D
    {"interval": "1wk", "period": "2y",   "label": "W"},
    {"interval": "1wk", "period": "5y",   "label": "W-L"},
    {"interval": "1mo", "period": "5y",   "label": "M"},
]

# --- 4. Sidebar ---
with st.sidebar:
    st.header("🔎 ASSET SYNC")
    ticker = st.text_input("SYMBOL", value="NVDA").upper()
    theme_color = st.color_picker("CHART ACCENT", "#4b4bff")
    st.divider()
    show_vol = st.checkbox("Show Volume", value=False)
    st.caption("Lower resolutions (<30m) filtered to reduce noise.")

# --- 5. Main Content Execution ---
if ticker:
    st.markdown(f"### 🧭 MTF FRACTAL SYNC: {ticker}")
    
    sync_report = {}
    with st.spinner(f"Synchronizing macro resolutions for {ticker}..."):
        for tf in TIMEFRAMES:
            raw_data = fetch_data(ticker, tf['interval'], tf['period'])
            if raw_data is not None and not raw_data.empty:
                data = raw_data.copy()
                if isinstance(data.columns, pd.MultiIndex):
                    try: data = data.xs(ticker, axis=1, level=1)
                    except: data.columns = data.columns.get_level_values(0)
                
                # Cleanup
                data.index = pd.to_datetime(data.index).tz_localize(None)
                if tf['interval'] in ['1d', '1wk', '1mo']: data.index = data.index.normalize()
                data = data[~data.index.duplicated(keep='last')]
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns: data[col] = pd.to_numeric(data[col].squeeze(), errors='coerce')
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
        if "BUY" in status: color = "#00FFAA"
        elif "SELL" in status: color = "#FF4B4B"
        elif "DIV" in status: color = "#00AAFF"
        with h_cols[i]:
            st.markdown(f"<div style='text-align:center;'><small><b>{label}</b></small><br><span style='color:{color}; font-size:9px;'>{status}</span></div>", unsafe_allow_html=True)

    st.divider()

    # --- SCOREBOARD ---
    score = calculate_confluence(sync_report)
    summary = generate_trade_summary(score)
    gauge_color = "#00FFAA" if score > 20 else "#FF4B4B" if score < -20 else "#888888"
    
    m_left, m_right = st.columns([1, 2])
    with m_left:
        st.markdown(f"""<div style="background-color: #1a1c24; padding: 15px; border-radius: 8px; border-left: 5px solid {gauge_color};"><h4 style="margin:0; color:{gauge_color}; font-size: 14px;">MACRO CONFLUENCE</h4><p style="margin:0; font-family:monospace; font-size: 28px; font-weight:bold;">{score:+.1f}%</p></div>""", unsafe_allow_html=True)
    with m_right:
        st.markdown(f"""<div style="background-color: #0e1117; padding: 15px; border: 1px solid #2b3040; border-radius: 8px; height: 80px; display: flex; align-items: center;"><p style="margin: 0; font-size: 14px;">{summary}</p></div>""", unsafe_allow_html=True)

    # --- 6. THE GRID (3x3 Plotly) ---
    cols_per_row = 3
    for r in range(3):
        grid_cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < len(TIMEFRAMES):
                tf_label = TIMEFRAMES[idx]['label']
                data, status = sync_report[tf_label]
                
                with grid_cols[c]:
                    if data is not None and len(data) > 5:
                        fig = go.Figure()

                        # Candlesticks
                        fig.add_trace(go.Candlestick(
                            x=data.index, open=data['Open'], high=data['High'],
                            low=data['Low'], close=data['Close'],
                            name=tf_label, increasing_line_color='#00FFAA', decreasing_line_color='#FF4B4B'
                        ))

                        # Signals (TD Sequential)
                        if 'Setup_Signal' in data.columns:
                            buy_9 = data[data['Setup_Signal'] == 1]
                            sell_9 = data[data['Setup_Signal'] == -1]
                            
                            fig.add_trace(go.Scatter(
                                x=buy_9.index, y=buy_9['Low'] * 0.98,
                                mode='markers+text', text="9", textposition="bottom center",
                                marker=dict(symbol='triangle-up', color='#00FFAA', size=10),
                                name='TD Buy 9', hoverinfo='skip'
                            ))
                            fig.add_trace(go.Scatter(
                                x=sell_9.index, y=sell_9['High'] * 1.02,
                                mode='markers+text', text="9", textposition="top center",
                                marker=dict(symbol='triangle-down', color='#FF4B4B', size=10),
                                name='TD Sell 9', hoverinfo='skip'
                            ))

                        # Signals (RSI Div)
                        if 'Signal' in data.columns:
                            rsi_div = data[data['Signal'] == 1]
                            fig.add_trace(go.Scatter(
                                x=rsi_div.index, y=rsi_div['Low'] * 0.96,
                                mode='markers', marker=dict(symbol='star', color='#00AAFF', size=8),
                                name='RSI Div', hoverinfo='skip'
                            ))

                        # Layout Tuning
                        fig.update_layout(
                            template='plotly_dark',
                            title=dict(text=f"<b>{tf_label}</b>", x=0.05, y=0.9, font=dict(size=14)),
                            xaxis_rangeslider_visible=False,
                            showlegend=False,
                            margin=dict(l=5, r=5, t=30, b=5),
                            height=350,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                            yaxis=dict(showgrid=True, gridcolor='#2b3040', zeroline=False, side='right')
                        )

                        # Signal Border via Annotation (Simulated)
                        if "BUY" in status or "SELL" in status:
                            b_color = "#00FFAA" if "BUY" in status else "#FF4B4B"
                            fig.update_layout(
                                shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color=b_color, width=3))]
                            )

                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.error(f"OFFLINE: {tf_label}")
else:
    st.info("👈 Enter a ticker symbol in the sidebar to synchronize resolutions.")

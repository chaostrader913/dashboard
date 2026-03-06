import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import from your new utils folder
from utils.data_loader import fetch_data
from utils.indicators import (
    apply_td_sequential, 
    apply_rsi_divergence, 
    apply_macd, 
    apply_bollinger_bands,
    apply_advanced_trendlines
)

# --- 1. Terminal UI Styling ---
st.markdown("### 📡 MODULE: ADVANCED SIGNAL SCANNER")
st.divider()

# --- 2. Command Row (User Inputs) ---
col1, col2, col3, col4, col5 = st.columns([1.5, 1, 2, 2, 1])
with col1:
    ticker = st.text_input("TARGET ASSET", value="BTC-USD").upper()
with col2:
    timeframe = st.selectbox("TIMEFRAME", options=["1d", "1h", "15m"], index=0)
with col3:
    overlays = st.multiselect(
        "PRICE OVERLAYS", 
        options=["TD Sequential", "Auto Trendlines", "Bollinger Bands"],
        default=["TD Sequential", "Auto Trendlines"]
    )
with col4:
    oscillators = st.multiselect(
        "OSCILLATORS", 
        options=["RSI Divergence", "MACD"],
        default=["RSI Divergence"]
    )
with col5:
    # New Slider for Trendline Density
    if "Auto Trendlines" in overlays:
        max_tl = st.slider("MAX TRENDLINES", min_value=1, max_value=10, value=3)
    else:
        max_tl = 3 # Default fallback
        
# --- 3. Data Processing & Indicator Application ---
with st.spinner(f"EXECUTING ALGORITHMS FOR {ticker}..."):
    data = fetch_data(ticker, interval=timeframe, period="1y" if timeframe == "1d" else "60d")

if data is None or data.empty:
    st.error(f"ERR: NO DATA FOUND FOR {ticker}.")
    st.stop()

# Apply selected mathematical models (Dataframe column additions)
if "TD Sequential" in overlays:
    data = apply_td_sequential(data)
if "Bollinger Bands" in overlays:
    data = apply_bollinger_bands(data)
if "RSI Divergence" in oscillators:
    data = apply_rsi_divergence(data)
if "MACD" in oscillators:
    data = apply_macd(data)

# --- 4. Dynamic Charting Engine ---
# --- 4. Dynamic Charting Engine ---
# Calculate how many rows we need based on selected oscillators
total_rows = 1 + len(oscillators)
row_heights = [0.6] + [0.4 / len(oscillators)] * len(oscillators) if oscillators else [1.0]

# 🔥 DYNAMIC SPECS: Enable a secondary Y-axis ONLY for the top Price/Volume row
specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * len(oscillators)

# INITIALIZE FIG
fig = make_subplots(
    rows=total_rows, cols=1, shared_xaxes=True, 
    vertical_spacing=0.03, row_heights=row_heights,
    specs=specs
)

# [ROW 1] BASE PRICE CHART
fig.add_trace(go.Candlestick(
    x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
    name="Price", increasing_line_color='#00FFAA', decreasing_line_color='#FF4B4B'
), row=1, col=1, secondary_y=False)

# 🔥 [ROW 1] THE VOLUME OVERLAY
if "Volume" in data.columns:
    # Color volume green if close > open, else red
    vol_colors = ['#00FFAA' if row['Close'] >= row['Open'] else '#FF4B4B' for idx, row in data.iterrows()]
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'], marker_color=vol_colors, 
        opacity=0.2, # Ghosted in the background
        name="Volume", hoverinfo='skip'
    ), row=1, col=1, secondary_y=True)

    # Trick Plotly into pinning the volume to the bottom 20% of the chart
    max_vol = data['Volume'].max()
    fig.update_yaxes(range=[0, max_vol * 5], showgrid=False, showticklabels=False, secondary_y=True, row=1, col=1)

# [ROW 1] OVERLAYS
if "Auto Trendlines" in overlays:
    upper_lines, lower_lines = apply_advanced_trendlines(data, window=5, pct_limit=10.0, breaks_limit=2)
    
    for i, line in enumerate(upper_lines):
        start_coord, end_coord = line
        fig.add_trace(go.Scatter(
            x=[start_coord[0], end_coord[0]], y=[start_coord[1], end_coord[1]], 
            mode='lines', name=f'Resist {i}', 
            line=dict(color='#FF4B4B', dash='dot', width=1.5),
            hoverinfo='skip', showlegend=False
        ), row=1, col=1, secondary_y=False)

    for i, line in enumerate(lower_lines):
        start_coord, end_coord = line
        fig.add_trace(go.Scatter(
            x=[start_coord[0], end_coord[0]], y=[start_coord[1], end_coord[1]], 
            mode='lines', name=f'Support {i}', 
            line=dict(color='#00FFAA', dash='dot', width=1.5),
            hoverinfo='skip', showlegend=False
        ), row=1, col=1, secondary_y=False)

if "Bollinger Bands" in overlays:
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Up', line=dict(color='#4B4BFF', width=1)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Low', line=dict(color='#4B4BFF', width=1)), row=1, col=1, secondary_y=False)

if "TD Sequential" in overlays:
    td_setup_buy = data[data.get('Setup_Signal', 0) == 1]
    td_setup_sell = data[data.get('Setup_Signal', 0) == -1]
    fig.add_trace(go.Scatter(x=td_setup_buy.index, y=td_setup_buy['Low'] * 0.98, mode='text', text='9', textfont=dict(color='#00FFAA', size=13, family='Courier New'), name='TD Buy 9'), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=td_setup_sell.index, y=td_setup_sell['High'] * 1.02, mode='text', text='9', textfont=dict(color='#FF4B4B', size=13, family='Courier New'), name='TD Sell 9'), row=1, col=1, secondary_y=False)

    td_cd_buy = data[data.get('Countdown_Signal', 0) == 1]
    td_cd_sell = data[data.get('Countdown_Signal', 0) == -1]
    fig.add_trace(go.Scatter(x=td_cd_buy.index, y=td_cd_buy['Low'] * 0.96, mode='text', text='13', textfont=dict(color='#00FFAA', size=16, family='Courier New'), name='TD Buy 13'), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=td_cd_sell.index, y=td_cd_sell['High'] * 1.04, mode='text', text='13', textfont=dict(color='#FF4B4B', size=16, family='Courier New'), name='TD Sell 13'), row=1, col=1, secondary_y=False)

# [ROWS 2+] OSCILLATORS
current_row = 2

if "RSI Divergence" in oscillators:
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='#00AAFF', width=1.5)), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF4B4B", line_width=1, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00FFAA", line_width=1, row=current_row, col=1)
    
    if 'Signal' in data.columns: 
        rsi_bulls = data[data['Signal'] == 1]
        fig.add_trace(go.Scatter(x=rsi_bulls.index, y=rsi_bulls['RSI'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00FFAA'), name='RSI Div'), row=current_row, col=1)
    current_row += 1

if "MACD" in oscillators:
    hist_colors = ['#00FFAA' if val >= 0 else '#FF4B4B' for val in data['MACD_Hist']]
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], marker_color=hist_colors, name='Histogram'), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='#00AAFF', width=1.5)), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal', line=dict(color='#FFBB00', width=1.5)), row=current_row, col=1)
    current_row += 1
    
# --- 5. Apply Quant Terminal Theme ---
fig.update_layout(
    template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
    margin=dict(l=10, r=10, t=10, b=10), height=800 if total_rows > 1 else 600, 
    xaxis_rangeslider_visible=False, showlegend=False,
    font=dict(family="Courier New, monospace", color="#E0E6ED")
)

# 🔥 ADDED: type='category' removes all weekend/overnight gaps natively
# 🔥 ADDED: nticks=10 prevents the x-axis from getting cluttered with labels
fig.update_xaxes(
    showgrid=True, gridwidth=1, gridcolor='#2B3040', zeroline=False,
    type='category',
    categoryorder='category ascending',
    nticks=10 
)

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2B3040', zeroline=False)

with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True)

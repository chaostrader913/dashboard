import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import fetch_data
from utils.indicators import apply_td_sequential, apply_rsi_divergence, get_pivot_points, apply_macd, apply_bollinger_bands

# --- 1. Terminal UI Styling ---
st.markdown("### 📡 MODULE: ADVANCED SIGNAL SCANNER")
st.divider()

# --- 2. Command Row (User Inputs) ---
col1, col2, col3, col4 = st.columns([1.5, 1, 2, 2])
with col1:
    ticker = st.text_input("TARGET ASSET", value="BTC-USD").upper()
with col2:
    timeframe = st.selectbox("TIMEFRAME", options=["1d", "1h", "15m"], index=0)
with col3:
    # Overlays map directly onto the candlestick chart
    overlays = st.multiselect(
        "PRICE OVERLAYS", 
        options=["TD Sequential", "Auto Trendlines", "Bollinger Bands"],
        default=["TD Sequential", "Auto Trendlines"]
    )
with col4:
    # Oscillators require their own subplot rows
    oscillators = st.multiselect(
        "OSCILLATORS", 
        options=["RSI Divergence", "MACD"],
        default=["RSI Divergence"]
    )

# --- 3. Data Processing & Indicator Application ---
with st.spinner(f"EXECUTING ALGORITHMS FOR {ticker}..."):
    data = fetch_data(ticker, interval=timeframe, period="1y" if timeframe == "1d" else "60d")

if data is None or data.empty:
    st.error(f"ERR: NO DATA FOUND FOR {ticker}.")
    st.stop()

# Apply selected mathematical models
if "TD Sequential" in overlays:
    data = apply_td_sequential(data)
if "Auto Trendlines" in overlays:
    data = get_pivot_points(data, window=5)
if "Bollinger Bands" in overlays:
    data = apply_bollinger_bands(data)
if "RSI Divergence" in oscillators:
    data = apply_rsi_divergence(data)
if "MACD" in oscillators:
    data = apply_macd(data)

# --- 4. Dynamic Charting Engine ---
# Calculate how many rows we need based on selected oscillators
total_rows = 1 + len(oscillators)
# Price chart gets 60% of height, oscillators share the remaining 40%
row_heights = [0.6] + [0.4 / len(oscillators)] * len(oscillators) if oscillators else [1.0]

fig = make_subplots(
    rows=total_rows, cols=1, shared_xaxes=True, 
    vertical_spacing=0.03, row_heights=row_heights
)

# [ROW 1] BASE PRICE CHART
fig.add_trace(go.Candlestick(
    x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
    name="Price", increasing_line_color='#00FFAA', decreasing_line_color='#FF4B4B'
), row=1, col=1)

# [ROW 1] OVERLAYS
if "Auto Trendlines" in overlays:
    # connectgaps=True draws a line straight through the NaN values between pivots!
    fig.add_trace(go.Scatter(x=data.index, y=data['Pivot_High'], mode='lines', name='Resist', line=dict(color='#FF4B4B', dash='dot', width=2), connectgaps=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Pivot_Low'], mode='lines', name='Support', line=dict(color='#00FFAA', dash='dot', width=2), connectgaps=True), row=1, col=1)

if "Bollinger Bands" in overlays:
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Up', line=dict(color='#4B4BFF', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Low', line=dict(color='#4B4BFF', width=1)), row=1, col=1)

if "TD Sequential" in overlays:
    # Filter only the completed '9' counts
    td_buy = data[data['Signal'] == 1]
    td_sell = data[data['Signal'] == -1]
    # Plot '9' below the low for buys, above the high for sells
    fig.add_trace(go.Scatter(x=td_buy.index, y=td_buy['Low'] * 0.98, mode='text', text='9', textfont=dict(color='#00FFAA', size=14, family='Courier New'), name='TD Buy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=td_sell.index, y=td_sell['High'] * 1.02, mode='text', text='9', textfont=dict(color='#FF4B4B', size=14, family='Courier New'), name='TD Sell'), row=1, col=1)

# [ROWS 2+] OSCILLATORS
current_row = 2

if "RSI Divergence" in oscillators:
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='#00AAFF', width=1.5)), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF4B4B", line_width=1, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00FFAA", line_width=1, row=current_row, col=1)
    # Plot divergence markers if they exist
    if 'Signal' in data.columns: # Assuming RSI signal overrides others for simplicity here, or check specifically
        rsi_bulls = data[data['Signal'] == 1]
        fig.add_trace(go.Scatter(x=rsi_bulls.index, y=rsi_bulls['RSI'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00FFAA'), name='RSI Div'), row=current_row, col=1)
    current_row += 1

if "MACD" in oscillators:
    # Determine histogram colors (Green if positive, Red if negative)
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
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2B3040', zeroline=False)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2B3040', zeroline=False)

with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- 1. Terminal UI Styling & Header ---
st.markdown("### 📡 MODULE: SIGNAL SCANNER // RSI DIVERGENCE")
st.caption("EXECUTING REAL-TIME MARKET SCAN AND OVERLAY DIAGNOSTICS")
st.divider()

# --- 2. User Input Parameters (Command Row) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    ticker = st.text_input("TARGET ASSET", value="BTC-USD").upper()
with col2:
    timeframe = st.selectbox("TIMEFRAME", options=["1d", "1h", "15m", "5m"], index=0)
with col3:
    rsi_period = st.number_input("RSI PERIOD", min_value=2, max_value=50, value=14)
with col4:
    st.write("RSI BANDS")
    # A slider to define the overbought/oversold thresholds
    rsi_bands = st.slider("Bands", min_value=10, max_value=90, value=(30, 70), label_visibility="collapsed")

oversold_threshold, overbought_threshold = rsi_bands

# --- 3. Data Engine & Math ---
@st.cache_data(ttl=300) # Cache data for 5 minutes to prevent spamming Yahoo Finance
def fetch_and_calculate(ticker_symbol, interval, rsi_len):
    # Fetch data
    df = yf.download(ticker_symbol, period="3mo" if interval == "1d" else "5d", interval=interval, progress=False)
    
    if df.empty:
        return None
        
    # Flatten multi-index columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate RSI (Standard Exponential Moving Average method)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    ema_gain = gain.ewm(com=rsi_len - 1, adjust=False).mean()
    ema_loss = loss.ewm(com=rsi_len - 1, adjust=False).mean()
    
    rs = ema_gain / ema_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Generate Signals
    # Buy Signal: Place marker slightly below the Low when RSI drops below Oversold
    df['Buy_Signal'] = df.apply(lambda row: row['Low'] * 0.98 if row['RSI'] < oversold_threshold else None, axis=1)
    # Sell Signal: Place marker slightly above the High when RSI pushes above Overbought
    df['Sell_Signal'] = df.apply(lambda row: row['High'] * 1.02 if row['RSI'] > overbought_threshold else None, axis=1)
    
    return df

with st.spinner(f"FETCHING TICKER DATA FOR {ticker}..."):
    data = fetch_and_calculate(ticker, timeframe, rsi_period)

# --- 4. Render Charting Engine ---
if data is None or data.empty:
    st.error(f"ERR: NO DATA FOUND FOR {ticker}. CHECK TICKER SYMBOL.")
else:
    # Create a subplot grid: 2 rows, shared X axis. Top is 70% height, Bottom is 30%
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Plot 1: Candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
        name="Price",
        increasing_line_color='#00FFAA', increasing_fillcolor='#00FFAA', # Terminal Green
        decreasing_line_color='#FF4B4B', decreasing_fillcolor='#FF4B4B'  # Terminal Red
    ), row=1, col=1)

    # Plot 2: Buy Signals (Green Triangles pointing UP)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Buy_Signal'],
        mode='markers', name='Buy Signal',
        marker=dict(symbol='triangle-up', size=10, color='#00FFAA'),
        hoverinfo='skip'
    ), row=1, col=1)

    # Plot 3: Sell Signals (Red Triangles pointing DOWN)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Sell_Signal'],
        mode='markers', name='Sell Signal',
        marker=dict(symbol='triangle-down', size=10, color='#FF4B4B'),
        hoverinfo='skip'
    ), row=1, col=1)

    # Plot 4: RSI Line
    fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI'],
        mode='lines', name='RSI',
        line=dict(color='#00AAFF', width=1.5) # Neon Blue
    ), row=2, col=1)

    # Plot 5: RSI Overbought/Oversold Lines
    fig.add_hline(y=overbought_threshold, line_dash="dash", line_color="#FF4B4B", line_width=1, row=2, col=1)
    fig.add_hline(y=oversold_threshold, line_dash="dash", line_color="#00FFAA", line_width=1, row=2, col=1)

    # --- 5. Apply the "Quant Terminal" Dark Theme ---
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117", # Matches your config.toml base
        plot_bgcolor="#0E1117",
        margin=dict(l=10, r=10, t=10, b=10),
        height=600,
        xaxis_rangeslider_visible=False, # Hides the clunky volume slider at the bottom
        showlegend=False,
        font=dict(family="Courier New, monospace", color="#E0E6ED")
    )
    
    # Faint gridlines for that technical look
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2B3040', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2B3040', zeroline=False)

    # Push to Streamlit
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

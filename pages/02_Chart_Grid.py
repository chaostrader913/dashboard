import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import fetch_data  # Using our centralized data loader!

# --- 1. Terminal UI Styling ---
st.markdown("### 📊 MODULE: QUAD CHART GRID")
st.divider()

# --- 2. Ticker Database ---
TICKER_GROUPS = {
    'Equities & Indices': {'^GSPC': 'S&P 500', 'QQQ': 'Nasdaq 100 ETF', '^VIX': 'Volatility Index', 'IWM': 'Russell 2000 ETF'},
    'Sectors (US)': {'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy', 'XLV': 'Healthcare'},
    'International': {'FXI': 'China (Large Cap)', 'EWJ': 'Japan', 'EWG': 'Germany', 'INDA': 'India'},
    'Fixed Income': {'TLT': '20Y+ Treasury', 'IEF': '7-10Y Treasury', 'HYG': 'High Yield', 'LQD': 'Inv. Grade Corp'},
    'Currencies & Crypto': {'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'EURUSD=X': 'EUR/USD', 'DX-Y.NYB': 'DXY'},
    'Commodities': {'GC=F': 'Gold', 'CL=F': 'Crude Oil', 'SI=F': 'Silver', 'HG=F': 'Copper'}
}

# Flatten the dictionary for easy searching in the dropdowns
ALL_TICKERS = {ticker: f"{ticker} - {name}" for group in TICKER_GROUPS.values() for ticker, name in group.items()}

# --- 3. Command Row (User Inputs) ---
st.sidebar.markdown("### ⚙️ GRID CONTROLS")
timeframe = st.sidebar.selectbox("TIMEFRAME", options=["1d", "1h", "15m", "5m"], index=0)

st.sidebar.markdown("#### SCREEN ALLOCATION")
# Default to a classic macro dashboard: SPY, Gold, Oil, Bitcoin
screen_1 = st.sidebar.selectbox("SCREEN 1 (TOP LEFT)", options=ALL_TICKERS.keys(), format_func=lambda x: ALL_TICKERS[x], index=list(ALL_TICKERS.keys()).index('^GSPC'))
screen_2 = st.sidebar.selectbox("SCREEN 2 (TOP RIGHT)", options=ALL_TICKERS.keys(), format_func=lambda x: ALL_TICKERS[x], index=list(ALL_TICKERS.keys()).index('GC=F'))
screen_3 = st.sidebar.selectbox("SCREEN 3 (BOT LEFT)", options=ALL_TICKERS.keys(), format_func=lambda x: ALL_TICKERS[x], index=list(ALL_TICKERS.keys()).index('CL=F'))
screen_4 = st.sidebar.selectbox("SCREEN 4 (BOT RIGHT)", options=ALL_TICKERS.keys(), format_func=lambda x: ALL_TICKERS[x], index=list(ALL_TICKERS.keys()).index('BTC-USD'))

overlays = st.sidebar.multiselect("OVERLAYS", options=["EMA 20", "EMA 50"], default=["EMA 20"])

# --- 4. Plotly Chart Generator ---
def create_terminal_chart(ticker_symbol, df, title_name):
    """Generates a standalone, dark-themed Plotly candlestick chart with volume"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.8, 0.2])
    
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price", increasing_line_color='#00FFAA', decreasing_line_color='#FF4B4B'
    ), row=1, col=1)
    
    # Overlays
    if "EMA 20" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=20).mean(), mode='lines', name='EMA 20', line=dict(color='#00AAFF', width=1)), row=1, col=1)
    if "EMA 50" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50).mean(), mode='lines', name='EMA 50', line=dict(color='#FFBB00', width=1)), row=1, col=1)

    # Volume
    colors = ['#00FFAA' if row['Close'] >= row['Open'] else '#FF4B4B' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # Styling
    fig.update_layout(
        title=dict(text=f"{title_name} ({ticker_symbol})", font=dict(color="#E0E6ED", size=14)),
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        margin=dict(l=10, r=10, t=40, b=10), height=350, xaxis_rangeslider_visible=False, showlegend=False,
        font=dict(family="Courier New, monospace", color="#E0E6ED")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2B3040', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2B3040', zeroline=False)
    
    return fig

# --- 5. Render the Quad Grid ---
screens = [screen_1, screen_2, screen_3, screen_4]

# Create a 2x2 layout
col1, col2 = st.columns(2)
grid_cols = [col1, col2, col1, col2]

for i, ticker in enumerate(screens):
    with grid_cols[i]:
        with st.container(border=True):
            with st.spinner(f"Loading {ticker}..."):
                # Use our centralized data loader
                data = fetch_data(ticker, interval=timeframe, period="1y" if timeframe == "1d" else "30d")
                
                if data is not None and not data.empty:
                    fig = create_terminal_chart(ticker, data, ALL_TICKERS[ticker].split(' - ')[1])
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_{ticker}")
                else:
                    st.error(f"ERR: DATA FEED OFFLINE FOR {ticker}")

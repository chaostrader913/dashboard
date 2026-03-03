import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
import io

# --- IMPORT GLOBALS ---
from utils.data_loader import fetch_data
from utils.indicators import apply_td_sequential, apply_rsi_divergence

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. UI Setup ---
st.set_page_config(layout="wide", page_title="Macro Market Grid")

st.markdown("### 🌐 MODULE: MACRO MARKET GRID")
st.caption("INSTITUTIONAL SNAPSHOT ENGINE // STABLE BUILD")

# --- 2. Ticker Database (Updated) ---
TICKER_GROUPS = {
    'Indices (US)': {
        '^GSPC': 'S&P 500', '^DJI': 'Dow Jones', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000',
        'MTUM': 'US Momentum', 'VLUE': 'US Value', 'QUAL': 'US Quality', 'USMV': 'US Min Vol'
    },
    'Sectors (US)': {
        'XLK': 'Technology', 'XLV': 'Healthcare', 'XLF': 'Financials', 'XLE': 'Energy', 
        'XLI': 'Industrials', 'XLY': 'Cons. Disc.', 'XLP': 'Cons. Staples', 
        'XLU': 'Utilities', 'XLB': 'Materials', 'XLRE': 'Real Estate', 'XLC': 'Comm. Svcs'
    },
    'Themes (US)': {
        'SMH': 'Semiconductors', 'IGV': 'Software', 'XBI': 'Biotech', 'ARKK': 'Innovation', 
        'TAN': 'Solar', 'URA': 'Uranium', 'LIT': 'Lithium', 'PAVE': 'Infrastructure'
    },
    'International': {
        'VEA': 'Dev ex-US', 'VWO': 'Emerging Mkts', 'EWJ': 'Japan', 
        'FXI': 'China Large', 'INDA': 'India', 'EWG': 'Germany', 'EWU': 'UK', 'EWZ': 'Brazil'
    },
    'Fixed Income ETFs': {
        'SHY': '1-3Y Treas', 'IEF': '7-10Y Treas', 'TLT': '20Y+ Treas',
        'LQD': 'Inv. Grade', 'HYG': 'High Yield', 'BND': 'Total Bond', 
        'MBB': 'MBS ETF', 'TIP': 'TIPS Bond'
    },
    'Commodity, Currencies & Crypto': {
        'GLD': 'Gold', 'SLV': 'Silver', 'USO': 'Crude Oil', 'UUP': 'US Dollar', 
        'FXE': 'Euro', 'FXY': 'Jap Yen', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum'
    },
    'Major Stocks by Market Cap': {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'META': 'Meta', 'BRK-B': 'Berkshire', 'TSLA': 'Tesla'
    }
}

# --- 3. Plotting Engine ---
def get_chart_image(ticker, data, style, show_tdsq, show_rsi):
    apds = []
    
    # TDSQ Overlays (Green 9s)
    if show_tdsq and 'Setup_Signal' in data.columns:
        b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.98, np.nan)
        s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.02, np.nan)
        if not np.isnan(b9).all(): 
            apds.append(mpf.make_addplot(b9, type='scatter', marker=r'$9$', color='#00FFAA', markersize=40))
        if not np.isnan(s9).all(): 
            apds.append(mpf.make_addplot(s9, type='scatter', marker=r'$9$', color='#00FFAA', markersize=40))
        
    # RSI Divergence Overlays (Blue Up Arrow)
    if show_rsi and 'Signal' in data.columns:
        rsi_b = np.where(data['Signal'] == 1, data['Low'] * 0.95, np.nan)
        if not np.isnan(rsi_b).all(): 
            apds.append(mpf.make_addplot(rsi_b, type='scatter', marker='^', color='#00AAFF', markersize=60))

    fig, axlist = mpf.plot(data, type='candle', style=style, returnfig=True, 
                           figsize=(5, 3.5), tight_layout=True, addplot=apds if apds else None)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

# --- 4. Sidebar ---
with st.sidebar:
    st.header("GRID CONTROLS")
    period = st.selectbox('LOOKBACK PERIOD', ['1mo', '3mo', '6mo', '1y'], index=1)
    style_sel = st.selectbox('CHART THEME', ['nightclouds', 'yahoo', 'mike', 'blueskies'], index=0)
    cols_count = st.slider("GRID COLUMNS", 2, 6, 4)
    
    st.divider()
    st.markdown("#### INDICATORS")
    td_on = st.checkbox("TDSQ (9 Count)", value=True)
    rsi_on = st.checkbox("RSI Bullish Div", value=True)
    
    st.caption("Note: Charts load sequentially to ensure data integrity.")

# --- 5. Execution Loop ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        cols = st.columns(cols_count)
        for i, (ticker, name) in enumerate(tickers.items()):
            with cols[i % cols_count]:
                # Sequential fetch handles diverse ticker types (Index vs ETF vs Stock) safely
                data = fetch_data(ticker, "1d", period)
                
                if data is not None and not data.empty:
                    # Apply logic
                    data = apply_td_sequential(data)
                    data = apply_rsi_divergence(data)
                    
                    # Header Info
                    last_p = data['Close'].iloc[-1]
                    prev_p = data['Close'].iloc[-2] if len(data) > 1 else last_p
                    chg = ((last_p - prev_p) / prev_p) * 100
                    color = "green" if chg >= 0 else "red"
                    
                    st.markdown(f"**{name}**")
                    st.markdown(f"`{ticker}`: **${last_p:,.2f}** (:{color}[{chg:+.2f}%])")
                    
                    # Render Image
                    img = get_chart_image(ticker, data, style_sel, td_on, rsi_on)
                    st.image(img, use_container_width=True)
                else:
                    st.error(f"Data Error: {ticker}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
import io
import concurrent.futures

# --- IMPORT GLOBALS ---
from utils.data_loader import fetch_data
from utils.indicators import apply_td_sequential, apply_rsi_divergence

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. UI Setup ---
st.set_page_config(layout="wide")
st.markdown("### 🌐 MODULE: MACRO MARKET GRID")

# --- 2. Ticker Database ---
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
    # Setup additional plots (Signals)
    apds = []
    
    if show_tdsq and 'Setup_Signal' in data.columns:
        # Green 9s, Red 13s
        b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.98, np.nan)
        s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.02, np.nan)
        if not np.isnan(b9).all(): apds.append(mpf.make_addplot(b9, type='scatter', marker=r'$9$', color='#00FFAA', markersize=40))
        if not np.isnan(s9).all(): apds.append(mpf.make_addplot(s9, type='scatter', marker=r'$9$', color='#00FFAA', markersize=40))
        
    if show_rsi and 'Signal' in data.columns:
        rsi_b = np.where(data['Signal'] == 1, data['Low'] * 0.95, np.nan)
        if not np.isnan(rsi_b).all(): apds.append(mpf.make_addplot(rsi_b, type='scatter', marker='^', color='#00AAFF', markersize=60))

    fig, axlist = mpf.plot(data, type='candle', style=style, returnfig=True, 
                           figsize=(5, 3.5), tight_layout=True, addplot=apds if apds else None)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

# --- 4. Sidebar ---
with st.sidebar:
    period = st.selectbox('PERIOD', ['3mo', '6mo', '1y'], index=1)
    style_sel = st.selectbox('THEME', ['nightclouds', 'yahoo', 'mike'], index=0)
    cols_count = st.slider("GRID COLUMNS", 2, 5, 4)
    st.divider()
    td_on = st.checkbox("TDSQ Signals", value=True)
    rsi_on = st.checkbox("RSI Divergence", value=True)

# --- 5. Execution ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))
for tab, (group, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        cols = st.columns(cols_count)
        for i, (ticker, name) in enumerate(tickers.items()):
            with cols[i % cols_count]:
                data = fetch_data(ticker, "1d", period)
                if data is not None and not data.empty:
                    # Apply Logic
                    data = apply_td_sequential(data)
                    data = apply_rsi_divergence(data)
                    
                    # Header
                    last_p = data['Close'].iloc[-1]
                    st.markdown(f"**{name}** `${last_p:.2f}`")
                    
                    # Render
                    img = get_chart_image(ticker, data, style_sel, td_on, rsi_on)
                    st.image(img, use_container_width=True)


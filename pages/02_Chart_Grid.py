import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings

# --- IMPORT THE GLOBAL UTILITY ---
from utils.data_loader import fetch_data

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MOVING Averages IGNORED.*")

# --- 1. Terminal UI Styling ---
st.markdown("### 🌐 MODULE: MACRO MARKET GRID")
st.caption("STATIC SNAPSHOT ENGINE // BIRD'S EYE VIEW")
st.divider()

# --- 2. Ticker Database ---
TICKER_GROUPS = {
    'Equities & Indices': {
        '^GSPC': 'S&P 500', 'QQQ': 'Nasdaq 100 ETF', '^DJI': 'Dow Jones', 
        'IWM': 'Russell 2000 ETF', '^VIX': 'Volatility Index'
    },
    'Sectors (US)': {
        'XLK': 'Technology', 'XLV': 'Healthcare', 'XLF': 'Financials',
        'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Disc.'
    },
    'International': {
        'EWJ': 'Japan', 'FXI': 'China (Large Cap)', 'EWG': 'Germany', 
        'EWU': 'UK', 'INDA': 'India', 'EEM': 'Emerging Markets'
    },
    'Fixed Income': {
        'SHY': '1-3Y Treasury', 'IEF': '7-10Y Treasury', 'TLT': '20Y+ Treasury',
        'LQD': 'Inv. Grade Corp', 'HYG': 'High Yield'
    },
    'Currencies & Crypto': {
        'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'EURUSD=X': 'EUR/USD', 
        'JPY=X': 'USD/JPY', 'DX-Y.NYB': 'US Dollar Index'
    },
    'Commodities': {
        'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil',
        'NG=F': 'Natural Gas', 'HG=F': 'Copper'
    }
}

# --- 3. Plotting Engine ---
def plot_single_asset(ticker, name, data, chart_type, style, show_sma, show_vol):
    tech_types = {'OHLC': 'ohlc', 'Candlestick': 'candle', 'Renko': 'renko', 'Point and Figure': 'pnf'}
    
    if chart_type in tech_types:
        mpf_type = tech_types[chart_type]
        current_style = style
        
        # Style conflict resolution for P&F / Renko
        if mpf_type in ['renko', 'pnf'] and style == 'mike':
            current_style = 'yahoo'

        kwargs = dict(
            type=mpf_type, 
            style=current_style, 
            show_nontrading=False, 
            returnfig=True,
            title=f"{name} ({ticker})",
            figsize=(5, 3.5) # Compact for grid viewing
        )
        
        if show_sma and mpf_type not in ['renko', 'pnf']: 
            kwargs['mav'] = (20,)
        if show_vol and 'Volume' in data.columns: 
            kwargs['volume'] = True
        if mpf_type == 'renko': 
            kwargs['renko_params'] = {'brick_size': 'atr'}
        elif mpf_type == 'pnf': 
            kwargs['pnf_params'] = {'box_size': 'atr'}

        # Generate plot
        fig, axlist = mpf.plot(data, **kwargs)
        
        # Force dark background to match terminal if nightclouds is selected
        if style == 'nightclouds':
            fig.patch.set_facecolor('#0E1117')
            
        return fig
    else:
        # Fallback Line Chart
        fig, ax = plt.subplots(figsize=(5, 3.5))
        prices = data['Close']
        ax.plot(prices.index, prices, linewidth=1.5, color='#00FFAA' if style=='nightclouds' else 'blue')
        if show_sma: 
            ax.plot(prices.index, prices.rolling(20).mean(), linestyle='--', color='gray', alpha=0.7)
            
        ax.set_title(f"{name} ({ticker})", fontsize=10, color='white' if style=='nightclouds' else 'black')
        if style == 'nightclouds':
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#2B3040')
                
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        return fig

# --- 4. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ GRID CONTROLS")
    period_sel = st.selectbox('PERIOD', ['1mo', '3mo', '6mo', '1y', '2y'], index=1)
    interval_sel = st.selectbox('INTERVAL', ['1d', '1h', '15m', 'Custom Days'], index=0)
    
    is_custom = (interval_sel == 'Custom Days')
    day_slider = st.slider('CUSTOM BARS (DAYS)', min_value=2, max_value=10, value=3, disabled=not is_custom)
    
    st.divider()
    chart_sel = st.selectbox('CHART TYPE', ['Candlestick', 'OHLC', 'Line', 'Renko', 'Point and Figure'], index=0)
    style_sel = st.selectbox('THEME', ['nightclouds', 'yahoo', 'blueskies', 'mike'], index=0) 
    
    st.divider()
    sma_check = st.checkbox('OVERLAY: 20 SMA', value=True)
    vol_check = st.checkbox('SHOW VOLUME', value=True)

# --- 5. Main App Execution (Tabs & Grid) ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        st.write("") 
        cols = st.columns(3)
        for i, (ticker, name) in enumerate(tickers.items()):
            with cols[i % 3]:
                with st.container(border=True):
                    with st.spinner(f"Loading {ticker}..."):
                        
                        # Fetch using our centralized utility
                        data = fetch_data(
                            ticker=ticker, 
                            interval=interval_sel, 
                            period=period_sel, 
                            custom_days=day_slider
                        )
                        
                        if data is not None and not data.empty:
                            fig = plot_single_asset(ticker, name, data, chart_sel, style_sel, sma_check, vol_check)
                            st.pyplot(fig)
                            plt.close(fig) # Critical to prevent RAM overflow
                        else:
                            st.error(f"ERR: NO DATA FOR {ticker}")

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

# --- 1. Terminal UI & CSS Overrides ---
# This CSS strips out all of Streamlit's default margins and column gaps
# st.markdown("""
# <style>
#     /* Remove padding from the main block */
#     .block-container {
#         padding-top: 1rem;
#         padding-bottom: 0rem;
#         padding-left: 1rem;
#         padding-right: 1rem;
#         max-width: 100%;
#     }
#     /* Force columns to have zero gap and zero padding */
#     div[data-testid="column"] {
#         padding: 2px !important; /* Tiny 2px gap to separate chart borders */
#     }
#     div[data-testid="stHorizontalBlock"] {
#         gap: 0rem !important;
#     }
#     /* Remove tab spacing */
#     div[data-testid="stTabs"] {
#         gap: 0rem !important;
#     }
# </style>
# """, unsafe_allow_html=True)

st.markdown("### 🌐 MODULE: MACRO MARKET GRID")
st.caption("STATIC SNAPSHOT ENGINE // BIRD'S EYE VIEW")

# --- 2. Upgraded Institutional Ticker Database ---
TICKER_GROUPS = {
    'Indices (US)': {
        '^GSPC': 'S&P 500', '^DJI': 'Dow Jones', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000',
        'MTUM': 'US Momentum Factor', 'VLUE': 'US Value Factor', 'QUAL': 'US Quality Factor', 'USMV': 'US Min Volatility'
    },
    'Sectors (US)': {
        'XLK': 'Technology', 'XLV': 'Healthcare', 'XLF': 'Financials', 'XLE': 'Energy', 
        'XLI': 'Industrials', 'XLY': 'Consumer Disc.', 'XLP': 'Consumer Staples', 
        'XLU': 'Utilities', 'XLB': 'Materials', 'XLRE': 'Real Estate', 'XLC': 'Comm. Services'
    },
    'Themes (US)': {
        'SMH': 'Semiconductors', 'IGV': 'Software', 'XBI': 'Biotech', 'ARKK': 'Innovation', 
        'TAN': 'Solar Energy', 'URA': 'Uranium', 'LIT': 'Lithium & Battery', 'PAVE': 'Infrastructure'
    },
    'International': {
        'VEA': 'Developed Markets ex-US', 'VWO': 'Emerging Markets', 'EWJ': 'Japan', 
        'FXI': 'China Large-Cap', 'INDA': 'India', 'EWG': 'Germany', 'EWU': 'UK', 'EWZ': 'Brazil'
    },
    'Fixed Income ETFs': {
        'SHY': '1-3Y Treasury', 'IEF': '7-10Y Treasury', 'TLT': '20Y+ Treasury',
        'LQD': 'Inv. Grade Corp', 'HYG': 'High Yield Corp', 'BND': 'Total Bond Market', 
        'MBB': 'MBS ETF', 'TIP': 'TIPS Bond ETF'
    },
    'Commodity, Currencies & Crypto': {
        'GLD': 'Gold', 'SLV': 'Silver', 'USO': 'Crude Oil', 'UUP': 'US Dollar', 
        'FXE': 'Euro', 'FXY': 'Japanese Yen', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum'
    },
    'Major Stocks by Market Cap': {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'META': 'Meta', 'BRK-B': 'Berkshire Hathaway', 'TSLA': 'Tesla'
    }
}

# --- 3. Plotting Engine ---
def plot_single_asset(ticker, name, data, chart_type, style, show_sma, show_vol):
    tech_types = {'OHLC': 'ohlc', 'Candlestick': 'candle', 'Renko': 'renko', 'Point and Figure': 'pnf'}
    
    if chart_type in tech_types:
        mpf_type = tech_types[chart_type]
        current_style = style
        
        if mpf_type in ['renko', 'pnf'] and style == 'mike':
            current_style = 'yahoo'

        kwargs = dict(
            type=mpf_type, 
            style=current_style, 
            show_nontrading=False, 
            returnfig=True,
            title=f"{name} ({ticker})",
            figsize=(5, 3.2), # Slightly squashed height to fit more rows
            tight_layout=True # Strips out Matplotlib padding
        )
        
        if show_sma and mpf_type not in ['renko', 'pnf']: 
            kwargs['mav'] = (20,)
        if show_vol and 'Volume' in data.columns: 
            kwargs['volume'] = True
        if mpf_type == 'renko': 
            kwargs['renko_params'] = {'brick_size': 'atr'}
        elif mpf_type == 'pnf': 
            kwargs['pnf_params'] = {'box_size': 'atr'}

        fig, axlist = mpf.plot(data, **kwargs)
        
        if style == 'nightclouds':
            fig.patch.set_facecolor('#0E1117')
            
        return fig
    else:
        # Fallback Line Chart
        fig, ax = plt.subplots(figsize=(5, 3.2))
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
                
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Aggressive whitespace removal
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5, rect=[0, 0.03, 1, 0.95])
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
    
    # Let the user choose how dense the grid is
    cols_count = st.slider("GRID COLUMNS", min_value=2, max_value=6, value=4)

# --- 5. Main App Execution (Tabs & Grid) ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        # We removed the bordered container here so the charts sit flush against each other
        cols = st.columns(cols_count)
        for i, (ticker, name) in enumerate(tickers.items()):
            with cols[i % cols_count]:
                with st.spinner(f"Loading {ticker}..."):
                    data = fetch_data(
                        ticker=ticker, 
                        interval=interval_sel, 
                        period=period_sel, 
                        custom_days=day_slider
                    )
                    
                    if data is not None and not data.empty:
                        fig = plot_single_asset(ticker, name, data, chart_sel, style_sel, sma_check, vol_check)
                        st.pyplot(fig)
                        plt.close(fig) 
                    else:
                        st.error(f"ERR: {ticker}")



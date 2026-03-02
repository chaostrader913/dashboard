import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MOVING Averages IGNORED.*")

# --- Page Config ---
st.set_page_config(page_title="Market Intelligence Dashboard", layout="wide")
st.title("Market Intelligence Dashboard")

# --- Expanded Ticker Groups ---
TICKER_GROUPS = {
    'Equities & Indices': {
        '^GSPC': 'S&P 500', 'SPY': 'S&P 500 ETF', 'VOO': 'Vanguard S&P 500',

        '^IXIC': 'NASDAQ Composite', 'QQQ': 'Nasdaq 100 ETF',

        '^DJI': 'Dow Jones', 'DIA': 'Dow Jones ETF',

        '^RUT': 'Russell 2000', 'IWM': 'Russell 2000 ETF',

        '^VIX': 'Volatility Index', 'VIXY': 'VIX Short-Term Futures'
    },
    'Sectors (US)': {
        'XLK': 'Technology', 'XLV': 'Healthcare', 'XLF': 'Financials',
        'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Disc.',
        'XLC': 'Comm. Services', 'XLP': 'Consumer Staples', 'XLU': 'Utilities',
        'XLB': 'Materials', 'XLRE': 'Real Estate', 'GLD': 'GOLD ETF'
    },
    'International': {
        'EWJ': 'Japan', 'FXI': 'China (Large Cap)', 'MCHI': 'China (MSCI)',
        'EWG': 'Germany', 'EWU': 'UK', 'INDA': 'India',
        'EWW': 'Mexico', 'EWZ': 'Brazil', 'EFA': 'MSCI EAFE (Dev. ex-US)',
        'IEMG': 'MSCI Em. Markets', 'VGK': 'Europe', 'EWC': 'Canada', 'EWA': 'Australia',
        'KSA': 'Saudi Arabia'
    },
    'Fixed Income': {
        'SHY': '1-3Y Treasury', 'IEF': '7-10Y Treasury', 'TLT': '20Y+ Treasury',
        'LQD': 'Inv. Grade Corp', 'HYG': 'High Yield', 'BND': 'Total Bond Market'
    },
    'Currencies & Crypto': {
        'EURUSD=X': 'EUR/USD', 'JPY=X': 'USD/JPY', 'GBPUSD=X': 'GBP/USD',
        'DX-Y.NYB': 'US Dollar Index', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum'
    },
    'Commodities': {
        'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil',
        'NG=F': 'Natural Gas', 'HG=F': 'Copper', 'ZC=F': 'Corn'
    }
}

# --- Core Logic ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_resample(ticker, period, interval, custom_days):
    try:
        fetch_i = '1d' if interval == 'Custom Days' else interval
        data = yf.download(ticker, period=period, interval=fetch_i, progress=False, auto_adjust=True)
        
        if data is None or data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.index = pd.to_datetime(data.index)

        if interval == 'Custom Days':
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            data = data.resample(f'{custom_days}B').apply(logic).dropna()

        data.index.name = 'Date'
        return data.dropna()
    except Exception:
        return None

def plot_single_asset(ticker, name, data, gold_df, chart_type, style, rel_gold, show_sma, show_vol):
    tech_types = {'OHLC': 'ohlc', 'Candlestick (Raw)': 'candle', 'Renko': 'renko', 'Point and Figure': 'pnf'}
    
    if chart_type in tech_types:
        mpf_type = tech_types[chart_type]
        current_style = style
        
        # Style conflict resolution
        if mpf_type in ['renko', 'pnf'] and style == 'mike':
            current_style = 'yahoo'

        kwargs = dict(
            type=mpf_type, 
            style=current_style, 
            show_nontrading=False, 
            returnfig=True,
            title=f"{name} ({ticker})"
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
        return fig
    else:
        # Line chart logic
        fig, ax = plt.subplots(figsize=(6, 4))
        prices = (data['Close'] / data['Close'].iloc[0]) * 100
        
        if rel_gold and gold_df is not None:
            g_prices = (gold_df['Close'] / gold_df['Close'].iloc[0]) * 100
            prices = (prices / g_prices.reindex(prices.index).ffill()).bfill() * 100

        ax.plot(prices.index, prices, linewidth=1.5, label='Price')
        if show_sma: 
            ax.plot(prices.index, prices.rolling(20).mean(), linestyle='--', label='20 SMA')
            
        ax.set_title(f"{name} ({ticker})", fontsize=11)
        ax.axhline(100, color='gray', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        if show_sma: ax.legend()
        plt.tight_layout()
        return fig

# --- UI Setup (Sidebar) ---
with st.sidebar:
    st.header("Controls")
    period_sel = st.selectbox('Period:', ['1mo', '6mo', '1y', '5y', 'max'], index=2)
    interval_sel = st.selectbox('Interval:', ['1h', '1d', 'Custom Days', '1wk', '1mo'], index=1)
    
    # Disable day slider based on interval
    is_custom = (interval_sel == 'Custom Days')
    day_slider = st.slider('Custom Days:', min_value=2, max_value=30, value=3, disabled=not is_custom)
    
    st.divider()
    chart_sel = st.selectbox('Chart:', ['Line', 'OHLC', 'Candlestick (Raw)', 'Renko', 'Point and Figure'], index=2)
    style_sel = st.selectbox('Style:', ['yahoo', 'nightclouds', 'blueskies', 'mike'], index=0)
    
    st.divider()
    sma_check = st.checkbox('SMA (20)', value=True)
    vol_check = st.checkbox('Volume', value=True)
    rel_gold = st.checkbox('Rel. Gold', value=False)
    
    st.divider()
    custom_ticker_input = st.text_input('Custom Search:', placeholder='AAPL, TSLA, NVDA...')

# Fetch Gold baseline if needed
gold_df = fetch_and_resample('GC=F', period_sel, interval_sel, day_slider) if rel_gold else None

# --- Main App Execution ---

# 1. Custom Search Rendering
if custom_ticker_input:
    st.subheader("Custom Search Results")
    search_list = [t.strip().upper() for t in custom_ticker_input.split(',') if t.strip()]
    
    cols = st.columns(3)
    for i, ticker in enumerate(search_list):
        with st.spinner(f"Loading {ticker}..."):
            data = fetch_and_resample(ticker, period_sel, interval_sel, day_slider)
            
            if data is not None and not data.empty:
                fig = plot_single_asset(ticker, ticker, data, gold_df, chart_sel, style_sel, rel_gold, sma_check, vol_check)
                with cols[i % 3]:
                    st.pyplot(fig)
                    plt.close(fig) # Prevent memory leaks
            else:
                with cols[i % 3]:
                    st.warning(f"No data for {ticker}")
    st.divider()

# 2. Tabbed Rendering
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        cols = st.columns(3)
        for i, (ticker, name) in enumerate(tickers.items()):
            data = fetch_and_resample(ticker, period_sel, interval_sel, day_slider)
            
            if data is not None and not data.empty:
                fig = plot_single_asset(ticker, name, data, gold_df, chart_sel, style_sel, rel_gold, sma_check, vol_check)
                with cols[i % 3]:
                    st.pyplot(fig)
                    plt.close(fig) # Prevent memory leaks
            else:
                 with cols[i % 3]:

                    st.warning(f"No data for {ticker}")


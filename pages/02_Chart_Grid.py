import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
import matplotlib

# Force a non-interactive backend for server stability
matplotlib.use('Agg')

# --- IMPORT GLOBALS ---
from utils.data_loader import fetch_data
from utils.indicators import apply_td_sequential, apply_rsi_divergence, apply_jma

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MOVING Averages IGNORED.*")
warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")

st.markdown("### 🌐 MODULE: MACRO MARKET GRID")
st.caption("STATIC SNAPSHOT ENGINE // BIRD'S EYE VIEW")

# --- 2. Upgraded Institutional Ticker Database ---
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
        'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil', 'HG=F': 'Copper', 
        'EURUSD=X': 'Euro','GBPUSD=X': 'British Pound', 'JPY=X': 'Jap Yen', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum'
    },
    'Major Stocks by Market Cap': {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'META': 'Meta', 'BRK-B': 'Berkshire', 'TSLA': 'Tesla',
        '0700.HK': 'Tencent', '9988.HK': 'Alibaba', '0981.HK': 'SMIC', '1810.HK': 'Xiaomi',
        '0992.HK': 'Lenovo', '1347.HK': 'Hua Hong Semiconductor', '6869.HK': 'Yangtze Optical', '0005.HK': 'HSBC',
        '3690.HK': 'Meituan', '1299.HK': 'AIA', '1888.HK': 'Kingboard Laminates', '0883.HK': 'CNOOC',
        '0148.HK': 'Kingboard Holdings', '2899.HK': 'Zijin Mining', '1211.HK': 'BYD', '2513.HK': 'Knowledge Atlas',
        '3750.HK': 'CATL', '9992.HK': 'Pop Mart', '1024.HK': 'Kuaishou', '0388.HK': 'HKEX',
        '0939.HK': 'CCB', '0941.HK': 'China Mobile', '9999.HK': 'NetEase', '9926.HK': 'Akeso',
        '0175.HK': 'Geely', '9903.HK': 'Iluvatar CoreX', '1398.HK': 'ICBC', '1378.HK': 'China Hongqiao',
        '2382.HK': 'Sunny Optical', '6651.HK': '51WORLD'
    }
}

# --- 3. Plotting Engine ---
def plot_single_asset(ticker, name, data, chart_type, style, show_sma, show_vol, show_tdsq, show_rsi):
    tech_types = {'OHLC': 'ohlc', 'Candlestick': 'candle', 'Renko': 'renko', 'Point and Figure': 'pnf'}
    
    if chart_type in tech_types:
        mpf_type = tech_types[chart_type]
        current_style = style if not (mpf_type in ['renko', 'pnf'] and style == 'mike') else 'yahoo'

        kwargs = dict(
            type=mpf_type, style=current_style, show_nontrading=False, returnfig=True,
            title=f"{name} ({ticker})", figsize=(5, 3.2)
        )
        
        if show_sma and mpf_type not in ['renko', 'pnf']: kwargs['mav'] = (20,50)
        if show_vol and 'Volume' in data.columns: kwargs['volume'] = True
        if mpf_type == 'renko': kwargs['renko_params'] = {'brick_size': 'atr'}
        elif mpf_type == 'pnf': kwargs['pnf_params'] = {'box_size': 'atr'}

        # --- SIGNAL OVERLAYS ---
        apds = []
        if mpf_type not in ['renko', 'pnf']:
            # 1. TDSQ Signals (Using standard markers to avoid font crashes)
            if show_tdsq and 'Setup_Signal' in data.columns:
                b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.98, np.nan)
                s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.02, np.nan)
                b13 = np.where(data['Countdown_Signal'] == 1, data['Low'] * 0.99, np.nan)
                s13 = np.where(data['Countdown_Signal'] == -1, data['High'] * 1.01, np.nan)
                
                # Setup (9) = Circles
                if not np.isnan(b9).all(): apds.append(mpf.make_addplot(b9, type='scatter', marker='$9$', color='green', markersize=50))
                if not np.isnan(s9).all(): apds.append(mpf.make_addplot(s9, type='scatter', marker='$9$', color='green', markersize=50))
                
                # Countdown (13) = Stars
                if not np.isnan(b13).all(): apds.append(mpf.make_addplot(b13, type='scatter', marker='$13$', color='red', markersize=70))
                if not np.isnan(s13).all(): apds.append(mpf.make_addplot(s13, type='scatter', marker='$13$', color='red', markersize=70))

            # 2. RSI Divergence Signals
            if show_rsi and 'Signal' in data.columns:
                rsi_b = np.where(data['Signal'] == 1, data['Low'] * 0.99, np.nan)
                if not np.isnan(rsi_b).all(): apds.append(mpf.make_addplot(rsi_b, type='scatter', marker='^', color='yellow', markersize=80))

            if apds: kwargs['addplot'] = apds

        fig, axlist = mpf.plot(data, **kwargs)
        
        if style == 'nightclouds': fig.patch.set_facecolor('#0E1117')
        
        # Manually adjust subplots to leave room for the title without using tight_layout
        fig.subplots_adjust(top=0.82, bottom=0.15, left=0.1, right=0.9, hspace=0, wspace=0)
        return fig
        
    else:
        # Fallback Line Chart
        fig, ax = plt.subplots(figsize=(5, 3.2))
        prices = data['Close']
        ax.plot(prices.index, prices, linewidth=1.5, color='#00FFAA' if style=='nightclouds' else 'blue')
        if show_sma: ax.plot(prices.index, prices.rolling(20).mean(), linestyle='--', color='gray', alpha=0.7)
            
        ax.set_title(f"{name} ({ticker})", fontsize=10, color='white' if style=='nightclouds' else 'black', pad=12)
        
        if style == 'nightclouds':
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_edgecolor('#2B3040')
                
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        fig.subplots_adjust(top=0.82, bottom=0.2, left=0.15, right=0.9)
        return fig

# --- 4. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ GRID CONTROLS")
    period_sel = st.selectbox('PERIOD', ['1mo', '3mo', '6mo', '1y', '2y'], index=2)
    interval_sel = st.selectbox('INTERVAL', ['1d', '1h', '15m', 'Custom Days'], index=0)
    
    is_custom = (interval_sel == 'Custom Days')
    day_slider = st.slider('CUSTOM BARS (DAYS)', min_value=2, max_value=10, value=3, disabled=not is_custom)
    
    st.divider()
    chart_sel = st.selectbox('CHART TYPE', ['Candlestick', 'OHLC', 'Line', 'Renko', 'Point and Figure'], index=0)
    style_sel = st.selectbox('THEME', ['nightclouds', 'yahoo', 'blueskies', 'mike'], index=0) 
    
    st.divider()
    st.markdown("#### OVERLAYS")
    sma_check = st.checkbox('SMA', value=True)
    vol_check = st.checkbox('VOLUME', value=True)
    
    tdsq_check = st.checkbox('TDSQ (Circles/Stars)', value=True)
    rsi_check = st.checkbox('RSI DIVERGENCE', value=True)
    
    st.divider()
    cols_count = st.slider("GRID COLUMNS", min_value=2, max_value=6, value=4)

# --- 5. Main App Execution (Tabs & Grid) ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        cols = st.columns(cols_count)
        for i, (ticker, name) in enumerate(tickers.items()):
            with cols[i % cols_count]:
                with st.spinner(f"Loading {ticker}..."):
                    data = fetch_data(ticker=ticker, interval=interval_sel, period=period_sel, custom_days=day_slider)
                    
                    if data is not None and not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                
                        data = data.loc[~data.index.duplicated(keep='first')]
                        if tdsq_check:
                            data = apply_td_sequential(data)
                        if rsi_check:
                            data = apply_rsi_divergence(data)
                            
                        fig = plot_single_asset(ticker, name, data, chart_sel, style_sel, sma_check, vol_check, tdsq_check, rsi_check)
                        
                        # Updated plotting call for 2026 Streamlit compliance
                        st.pyplot(fig, width='stretch')
                        plt.close(fig) 
                    else:
                        st.error(f"ERR: {ticker}")

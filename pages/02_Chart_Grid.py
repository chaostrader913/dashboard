import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
import io
import concurrent.futures
import plotly.graph_objects as go

# --- IMPORT GLOBALS ---
from utils.data_loader import fetch_data
from utils.indicators import apply_td_sequential, apply_rsi_divergence

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MOVING Averages IGNORED.*")

# --- 1. Terminal UI & CSS Overrides ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 1rem; padding-right: 1rem; max-width: 100%; }
    div[data-testid="column"] { padding: 4px !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0rem !important; }
    div[data-testid="stTabs"] { gap: 0rem !important; }
    /* Make the deep dive button sleek */
    .stButton>button { height: 30px; min-height: 30px; padding: 0px; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("### 🌐 MODULE: MACRO MARKET GRID")
st.caption("MULTI-THREADED SNAPSHOT ENGINE // INTERACTIVE DEEP DIVE")

# --- 2. Institutional Ticker Database ---
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

# --- 3. Interactive Modal (Deep Dive) ---
@st.dialog("🔎 INTERACTIVE TERMINAL", width="large")
def show_deep_dive(ticker, name, data):
    st.markdown(f"#### {name} ({ticker})")
    
    # Generate an interactive Plotly chart on the fly
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
        increasing_line_color='#00FFAA', decreasing_line_color='#FF4B4B'
    )])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=500, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Courier New, monospace", color="#E0E6ED")
    )
    fig.update_xaxes(showgrid=True, gridcolor='#2B3040')
    fig.update_yaxes(showgrid=True, gridcolor='#2B3040')
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Use your mouse to zoom, pan, and hover for precise data points.")

# --- 4. Render Caching Engine ---
@st.cache_data(show_spinner=False, ttl=900)
def get_chart_image_bytes(ticker, data, chart_type, style, show_sma, show_vol, show_tdsq, show_rsi):
    """
    Renders the Matplotlib chart in memory and returns PNG bytes.
    This prevents memory leaks and makes tab-switching instantaneous.
    """
    tech_types = {'OHLC': 'ohlc', 'Candlestick': 'candle', 'Renko': 'renko', 'Point and Figure': 'pnf'}
    
    if chart_type in tech_types:
        mpf_type = tech_types[chart_type]
        current_style = style if not (mpf_type in ['renko', 'pnf'] and style == 'mike') else 'yahoo'
        kwargs = dict(type=mpf_type, style=current_style, show_nontrading=False, returnfig=True, figsize=(5, 3.2))
        
        if show_sma and mpf_type not in ['renko', 'pnf']: kwargs['mav'] = (20,)
        if show_vol and 'Volume' in data.columns: kwargs['volume'] = True
        if mpf_type == 'renko': kwargs['renko_params'] = {'brick_size': 'atr'}
        elif mpf_type == 'pnf': kwargs['pnf_params'] = {'box_size': 'atr'}

        apds = []
        if mpf_type not in ['renko', 'pnf']:
            if show_tdsq and 'Setup_Signal' in data.columns:
                b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.98, np.nan)
                s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.02, np.nan)
                b13 = np.where(data['Countdown_Signal'] == 1, data['Low'] * 0.96, np.nan)
                s13 = np.where(data['Countdown_Signal'] == -1, data['High'] * 1.04, np.nan)
                
                if not np.isnan(b9).all(): apds.append(mpf.make_addplot(b9, type='scatter', marker=r'$9$', color='#00FFAA', markersize=40))
                if not np.isnan(s9).all(): apds.append(mpf.make_addplot(s9, type='scatter', marker=r'$9$', color='#00FFAA', markersize=40))
                if not np.isnan(b13).all(): apds.append(mpf.make_addplot(b13, type='scatter', marker=r'$13$', color='#FF4B4B', markersize=60))
                if not np.isnan(s13).all(): apds.append(mpf.make_addplot(s13, type='scatter', marker=r'$13$', color='#FF4B4B', markersize=60))

            if show_rsi and 'Signal' in data.columns:
                rsi_b = np.where(data['Signal'] == 1, data['Low'] * 0.95, np.nan)
                if not np.isnan(rsi_b).all(): apds.append(mpf.make_addplot(rsi_b, type='scatter', marker='^', color='#00AAFF', markersize=80))

            if apds: kwargs['addplot'] = apds

        fig, axlist = mpf.plot(data, **kwargs)
        if style == 'nightclouds': fig.patch.set_facecolor('#0E1117')
        fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5, rect=[0, 0.03, 1, 1])

    else:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        prices = data['Close']
        ax.plot(prices.index, prices, linewidth=1.5, color='#00FFAA' if style=='nightclouds' else 'blue')
        if show_sma: ax.plot(prices.index, prices.rolling(20).mean(), linestyle='--', color='gray', alpha=0.7)
            
        if style == 'nightclouds':
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_edgecolor('#2B3040')
                
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5, rect=[0, 0.03, 1, 1])
        
    # Save the figure to an in-memory buffer to cache it
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# --- 5. Sidebar Controls ---
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
    st.markdown("#### OVERLAYS")
    sma_check = st.checkbox('20 SMA', value=True)
    vol_check = st.checkbox('VOLUME', value=True)
    tdsq_check = st.checkbox('TDSQ (9 & 13)', value=True)
    rsi_check = st.checkbox('RSI DIVERGENCE', value=True)
    
    st.divider()
    cols_count = st.slider("GRID COLUMNS", min_value=2, max_value=6, value=4)

# --- 6. Main App Execution (Tabs & Grid) ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        # Multi-Threaded Data Fetching: Grabs all tab tickers simultaneously
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(fetch_data, t, interval_sel, period_sel, day_slider): t 
                for t in tickers.keys()
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                t = future_to_ticker[future]
                try:
                    results[t] = future.result()
                except Exception:
                    results[t] = None

        # Render Loop
        cols = st.columns(cols_count)
        for i, (ticker, name) in enumerate(tickers.items()):
            data = results.get(ticker)
            
            with cols[i % cols_count]:
                if data is not None and not data.empty:
                    # Apply indicators
                    if tdsq_check: data = apply_td_sequential(data)
                    if rsi_check: data = apply_rsi_divergence(data)
                        
                    # Calculate Live Market Context Metrics
                    last_close = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
                    pct_change = ((last_close - prev_close) / prev_close) * 100
                    
                    color = "#00FFAA" if pct_change >= 0 else "#FF4B4B"
                    sign = "+" if pct_change > 0 else ""
                    icon = "🟢" if pct_change >= 0 else "🔴"
                    
                    # Inject High-Density Info Title
                    st.markdown(
                        f"<div style='text-align: center; font-family: monospace; font-size: 13px; font-weight: bold; color: #E0E6ED; padding-top: 5px;'>"
                        f"{name} ({ticker}) <br> <span style='color: {color};'>${last_close:.2f} | {sign}{pct_change:.2f}% {icon}</span>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                    
                    # Render Image from Cache
                    img_bytes = get_chart_image_bytes(ticker, data, chart_sel, style_sel, sma_check, vol_check, tdsq_check, rsi_check)
                    st.image(img_bytes, use_container_width=True)
                    
                    # The Interactive Deep Dive Button
                    if st.button("🔎 DEEP DIVE", key=f"zoom_{ticker}_{group_name}", use_container_width=True):
                        show_deep_dive(ticker, name, data)
                        
                else:
                    st.error(f"ERR: {ticker} OFFLINE")

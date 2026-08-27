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
from utils.indicators import apply_td_sequential, apply_navigator, apply_jma
from utils.renko_matrix import run_relative_strength_matrix
from utils.wave import apply_dwave

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
        'XLU': 'Utilities', 'XLB': 'Materials', 'XLRE': 'Real Estate', 'XLC': 'Comm. Svcs',
        'SMH': 'Semiconductors', 'IGV': 'Software'
    },
    'Major Stocks (US)': {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'META': 'Meta', 'BRK-B': 'Berkshire', 'TSLA': 'Tesla'
    },
    'International': {
        'EFA': 'iShares MSCI EAFE', 'EEM': 'iShares MSCI EM', 
        'EWC': 'iShares MSCI Canada ETF', 'EWA': 'iShares MSCI Australia ETF', 
        'ENZL': 'iShares MSCI New Zealand ETF', 'EWU': 'iShares MSCI United Kingdom ETF',
        'EWQ': 'iShares MSCI France ETF', 'EWG': 'iShares MSCI Germany ETF', 
        'EWJ': 'iShares MSCI Japan ETF', 'EWH': 'iShares MSCI Hong Kong ETF',
        'MCHI': 'iShares MSCI China ETF', 'EWT': 'iShares MSCI Taiwan ETF', 
        'EWY': 'iShares MSCI South Korea ETF', 'INDA': 'iShares MSCI India ETF',
        'EWZ': 'iShares MSCI Brazil ETF', 'EZA': 'iShares MSCI South Africa ETF', 
        'EWW': 'iShares MSCI Mexico ETF','AAXJ': 'iShares MSCI AXASJ',
        'EWS': 'iShares MSCI Singapore ETF', 'EWM': 'iShares MSCI Malaysia ETF', 
        'THD': 'iShares MSCI Thailand ETF', 'EIDO': 'iShares MSCI Indonesia ETF', 
        'EPHE': 'iShares MSCI Philippines ETF'
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
    'Major Hong Kong Stocks by Market Cap': {
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
def plot_single_asset(ticker, name, data, chart_type, style, show_sma, show_vol, show_tdsq, show_nav, show_jma):
    tech_types = {'OHLC': 'ohlc', 'Candlestick': 'candle', 'Renko': 'renko', 'Point and Figure': 'pnf'}
    
    if chart_type in tech_types:
        mpf_type = tech_types[chart_type]
        current_style = style if not (mpf_type in ['renko', 'pnf'] and style == 'mike') else 'yahoo'

        kwargs = dict(
            type=mpf_type, style=current_style, show_nontrading=False, returnfig=True,
            title=f"{name} ({ticker})", figsize=(5, 3.2)
        )
        
        if show_sma and mpf_type not in ['renko', 'pnf']: kwargs['mav'] = (20,50)
        if mpf_type == 'renko': kwargs['renko_params'] = {'brick_size': 'atr'}
        elif mpf_type == 'pnf': kwargs['pnf_params'] = {'box_size': 'atr'}

        # --- DYNAMIC PANEL MANAGEMENT ---
        if show_vol and 'Volume' in data.columns:
            kwargs['volume'] = True
            if show_nav and mpf_type not in ['renko', 'pnf']:
                kwargs['volume_panel'] = 2

        # --- SIGNAL OVERLAYS ---
        apds = []
        if mpf_type not in ['renko', 'pnf']:
            # 1. TDSQ Signals
            if show_tdsq and 'Setup_Signal' in data.columns:
                b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.98, np.nan)
                s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.02, np.nan)
                b13 = np.where(data['Countdown_Signal'] == 1, data['Low'] * 0.99, np.nan)
                s13 = np.where(data['Countdown_Signal'] == -1, data['High'] * 1.01, np.nan)
                
                if not np.isnan(b9).all(): apds.append(mpf.make_addplot(b9, type='scatter', marker='$9$', color='green', markersize=50, panel=0))
                if not np.isnan(s9).all(): apds.append(mpf.make_addplot(s9, type='scatter', marker='$9$', color='green', markersize=50, panel=0))
                if not np.isnan(b13).all(): apds.append(mpf.make_addplot(b13, type='scatter', marker='$13$', color='red', markersize=70, panel=0))
                if not np.isnan(s13).all(): apds.append(mpf.make_addplot(s13, type='scatter', marker='$13$', color='red', markersize=70, panel=0))

            # 2. HTF Weekly JMA Signal
            if show_jma and 'JMA' in data.columns:
                apds.append(mpf.make_addplot(data['JMA'], type='line', color='#FF00FF', panel=0, width=2.5))

            # 3. Navigator Signals
            if show_nav and 'Nav_Output' in data.columns:
                if 'Nav_Green' in data.columns and not data['Nav_Green'].isna().all():
                    apds.append(mpf.make_addplot(data['Nav_Green'], type='line', color='#00E5FF', panel=1, width=1.2))
                if 'Nav_Red' in data.columns and not data['Nav_Red'].isna().all():
                    apds.append(mpf.make_addplot(data['Nav_Red'], type='line', color='#FF9100', panel=1, width=1.2))
                if 'Nav_Signal' in data.columns and not data['Nav_Signal'].isna().all():
                    apds.append(mpf.make_addplot(data['Nav_Signal'], type='line', color='gray', panel=1, width=0.8, linestyle='--'))

                if 'Nav_W_Green' in data.columns and not data['Nav_W_Green'].isna().all():
                    apds.append(mpf.make_addplot(data['Nav_W_Green'], type='line', color='#00FF7F', panel=1, width=2.5))
                if 'Nav_W_Red' in data.columns and not data['Nav_W_Red'].isna().all():
                    apds.append(mpf.make_addplot(data['Nav_W_Red'], type='line', color='#FF1744', panel=1, width=2.5))

                # if 'Nav_Squeeze' in data.columns:
                #     squeeze = np.where(data['Nav_Squeeze'], data['Low'] * 0.97, np.nan)
                #     if not np.isnan(squeeze).all():
                #         apds.append(mpf.make_addplot(squeeze, type='scatter', marker='o', color='yellow', markersize=25, panel=0))

            # 4. DeMark D-Wave Count (Current Day Only)
            if dwave_check and 'DWave_Up_State' in data.columns and 'DWave_Dn_State' in data.columns:
                wave_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: 'A', 7: 'B', 8: 'C'}
                
                # Get the active wave strictly as of the final bar
                current_up_val = data['DWave_Up_State'].iloc[-1]
                current_dn_val = data['DWave_Dn_State'].iloc[-1]
                
                up_text = wave_labels.get(current_up_val, '0')
                dn_text = wave_labels.get(current_dn_val, '0')

                # Create empty arrays filled with NaNs for the whole chart
                last_bar_up = np.full(len(data), np.nan)
                last_bar_dn = np.full(len(data), np.nan)
                
                # Assign a value ONLY to the very last index [-1]
                if current_up_val > 0:
                    last_bar_up[-1] = data['Low'].iloc[-1] * 0.94
                    apds.append(mpf.make_addplot(last_bar_up, type='scatter', marker=f'${up_text}$', color='cyan', markersize=100, panel=0))
                    
                if current_dn_val > 0:
                    last_bar_dn[-1] = data['High'].iloc[-1] * 1.06
                    apds.append(mpf.make_addplot(last_bar_dn, type='scatter', marker=f'${dn_text}$', color='yellow', markersize=100, panel=0))

            if apds: kwargs['addplot'] = apds

        fig, axlist = mpf.plot(data, **kwargs)
        if style == 'nightclouds': fig.patch.set_facecolor('#0E1117')
        fig.subplots_adjust(top=0.82, bottom=0.15, left=0.1, right=0.9, hspace=0, wspace=0)
        return fig
        
    else:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        prices = data['Close']
        ax.plot(prices.index, prices, linewidth=1.5, color='#00FFAA' if style=='nightclouds' else 'blue')
        if show_sma: ax.plot(prices.index, prices.rolling(20).mean(), linestyle='--', color='gray', alpha=0.7)
        if show_jma and 'JMA' in data.columns:
            ax.plot(prices.index, data['JMA'], linestyle='-', color='#FF00FF', linewidth=2.5)
            
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
    sma_check = st.checkbox('SMA', value=False)
    jma_check = st.checkbox('JMA (Weekly)', value=True)
    if jma_check:
        jma_length = st.slider("JMA Length", min_value=5, max_value=100, value=7)
    
    vol_check = st.checkbox('VOLUME', value=False)
    
    tdsq_check = st.checkbox('TDSQ (Circles/Stars)', value=True)
    dwave_check = st.checkbox('Wave Count', value=True)
    nav_check = st.checkbox('NAVIGATOR (Current + Weekly)', value=True)
    
    st.divider()
    cols_count = st.slider("GRID COLUMNS", min_value=2, max_value=6, value=4)

# --- Define Background Fetch Padding ---
if interval_sel in ['1d', 'Custom Days']:
    bg_period = '5y'
elif interval_sel == '1h':
    bg_period = '730d' 
else:
    bg_period = '60d'  

# Date mapping to slice the display chart
slice_map = {
    '1mo': pd.DateOffset(months=1),
    '3mo': pd.DateOffset(months=3),
    '6mo': pd.DateOffset(months=6),
    '1y': pd.DateOffset(years=1),
    '2y': pd.DateOffset(years=2)
}

# --- 5. Main App Execution (Tabs & Grid) ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        
        # We will store the correctly sorted dictionary of tickers here
        sorted_tickers = tickers.copy() 
        
        # --- RENKO MATRIX EXPANDER & SORTING ENGINE ---
        with st.expander(f"🥊 View {group_name} Renko Relative-Strength Matrix", expanded=False):
            with st.spinner(f"Calculating vectorized pairwise ratios for {group_name}..."):
                try:
                    matrix_df = run_relative_strength_matrix(tickers, period=period_sel)
                    if not matrix_df.empty:
                        st.dataframe(matrix_df, use_container_width=True)
                        
                        # --- RE-SORT GRID BASED ON MATRIX RANKING ---
                        # The matrix output uses 'name' as its index. We need to map that back to the 'ticker' 
                        # so the grid loop can fetch data correctly.
                        name_to_ticker = {v: k for k, v in tickers.items()}
                        sorted_tickers = {}
                        
                        for asset_name in matrix_df.index:
                            if asset_name in name_to_ticker:
                                tick = name_to_ticker[asset_name]
                                sorted_tickers[tick] = asset_name
                                
                        # Append any tickers that may have failed/dropped during matrix calculation at the very end
                        for tick, asset_name in tickers.items():
                            if tick not in sorted_tickers:
                                sorted_tickers[tick] = asset_name
                                
                    else:
                        st.warning("Not enough data to calculate matrix. Chart grid will use default order.")
                except Exception as e:
                    st.error(f"Matrix calculation failed: {e}")
                    
        st.divider()
        
        # --- CHART GRID ---
        cols = st.columns(cols_count)
        
        # Iterate over the NEWLY SORTED dictionary of tickers!
        for i, (ticker, name) in enumerate(sorted_tickers.items()):
            with cols[i % cols_count]:
                with st.spinner(f"Loading {ticker}..."):
                    
                    data = fetch_data(ticker=ticker, interval=interval_sel, period=bg_period, custom_days=day_slider)
                    
                    if data is not None and not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                
                        data = data.loc[~data.index.duplicated(keep='first')]
                        
                        if tdsq_check:
                            try:
                                data = apply_td_sequential(data)
                            except Exception: pass
                                
                        if dwave_check:
                            try:
                                data = apply_dwave(data)
                            except Exception: pass
                                
                        if nav_check:
                            try:
                                data = apply_navigator(data)
                            except Exception: pass
                            
                        # --- WEEKLY JMA LOGIC ---
                        if jma_check:
                            try:
                                logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
                                if 'Volume' in data.columns:
                                    logic['Volume'] = 'sum'
                                df_w = data.resample('W').apply(logic).dropna(subset=['Close'])
                                df_w = apply_jma(df_w,length=jma_length)
                                df_w.index = df_w.index - pd.Timedelta(days=6)
                                if 'JMA' in df_w.columns:
                                    data['JMA'] = df_w['JMA'].reindex(data.index, method='ffill')
                            except Exception: pass
                            
                        # Slice to final display length
                        if not data.empty and period_sel in slice_map:
                            end_dt = data.index[-1]
                            start_dt = end_dt - slice_map[period_sel]
                            data = data.loc[start_dt:]

                        fig = plot_single_asset(
                            ticker, name, data, chart_sel, style_sel, 
                            sma_check, vol_check, tdsq_check, nav_check, jma_check
                        )
                        
                        st.pyplot(fig, width='stretch')
                        plt.close(fig) 
                    else:
                        st.error(f"ERR: {ticker}")

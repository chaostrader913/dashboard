import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.express as px
import warnings

# --- IMPORT GLOBALS ---
from utils.data_loader import fetch_data
from utils.indicators import apply_td_sequential, apply_rsi_divergence

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MOVING Averages IGNORED.*")

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
        'GLD': 'Gold', 'SLV': 'Silver', 'USO': 'Crude Oil', 'UUP': 'US Dollar', 
        'FXE': 'Euro', 'FXY': 'Jap Yen', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum'
    },
    'Major Stocks by Market Cap': {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'META': 'Meta', 'BRK-B': 'Berkshire', 'TSLA': 'Tesla'
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
            title=f"{name} ({ticker})", figsize=(5, 3.2), 
            tight_layout=False 
        )
        
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
        fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5, rect=[0, 0.03, 1, 0.88])
        return fig
        
    else:
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
        fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5, rect=[0, 0.03, 1, 0.88])
        return fig

def plot_bump_chart_plotly(resampled_data, tickers_dict, group_name, period_sel, style, lookback_months):
    """Generates an interactive Plotly Bump Chart from pre-calculated resampled data"""
    rolling_return = resampled_data.pct_change(periods=lookback_months)
    rolling_return = rolling_return.dropna(how='all')
    
    if len(rolling_return) < 2:
        st.warning(f"Not enough data points after applying a {lookback_months}-month rolling window.")
        return

    ranks = rolling_return.rank(axis=1, ascending=False, method='min')
    
    # Slice to match the UI timeframe
    slice_map = {'1mo': 3, '3mo': 4, '6mo': 7, '1y': 13, '2y': 25}
    slice_n = slice_map.get(period_sel, len(ranks))
    ranks = ranks.iloc[-slice_n:]
    
    ranks_reset = ranks.reset_index()
    df_melted = ranks_reset.melt(id_vars='Date', var_name='Ticker', value_name='Rank')
    df_melted['Name'] = df_melted['Ticker'].map(tickers_dict)
    
    max_rank = int(df_melted['Rank'].max())
    
    is_dark = (style == 'nightclouds')
    bg_color = '#0E1117' if is_dark else 'white'
    text_color = 'white' if is_dark else 'black'

    fig = px.line(
        df_melted, 
        x='Date', 
        y='Rank', 
        color='Name', 
        markers=True,
        title=f"🏆 {group_name} - {lookback_months}M Rolling Performance Rank",
        hover_data={"Date": "|%B %Y", "Ticker": False}
    )
    
    fig.update_traces(line=dict(width=4), marker=dict(size=10))

    last_date = df_melted['Date'].max()
    for ticker_name in df_melted['Name'].unique():
        ticker_data = df_melted[(df_melted['Name'] == ticker_name) & (df_melted['Date'] == last_date)]
        if not ticker_data.empty:
            last_rank = ticker_data['Rank'].values[0]
            trace_color = next(trace.line.color for trace in fig.data if trace.name == ticker_name)

            fig.add_annotation(
                x=last_date,
                y=last_rank,
                text=f" {ticker_name}",
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                font=dict(size=11, color=trace_color, family="Arial Black")
            )

    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        yaxis=dict(
            autorange="reversed",
            tickvals=list(range(1, max_rank + 1)),
            title="",
            gridcolor='#333333' if is_dark else '#e0e0e0',
            zeroline=False
        ),
        xaxis=dict(
            title="",
            gridcolor='#333333' if is_dark else '#e0e0e0',
            dtick="M1", 
            tickformat="%b\n%Y"
        ),
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        margin=dict(l=20, r=120, t=60, b=20) 
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- Fragment for the Bump Chart & Summary Table ---
@st.fragment
def render_bump_chart_ui(tickers, group_name, period_sel, style_sel):
    # 1. Fetch data once for both the table and the chart
    all_close_data = {}
    with st.spinner(f"Processing data for {group_name} Bump Chart..."):
        for ticker in tickers.keys():
            df = fetch_data(ticker=ticker, interval='1d', period='5y')
            if df is not None and not df.empty and 'Close' in df.columns:
                df = df.loc[~df.index.duplicated(keep='first')]
                all_close_data[ticker] = df['Close']
                
    if not all_close_data:
        st.warning(f"No valid data available to build bump chart for {group_name}")
        return
        
    combined_df = pd.DataFrame(all_close_data)
    
    try:
        resampled = combined_df.resample('ME').last()
    except ValueError:
        resampled = combined_df.resample('M').last() 
        
    resampled = resampled.dropna(how='all').ffill().bfill()
    
    # 2. Build the Latest Ranks Summary Table
    lookbacks = [3, 6, 9]
    summary_data = {}
    
    for lb in lookbacks:
        roll_ret = resampled.pct_change(periods=lb).dropna(how='all')
        if not roll_ret.empty:
            # Get the rank of the very last available month
            ranks = roll_ret.rank(axis=1, ascending=False, method='min').iloc[-1]
            summary_data[f"{lb}M Rank"] = ranks
            
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Map tickers to names for readability
        summary_df.index = summary_df.index.map(lambda x: tickers.get(x, x))
        summary_df.index.name = "Asset"
        
        # Sort ascending by 3M Rank, then convert to nullable integers to remove decimals
        if "3M Rank" in summary_df.columns:
            summary_df = summary_df.sort_values("3M Rank")
        summary_df = summary_df.astype('Int64')
        
        st.markdown(f"**Current Leaderboard (Latest Month)**")
        st.dataframe(summary_df, use_container_width=True)
    
    st.divider()

    # 3. Interactive Chart Controls
    lookback_sel = st.pills(
        f"Select Lookback Period for Chart", 
        options=[3, 6, 9], 
        format_func=lambda x: f"{x} Months",
        default=3,
        key=f"lookback_{group_name}" 
    )
    safe_lookback = lookback_sel if lookback_sel else 3 
    
    # Pass the pre-processed data directly into the plot function
    plot_bump_chart_plotly(resampled, tickers, group_name, period_sel, style_sel, safe_lookback)

# --- 4. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ GRID CONTROLS")
    period_sel = st.selectbox('PERIOD', ['1mo', '3mo', '6mo', '1y', '2y'], index=3)
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

# --- 5. Main App Execution (Tabs & Grid) ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        
        # --- BUMP CHART EXPANDER ---
        with st.expander(f"📊 View {group_name} Performance Bump Chart", expanded=False):
            render_bump_chart_ui(tickers, group_name, period_sel, style_sel)
            
        st.divider()
        
        # --- GRID PLOTS ---
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
                            try:
                                data = apply_td_sequential(data)
                            except: pass
                        if rsi_check:
                            try:
                                data = apply_rsi_divergence(data)
                            except: pass
                            
                        fig = plot_single_asset(ticker, name, data, chart_sel, style_sel, sma_check, vol_check, tdsq_check, rsi_check)
                        st.pyplot(fig)
                        plt.close(fig) 
                    else:
                        st.error(f"ERR: {ticker}")

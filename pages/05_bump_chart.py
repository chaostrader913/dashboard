import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import warnings

# Force a non-interactive backend for server stability
matplotlib.use('Agg')

# --- IMPORT GLOBALS ---
from utils.data_loader import fetch_data
from utils.indicators import apply_td_sequential, apply_rsi_divergence

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MOVING Averages IGNORED.*")
warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")

st.markdown("### 🌐 MODULE: MACRO MARKET GRID")
st.caption("STATIC SNAPSHOT ENGINE // BIRD'S EYE VIEW")

# --- 2. Upgraded Institutional Ticker Database ---
TICKER_GROUPS = {
 
    'Commodity, Currencies & Crypto': {
        'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil', 
        'HG=F': 'Copper', 'DX-Y.NYB':'Dollar Index',
        'EURUSD=X': 'Euro','GBPUSD=X': 'British Pound', 'JPY=X': 'Dollar Yen',
        'AUDUSD=X': 'Aussie','NZDUSD=X': 'Kiwi','EURJPY=X': 'Euro Yen',
        'CNY=X': 'Dollar Yuan','TWD=X': 'Taiwanese Dollar','HKD=X': 'Hong Kong Dollar',
        'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum','SOL-USD': 'Solana'
    },
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
        
        if show_sma and mpf_type not in ['renko', 'pnf']: kwargs['mav'] = (20,)
        if show_vol and 'Volume' in data.columns: kwargs['volume'] = True
        if mpf_type == 'renko': kwargs['renko_params'] = {'brick_size': 'atr'}
        elif mpf_type == 'pnf': kwargs['pnf_params'] = {'box_size': 'atr'}

        apds = []
        if mpf_type not in ['renko', 'pnf']:
            # 1. TDSQ Signals (Changed to standard markers to avoid font rendering crashes)
            if show_tdsq and 'Setup_Signal' in data.columns:
                b9 = np.where(data['Setup_Signal'] == 1, data['Low'] * 0.98, np.nan)
                s9 = np.where(data['Setup_Signal'] == -1, data['High'] * 1.02, np.nan)
                b13 = np.where(data['Countdown_Signal'] == 1, data['Low'] * 0.96, np.nan)
                s13 = np.where(data['Countdown_Signal'] == -1, data['High'] * 1.04, np.nan)
                
                # Setup (9) = Circles
                if not np.isnan(b9).all(): apds.append(mpf.make_addplot(b9, type='scatter', marker='o', color='#00FFAA', markersize=30))
                if not np.isnan(s9).all(): apds.append(mpf.make_addplot(s9, type='scatter', marker='o', color='#00FFAA', markersize=30))
                # Countdown (13) = Stars
                if not np.isnan(b13).all(): apds.append(mpf.make_addplot(b13, type='scatter', marker='*', color='#FF4B4B', markersize=70))
                if not np.isnan(s13).all(): apds.append(mpf.make_addplot(s13, type='scatter', marker='*', color='#FF4B4B', markersize=70))

            if show_rsi and 'Signal' in data.columns:
                rsi_b = np.where(data['Signal'] == 1, data['Low'] * 0.95, np.nan)
                if not np.isnan(rsi_b).all(): apds.append(mpf.make_addplot(rsi_b, type='scatter', marker='^', color='#00AAFF', markersize=80))

            if apds: kwargs['addplot'] = apds

        fig, axlist = mpf.plot(data, **kwargs)
        
        if style == 'nightclouds': fig.patch.set_facecolor('#0E1117')
        
        # Avoid tight_layout; use subplots_adjust for title safety
        fig.subplots_adjust(top=0.82, bottom=0.15, left=0.1, right=0.9, hspace=0, wspace=0)
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
        fig.subplots_adjust(top=0.82, bottom=0.2, left=0.15, right=0.9)
        return fig

def plot_bump_chart_plotly(resampled_data, tickers_dict, group_name, period_sel, style, lookback_months, highlight_name):
    """Generates a Storytelling-style Plotly Bump Chart using Graph Objects"""
    rolling_return = resampled_data.pct_change(periods=lookback_months)
    rolling_return = rolling_return.dropna(how='all')
    
    if len(rolling_return) < 2:
        st.warning(f"Not enough data points after applying a {lookback_months}-month rolling window.")
        return

    ranks = rolling_return.rank(axis=1, ascending=False, method='min')
    
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
    
    default_colors = px.colors.qualitative.Plotly if is_dark else px.colors.qualitative.D3
    accent_color = '#00FFAA' if is_dark else '#D62728' 
    grey_color = 'rgba(255, 255, 255, 0.15)' if is_dark else 'rgba(0, 0, 0, 0.12)'

    fig = go.Figure()
    names = list(df_melted['Name'].unique())
    is_highlight_mode = (highlight_name != "None (Show All)")
    
    if is_highlight_mode and highlight_name in names:
        names.remove(highlight_name)
        names.append(highlight_name)

    for idx, name in enumerate(names):
        ticker_data = df_melted[df_melted['Name'] == name]
        
        if is_highlight_mode:
            if name == highlight_name:
                line_color = accent_color
                line_width = 5
                opacity = 1.0
                marker_size = 12
                show_marker = True
            else:
                line_color = grey_color
                line_width = 2
                opacity = 0.8
                marker_size = 0
                show_marker = False
        else:
            line_color = default_colors[idx % len(default_colors)]
            line_width = 3.5
            opacity = 1.0
            marker_size = 8
            show_marker = True

        fig.add_trace(go.Scatter(
            x=ticker_data['Date'],
            y=ticker_data['Rank'],
            mode='lines+markers' if show_marker else 'lines',
            name=name,
            line=dict(color=line_color, width=line_width),
            marker=dict(size=marker_size, color=bg_color, line=dict(color=line_color, width=2)),
            opacity=opacity,
            hoverinfo='name+x+y'
        ))
        
        if show_marker or not is_highlight_mode: 
            last_row = ticker_data.iloc[-1]
            fig.add_annotation(
                x=last_row['Date'],
                y=last_row['Rank'],
                text=f" {name}",
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                font=dict(size=14 if is_highlight_mode else 11, 
                          color=line_color, 
                          family="Arial Black" if is_highlight_mode else "Arial")
            )

    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        title=dict(
            text=f"🏆 {group_name} - {lookback_months}M Rolling Performance Rank",
            font=dict(size=18, family="Arial Black" if is_dark else "Arial"),
            y=0.95
        ),
        yaxis=dict(
            autorange="reversed",
            tickvals=list(range(1, max_rank + 1)),
            title="Relative Rank",
            gridcolor='#222222' if is_dark else '#eeeeee',
            zeroline=False,
            showline=False
        ),
        xaxis=dict(
            title="",
            gridcolor='#222222' if is_dark else '#eeeeee',
            dtick="M1", 
            tickformat="%b\n%Y",
            showline=False
        ),
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=60, r=140, t=80, b=40) 
    )
    
    st.plotly_chart(fig, width=True)

@st.fragment
def render_bump_chart_ui(tickers, group_name, period_sel, style_sel):
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
    
    c1, c2 = st.columns([1, 2])
    with c1:
        lookback_sel = st.pills(
            f"Rolling Lookback", 
            options=[3, 6, 9], 
            format_func=lambda x: f"{x} Months",
            default=3,
            key=f"lookback_{group_name}" 
        )
    with c2:
        names_list = ["None (Show All)"] + list(tickers.values())
        highlight_sel = st.selectbox(
            "Highlight Asset",
            options=names_list,
            index=0,
            key=f"highlight_{group_name}"
        )

    safe_lookback = lookback_sel if lookback_sel else 3 
    plot_bump_chart_plotly(resampled, tickers, group_name, period_sel, style_sel, safe_lookback, highlight_sel)


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
    
    tdsq_check = st.checkbox('TDSQ (Circles/Stars)', value=True)
    rsi_check = st.checkbox('RSI DIVERGENCE', value=True)
    
    st.divider()
    cols_count = st.slider("GRID COLUMNS", min_value=2, max_value=6, value=4)


# --- 5. Main App Execution (Tabs & Grid) ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        with st.expander(f"📊 View {group_name} Performance Bump Chart", expanded=False):
            render_bump_chart_ui(tickers, group_name, period_sel, style_sel)
            
        st.divider()
        
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
                        
                        # Updated plotting call for 2026 compliance
                        st.pyplot(fig, width='stretch')
                        plt.close(fig) 
                    else:
                        st.error(f"ERR: {ticker}")

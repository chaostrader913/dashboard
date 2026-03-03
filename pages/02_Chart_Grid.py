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

# ... [Keep your TICKER_GROUPS dictionary here] ...

# --- 2. Plotting Engine ---
def plot_single_asset(ticker, name, data, chart_type, style, show_sma, show_vol):
    # ... [Keep your plot_single_asset function EXACTLY the same as the previous message] ...
    pass # (Placeholder so you don't have to copy paste the whole block again)

# --- 3. Sidebar Controls ---
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

# --- 4. Main App Execution (Tabs & Grid) ---
tabs = st.tabs(list(TICKER_GROUPS.keys()))

for tab, (group_name, tickers) in zip(tabs, TICKER_GROUPS.items()):
    with tab:
        st.write("") 
        cols = st.columns(3)
        for i, (ticker, name) in enumerate(tickers.items()):
            with cols[i % 3]:
                with st.container(border=True):
                    with st.spinner(f"Loading {ticker}..."):
                        
                        # ---> USE THE CENTRALIZED DATA LOADER <---
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
                            st.error(f"ERR: NO DATA FOR {ticker}")

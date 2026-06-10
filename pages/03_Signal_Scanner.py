import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

# Import from your utils folder
from utils.data_loader import fetch_data
from utils.indicators import (
    apply_td_sequential, 
    apply_corrected_qwma, 
    apply_jma
)

# --- 1. Terminal UI Styling ---
st.markdown("### 藤 MODULE: ADVANCED MPLFINANCE SIGNAL SCANNER")
st.divider()

# --- 2. Command Row (User Inputs) ---
col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
with col1:
    ticker = st.text_input("TARGET ASSET", value="BTC-USD").upper()
with col2:
    timeframe = st.selectbox("TIMEFRAME", options=["1d", "1h", "15m"], index=0)
with col3:
    theme_sel = st.selectbox("STYLE / THEME", ["mike", "classic", "charles", "yahoo"], index=0)
with col4:
    # New Input: Data Resampling Multiplier
    resample_n = st.number_input("RESAMPLE (N-Periods)", min_value=1, max_value=30, value=1)

# --- 3. Data Processing & Resampling ---
with st.spinner(f"EXECUTING ALGORITHMS FOR {ticker}..."):
    data = fetch_data(ticker, interval=timeframe, period="5y" if timeframe == "1d" else "60d")

if data is None or data.empty:
    st.error(f"ERR: NO DATA FOUND FOR {ticker}.")
    st.stop()

# Ensure dataframe index is a valid DatetimeIndex for mplfinance & resampling
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)

# Apply Pandas Resampling if N > 1
if resample_n > 1:
    # Map the dropdown timeframe to a valid pandas offset alias
    freq_map = {"1d": "D", "1h": "h", "15m": "min"}
    base_freq = freq_map.get(timeframe, "D")
    rule = f"{resample_n}{base_freq}"
    
    # Aggregation rules for OHLCV data
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Resample and drop incomplete/empty intervals
    data = data.resample(rule).agg(agg_dict).dropna()

# --- 4. Indicator Application ---
# Indicators must be applied AFTER resampling so they calculate on the N-day periods
data = apply_td_sequential(data)
data = apply_corrected_qwma(data)
try:
    data = apply_jma(data)
except Exception as e:
    st.warning(f"Note: Could not apply Jurik MA ({e})")

# --- 5. Heikin Ashi Calculation ---
ha_df = data.copy()
ha_close = (ha_df['Open'] + ha_df['High'] + ha_df['Low'] + ha_df['Close']) / 4
ha_open = np.zeros_like(ha_close)
ha_open[0] = (ha_df['Open'].iloc[0] + ha_df['Close'].iloc[0]) / 2
for i in range(1, len(ha_df)):
    ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2

ha_df['HAClose'] = ha_close
ha_df['HAOpen'] = ha_open
ha_df['HAHigh'] = np.maximum(ha_df['High'], np.maximum(ha_df['HAOpen'], ha_df['HAClose']))
ha_df['HALow'] = np.minimum(ha_df['Low'], np.minimum(ha_df['HAOpen'], ha_df['HAClose']))

# Filter to the last 500 records to look clean and legible
plot_df = data.iloc[-500:]
plot_ha_df = ha_df.iloc[-500:]

# --- 6. Charting Engine (Vertical Stack Dashboard) ---
st.markdown("### VISUALIZATION PANE")

# --- CHART 1: Point & Figure ---
st.markdown(f"#### Point & Figure (PnF): {ticker}")
try:
    fig1, axlist1 = mpf.plot(plot_df, type='pnf', style=theme_sel, returnfig=True, figsize=(10, 6))
    st.pyplot(fig1)
    plt.close(fig1)
except Exception as e:
    st.error(f"Error rendering PnF Chart: {e}")

st.divider()

# --- CHART 2: Renko ---
st.markdown("#### Renko Chart")
try:
    fig2, axlist2 = mpf.plot(plot_df, type='renko', mav=(5, 10, 20), style=theme_sel, returnfig=True, figsize=(10, 6))
    st.pyplot(fig2)
    plt.close(fig2)
except Exception as e:
    st.error(f"Error rendering Renko Chart: {e}")

st.divider()

# --- CHART 3: Heikin Ashi ---
st.markdown("#### Heikin Ashi with Indicators")
try:
    apds = []
    
    # 1. Corrected QWMA Lines
    if 'CQWMA' in plot_ha_df.columns:
        apds.append(mpf.make_addplot(plot_ha_df['CQWMA'], color='#00FFAA', width=2))
    if 'CQWMA_Mid' in plot_ha_df.columns:
        apds.append(mpf.make_addplot(plot_ha_df['CQWMA_Mid'], color='#888888', linestyle='dashed', width=1))
    if 'CQWMA_Up' in plot_ha_df.columns:
        apds.append(mpf.make_addplot(plot_ha_df['CQWMA_Up'], color='#00FF00', linestyle='dotted', width=1))
    if 'CQWMA_Down' in plot_ha_df.columns:
        apds.append(mpf.make_addplot(plot_ha_df['CQWMA_Down'], color='#FF0000', linestyle='dotted', width=1))
        
    # 2. Jurik MA Line
    jma_cols = [c for c in plot_ha_df.columns if 'jurik' in c.lower() or 'jma' in c.lower()]
    if jma_cols:
        apds.append(mpf.make_addplot(plot_ha_df[jma_cols[0]], color='#FFAA00', width=1.5))
        
    # 3. TD Sequential Markers
    if 'Setup_Signal' in plot_ha_df.columns:
        buy_signals = np.where(plot_ha_df['Setup_Signal'] == 1, plot_ha_df['HALow'] * 0.98, np.nan)
        sell_signals = np.where(plot_ha_df['Setup_Signal'] == -1, plot_ha_df['HAHigh'] * 1.02, np.nan)
        if not np.isnan(buy_signals).all():
            apds.append(mpf.make_addplot(buy_signals, type='scatter', marker='^', markersize=100, color='#00FFAA'))
        if not np.isnan(sell_signals).all():
            apds.append(mpf.make_addplot(sell_signals, type='scatter', marker='v', markersize=100, color='#FF4B4B'))

    if 'Countdown_Signal' in plot_ha_df.columns:
        cd_buy = np.where(plot_ha_df['Countdown_Signal'] == 1, plot_ha_df['HALow'] * 0.96, np.nan)
        cd_sell = np.where(plot_ha_df['Countdown_Signal'] == -1, plot_ha_df['HAHigh'] * 1.04, np.nan)
        if not np.isnan(cd_buy).all():
            apds.append(mpf.make_addplot(cd_buy, type='scatter', marker='*', markersize=120, color='#00AAFF'))
        if not np.isnan(cd_sell).all():
            apds.append(mpf.make_addplot(cd_sell, type='scatter', marker='*', markersize=120, color='#FFAA00'))

    # Render Candle Chart with technical indicators
    if apds:
        fig3, axlist3 = mpf.plot(
            plot_ha_df, 
            type='candle', 
            style=theme_sel, 
            volume=True,
            mav=(20, 50, 200),
            columns=['HAOpen', 'HAHigh', 'HALow', 'HAClose', 'Volume'],
            addplot=apds,
            returnfig=True,
            figsize=(12, 8)
        )
    else:
        fig3, axlist3 = mpf.plot(
            plot_ha_df, 
            type='candle', 
            style=theme_sel, 
            volume=True,
            mav=(20, 50, 200),
            columns=['HAOpen', 'HAHigh', 'HALow', 'HAClose', 'Volume'],
            returnfig=True,
            figsize=(12, 8)
        )
        
    st.pyplot(fig3)
    plt.close(fig3)
except Exception as e:
    st.error(f"Error rendering Candle Chart: {e}")

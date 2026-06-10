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
    apply_jma,
    apply_rsi_divergence
)

# --- 1. Terminal UI Styling ---
st.markdown("### MODULE: ADVANCED MPLFINANCE SIGNAL SCANNER")
st.divider()

# --- 2. Command Row (User Inputs) ---
col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
with col1:
    ticker = st.text_input("TARGET ASSET", value="BTC-USD").upper()
with col2:
    timeframe = st.selectbox("TIMEFRAME", options=["1d", "1wk", "1h"], index=0)
with col3:
    theme_sel = st.selectbox("STYLE / THEME", ["mike", "classic", "charles", "yahoo"], index=0)
with col4:
    # New Input: Data Resampling Multiplier
    resample_n = st.number_input("RESAMPLE (N-Periods)", min_value=1, max_value=30, value=1)

# --- 3. Data Processing & Resampling ---
with st.spinner(f"EXECUTING ALGORITHMS FOR {ticker}..."):
    data = fetch_data(ticker, interval=timeframe, period="5y" if timeframe == "1wk" else "2y")

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
data = apply_rsi_divergence(data)
data = apply_jma(data)

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
plot_df = data.iloc[-250:]
plot_ha_df = ha_df.iloc[-100:]

# --- 6. Charting Engine (GridSpec Dashboard) ---
st.markdown("### VISUALIZATION PANE")

# Clean OHLCV data for PnF and Renko
clean_df = plot_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()

# --- BULLETPROOF BOX SIZE CALCULATION ---
tr1 = clean_df['High'] - clean_df['Low']
tr2 = (clean_df['High'] - clean_df['Close'].shift(1)).abs()
tr3 = (clean_df['Low'] - clean_df['Close'].shift(1)).abs()
true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

safe_box_size = true_range.mean()
if pd.isna(safe_box_size) or safe_box_size <= 0:
    safe_box_size = clean_df['Close'].iloc[-1] * 0.01

safe_box_size = float(safe_box_size)

# Rename HA columns so mplfinance reads them natively
ha_render_df = plot_ha_df.copy()
ha_render_df['Open'] = ha_render_df['HAOpen']
ha_render_df['High'] = ha_render_df['HAHigh']
ha_render_df['Low'] = ha_render_df['HALow']
ha_render_df['Close'] = ha_render_df['HAClose']

# 1. Initialize Figure and GridSpec Layout (Now 5 rows to fit RSI)
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(5, 2, width_ratios=[1, 1.5], wspace=0.15, hspace=0.4)

# 2. Assign Axes to the Grid
ax_pnf = fig.add_subplot(gs[0:2, 0])       
ax_renko = fig.add_subplot(gs[2:5, 0])     # Stretch Renko slightly to match height
ax_candle = fig.add_subplot(gs[0:3, 1])    
ax_vol = fig.add_subplot(gs[3, 1], sharex=ax_candle) 
ax_rsi = fig.add_subplot(gs[4, 1], sharex=ax_candle) # New RSI Axis

# Set titles directly on the axes
ax_pnf.set_title(f"Point & Figure (PnF): {ticker}", fontsize=12)
ax_renko.set_title(f"Renko Chart: {ticker}", fontsize=12)
ax_candle.set_title("Heikin Ashi + Technicals", fontsize=12)
ax_rsi.set_title("RSI & Divergence", fontsize=10)

# --- CHART 1: Point & Figure (Left Top) ---
try:
    mpf.plot(
        clean_df, 
        type='pnf', 
        pnf_params=dict(box_size='5%', reversal=3),
        style=theme_sel, 
        ax=ax_pnf,
        returnfig=False
    )
except Exception as e:
    ax_pnf.text(0.5, 0.5, f"PnF Error: {e}", ha='center', va='center')

# --- CHART 2: Renko (Left Bottom) ---
try:
    mpf.plot(
        clean_df, 
        type='renko', 
        renko_params=dict(brick_size='atr',atr_length=14),
        style=theme_sel, 
        ax=ax_renko,
        returnfig=False
    )
except Exception as e:
    ax_renko.text(0.5, 0.5, f"Renko Error: {e}", ha='center', va='center')

# --- CHART 3: Heikin Ashi + Indicators (Right Side) ---
try:
    apds = []
    
    # 1. Corrected QWMA Lines 
    if 'CQWMA' in ha_render_df.columns:
        cqwma = ha_render_df['CQWMA']
        cqwma_diff = cqwma.diff().fillna(1)
        
        up_mask = (cqwma_diff >= 0) | (cqwma_diff.shift(-1) >= 0)
        dn_mask = (cqwma_diff < 0) | (cqwma_diff.shift(-1) < 0)
        
        cqwma_up_line = cqwma.where(up_mask, np.nan)
        cqwma_dn_line = cqwma.where(dn_mask, np.nan)
        
        apds.append(mpf.make_addplot(cqwma_up_line, color='#00FFAA', width=2, ax=ax_candle))
        apds.append(mpf.make_addplot(cqwma_dn_line, color='#FF4B4B', width=2, ax=ax_candle))
        
    if 'CQWMA_Mid' in ha_render_df.columns:
        apds.append(mpf.make_addplot(ha_render_df['CQWMA_Mid'], color='#888888', linestyle='dashed', width=1, ax=ax_candle))
    if 'CQWMA_Up' in ha_render_df.columns:
        apds.append(mpf.make_addplot(ha_render_df['CQWMA_Up'], color='#00FF00', linestyle='dotted', width=1, ax=ax_candle))
    if 'CQWMA_Down' in ha_render_df.columns:
        apds.append(mpf.make_addplot(ha_render_df['CQWMA_Down'], color='#FF0000', linestyle='dotted', width=1, ax=ax_candle))
        
    # 2. Jurik MA Line
    jma_cols = [c for c in ha_render_df.columns if 'jurik' in c.lower() or 'jma' in c.lower()]
    if jma_cols:
        apds.append(mpf.make_addplot(ha_render_df[jma_cols[0]], color='#FFAA00', width=1.5, ax=ax_candle))
        
    # 3. TD Sequential Markers 
    if 'Setup_Signal' in ha_render_df.columns:
        buy_signals = np.where(ha_render_df['Setup_Signal'] == 1, ha_render_df['HALow'] * 0.98, np.nan)
        sell_signals = np.where(ha_render_df['Setup_Signal'] == -1, ha_render_df['HAHigh'] * 1.02, np.nan)
        
        if not np.isnan(buy_signals).all():
            apds.append(mpf.make_addplot(buy_signals, type='scatter', marker='$9$', markersize=150, color='green', ax=ax_candle))
        if not np.isnan(sell_signals).all():
            apds.append(mpf.make_addplot(sell_signals, type='scatter', marker='$9$', markersize=150, color='green', ax=ax_candle))

    if 'Countdown_Signal' in ha_render_df.columns:
        cd_buy = np.where(ha_render_df['Countdown_Signal'] == 1, ha_render_df['HALow'] * 0.99, np.nan)
        cd_sell = np.where(ha_render_df['Countdown_Signal'] == -1, ha_render_df['HAHigh'] * 1.1, np.nan)
        
        if not np.isnan(cd_buy).all():
            apds.append(mpf.make_addplot(cd_buy, type='scatter', marker='$13$', markersize=200, color='red', ax=ax_candle))
        if not np.isnan(cd_sell).all():
            apds.append(mpf.make_addplot(cd_sell, type='scatter', marker='$13$', markersize=200, color='red', ax=ax_candle))

    # 4. RSI & Divergence (Targeting ax_rsi)
    if 'RSI' in ha_render_df.columns:
        # Plot Main RSI Line
        apds.append(mpf.make_addplot(ha_render_df['RSI'], color='#00AAFF', width=1.5, ax=ax_rsi))
        
        # Plot 70 and 30 Overbought/Oversold thresholds
        apds.append(mpf.make_addplot([70]*len(ha_render_df), color='#FF4B4B', linestyle='dashed', width=1, ax=ax_rsi))
        apds.append(mpf.make_addplot([30]*len(ha_render_df), color='#00FFAA', linestyle='dashed', width=1, ax=ax_rsi))
        
        # Plot Bullish Divergence Markers
        if 'Signal' in ha_render_df.columns:
            rsi_div_signals = np.where(ha_render_df['Signal'] == 1, ha_render_df['RSI'] - 5, np.nan)
            if not np.isnan(rsi_div_signals).all():
                apds.append(mpf.make_addplot(rsi_div_signals, type='scatter', marker='^', markersize=100, color='#00FFAA', ax=ax_rsi))

    # Render Main Candle Plot
    mpf.plot(
        ha_render_df, 
        type='candle', 
        style=theme_sel, 
        ax=ax_candle,
        volume=ax_vol,
        addplot=apds if apds else None,
        returnfig=False
    )
    
except Exception as e:
    ax_candle.text(0.5, 0.5, f"HA Error: {e}", ha='center', va='center')

# Display Figure in Streamlit
st.pyplot(fig)

# Prevent memory leaks
plt.close(fig)

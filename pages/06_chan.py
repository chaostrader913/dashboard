import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, cwt, morlet2
import statsmodels.api as sm

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Cycle Scanner + Scalogram")
st.title("Cycle Scanner Dashboard")

# ---------------------------------------------------------
# 1. Data Fetching & Preprocessing
# ---------------------------------------------------------
col_input1, col_input2, col_input3, col_input4 = st.columns([1, 1, 1, 1.5])
with col_input1:
    ticker = st.text_input("Ticker Symbol (yfinance)", value="DX-Y.NYB")
with col_input2:
    start_date = st.date_input("Start Date", pd.to_datetime("2021-09-07"))
with col_input3:
    end_date = st.date_input("End Date", pd.to_datetime("today"))
with col_input4:
    uploaded_file = st.file_uploader("Or Upload Custom CSV", type=['csv'])

col_slider1, col_slider2 = st.columns(2)
with col_slider1:
    st_threshold = st.slider("Stability Threshold (Genuine %)", min_value=0.0, max_value=1.0, value=0.40, step=0.01)
with col_slider2:
    hp_lambda = st.select_slider(
        "Detrending HP Lambda", 
        options=[1e5, 1e6, 1e7, 1e8, 1e9, 1e10], 
        value=1e8
    )
st.divider()

@st.cache_data(ttl=3600)
def fetch_data(t, start, end):
    df = yf.download(t, start=start, end=end)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close'].iloc[:, [0]].copy() 
        df.columns = ['Close']
    else:
        df = df[['Close']]
    df = df.dropna()
    df.reset_index(inplace=True)
    df.rename(columns={'Close': 'Price', 'Date': 'Date'}, inplace=True)
    return df

def process_csv(file):
    df = pd.read_csv(file)
    date_col = next((col for col in df.columns if 'date' in col.lower()), df.columns[0])
    close_col = next((col for col in df.columns if 'close' in col.lower() or '收盘' in col), df.columns[-1])
    df = df[[date_col, close_col]].copy()
    df.columns = ['Date', 'Price']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

if uploaded_file is not None:
    df = process_csv(uploaded_file)
else:
    df = fetch_data(ticker, start_date, end_date)

if df.empty or len(df) < 50:
    st.error("Not enough data found.")
    st.stop()

# Detrending via HP Filter
cycle_comp, _ = sm.tsa.filters.hpfilter(df["Price"], lamb=hp_lambda)
df["Detrended"] = cycle_comp

# ---------------------------------------------------------
# 2. Wavelet Scalogram (FIXED LINE)
# ---------------------------------------------------------
# Define widths for scanning (10 to 400 bars)
widths = np.linspace(10, 400, 100)

# FIXED: Using lambda to pass 'w' to morlet2
cwt_matrix = cwt(df["Detrended"], lambda M, s: morlet2(M, s, w=6.28), widths)
magnitude = np.abs(cwt_matrix)

# ---------------------------------------------------------
# 3. Cycle Detection logic
# ---------------------------------------------------------
def analyze_cycles(data, min_len=10, max_len=400):
    n = len(data)
    t_full = np.arange(n)
    lengths = np.arange(min_len, max_len + 1)
    spectrum_amps = []
    
    for L in lengths:
        omega = 2 * np.pi / L
        dft_val = (2 / n) * np.sum(data.values * np.exp(-1j * omega * t_full))
        spectrum_amps.append(np.abs(dft_val))
        
    spectrum_amps = np.array(spectrum_amps)
    full_spectrum_df = pd.DataFrame({"Len": lengths, "Amp": spectrum_amps})
    peak_indices, _ = find_peaks(spectrum_amps)
    
    cycles = []
    for idx in peak_indices:
        length = lengths[idx]
        amp = spectrum_amps[idx]
        if length > n // 3: continue
        
        omega = 2 * np.pi / length
        last_chunk = data.values[-int(length):]
        t_last = np.arange(-len(last_chunk) + 1, 1) 
        last_dft = (2 / len(last_chunk)) * np.sum(last_chunk * np.exp(-1j * omega * t_last))
        current_phase = np.angle(last_dft)
        
        n_chunks = n // int(length)
        stability = 0.0
        if n_chunks >= 3:
            chunk_phases = [np.angle((2/int(length)) * np.sum(data.values[n-(i+1)*int(length):n-i*int(length)] * np.exp(-1j * omega * np.arange(-int(length)+1, 1)))) for i in range(n_chunks)]
            stability = np.abs(np.sum(np.exp(1j * np.array(chunk_phases)))) / n_chunks
            
        is_bullish = (amp * np.cos(omega * 1 + current_phase)) > (amp * np.cos(current_phase))
        cycles.append({"Len": int(length), "Amp": round(amp, 2), "Stab": round(stability, 2), "Phase": current_phase, "Bullish": is_bullish})
            
    return pd.DataFrame(cycles), full_spectrum_df

analyzed_cycles, full_spectrum_df = analyze_cycles(df["Detrended"])

# ---------------------------------------------------------
# 4. UI & Charting
# ---------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Detected Cycles")
    if not analyzed_cycles.empty:
        analyzed_cycles.insert(0, "Select", False)
        analyzed_cycles.loc[0:1, "Select"] = True # Default select top 2
        edited_cycles = st.data_editor(analyzed_cycles[["Select", "Len", "Stab"]], hide_index=True)
        selected_lens = edited_cycles[edited_cycles["Select"]]["Len"].tolist()
        active_cycles = analyzed_cycles[analyzed_cycles["Len"].isin(selected_lens)]
    else:
        active_cycles = pd.DataFrame()

with col1:
    # Build Composite Projection
    future_bars = 100
    total_bars = len(df) + future_bars
    composite_wave = np.zeros(total_bars)
    for _, row in active_cycles.iterrows():
        x_range = np.arange(total_bars) - (len(df) - 1)
        composite_wave += row["Amp"] * np.cos((2 * np.pi / row["Len"]) * x_range + row["Phase"])

    # Scale for Price Panel
    if len(active_cycles) > 0:
        p_min, p_max = df["Price"].min(), df["Price"].max()
        c_min, c_max = composite_wave.min(), composite_wave.max()
        composite_wave_scaled = ((composite_wave - ((c_max+c_min)/2)) * ((p_max-p_min)*0.8 / (c_max-c_min or 1))) + ((p_max+p_min)/2)
    else:
        composite_wave_scaled = np.full(total_bars, np.nan)

    all_dates = pd.concat([df["Date"], pd.Series(pd.bdate_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=future_bars))]).reset_index(drop=True)

    fig_main = go.Figure()
    # 1. Background Scalogram
    fig_main.add_trace(go.Heatmap(x=df["Date"], y=widths, z=magnitude, colorscale='Plasma', opacity=0.3, yaxis='y2', showscale=False, zsmooth='best'))
    # 2. Price
    fig_main.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name='Price', line=dict(color='black', width=1.5)))
    # 3. Projection
    fig_main.add_trace(go.Scatter(x=all_dates, y=composite_wave_scaled, name='Projection', line=dict(color='#D81B60', width=2)))

    fig_main.update_layout(
        template="plotly_white", height=600,
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Cycle Length", overlaying='y', side='right', range=[10, 400], showgrid=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_main, use_container_width=True)

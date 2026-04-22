import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import statsmodels.api as sm

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Cycle Scanner")
st.title("Cycle Scanner Dashboard")

# ---------------------------------------------------------
# 1. Data Fetching & Preprocessing
# ---------------------------------------------------------
col_input1, col_input2, col_input3, col_input4 = st.columns([1, 1, 1, 1.5])
with col_input1:
    ticker = st.text_input("Ticker Symbol (yfinance)", value="CL=F")
with col_input2:
    start_date = st.date_input("Start Date", pd.to_datetime("2021-09-08"))
with col_input3:
    end_date = st.date_input("End Date", pd.to_datetime("today"))
with col_input4:
    uploaded_file = st.file_uploader("Or Upload Custom CSV", type=['csv'])

st_threshold = st.slider("Stability Threshold (Genuine %)", min_value=0.0, max_value=1.0, value=0.40, step=0.01)
st.divider()

@st.cache_data(ttl=3600)
def fetch_data(t, start, end):
    df = yf.download(t, start=start, end=end)
    if df.empty:
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close'].iloc[:, [0]].copy() 
        df.columns = ['Close']
    else:
        df = df[['Close']]
        
    df = df.dropna()
    df.reset_index(inplace=True)
    df.rename(columns={'Close': 'Price', 'index': 'Date', 'Date': 'Date'}, inplace=True)
    return df

def process_csv(file):
    df = pd.read_csv(file)
    
    # Attempt to locate Date and Close columns dynamically
    date_col = next((col for col in df.columns if 'date' in col.lower()), df.columns[0])
    close_col = next((col for col in df.columns if 'close' in col.lower() or '收盘' in col), df.columns[-1])
    
    df = df[[date_col, close_col]].copy()
    df.columns = ['Date', 'Price']
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort chronologically (oldest to newest)
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

# Determine data source
if uploaded_file is not None:
    df = process_csv(uploaded_file)
    st.success(f"Loaded {len(df)} rows from {uploaded_file.name}")
else:
    df = fetch_data(ticker, start_date, end_date)

if df.empty or len(df) < 50:
    st.error("Not enough data found. Please check your ticker or upload a valid CSV with at least 50 bars.")
    st.stop()

# True HP Filter Detrending (The secret to matching the professional tool!)
cycle_comp, trend_comp = sm.tsa.filters.hpfilter(df["Price"], lamb=1e10)
df["Detrended"] = cycle_comp

# ---------------------------------------------------------
# 2, 3, 4. High-Resolution Cycle Detection & Validation
# ---------------------------------------------------------
def analyze_cycles(data, min_len=10, max_len=400):
    n = len(data)
    t_full = np.arange(n)
    
    spectrum_amps = []
    lengths = np.arange(min_len, max_len + 1)
    
    for L in lengths:
        omega = 2 * np.pi / L
        dft_val = (2 / n) * np.sum(data.values * np.exp(-1j * omega * t_full))
        spectrum_amps.append(np.abs(dft_val))
        
    spectrum_amps = np.array(spectrum_amps)
    full_spectrum_df = pd.DataFrame({"Len": lengths, "Amp": spectrum_amps})
    
    peak_indices, _ = find_peaks(spectrum_amps)
    
    cycles = []
    actual_max_len = n // 3 
    
    for idx in peak_indices:
        length = lengths[idx]
        amp = spectrum_amps[idx]
        
        if length > actual_max_len:
            continue
            
        omega = 2 * np.pi / length
        
        last_chunk = data.values[-length:]
        t_last = np.arange(length)
        last_dft = (2 / length) * np.sum(last_chunk * np.exp(-1j * omega * t_last))
        current_phase = np.angle(last_dft)
        
        n_chunks = n // length
        if n_chunks >= 3:
            chunk_phases = []
            for c in range(n_chunks):
                chunk = data.values[c * length : (c + 1) * length]
                t_chunk = np.arange(length)
                chunk_dft = (2 / length) * np.sum(chunk * np.exp(-1j * omega * t_chunk))
                chunk_phases.append(np.angle(chunk_dft))
            
            plv = np.abs(np.sum(np.exp(1j * np.array(chunk_phases)))) / n_chunks
            stability = plv
        else:
            stability = 0.0
            
        curr_val = amp * np.cos(current_phase)
        next_val = amp * np.cos(omega * 1 + current_phase)
        is_bullish = next_val > curr_val
        
        strength = amp / length
        
        cycles.append({
            "Len": int(length),
            "Amp": round(amp, 2),
            "Strg": round(strength, 4),
            "Stab": round(stability, 2),
            "Phase": current_phase,
            "Bullish": is_bullish
        })
            
    if not cycles:
        return pd.DataFrame(), full_spectrum_df
    
    cycles_df = pd.DataFrame(cycles)
    valid_cycles = cycles_df[cycles_df["Stab"] >= st_threshold].copy()
    
    if valid_cycles.empty:
        st.warning(f"No cycles passed the {st_threshold*100}% threshold. Displaying the top 5 most stable cycles instead.")
        valid_cycles = cycles_df.sort_values(by="Stab", ascending=False).head(5).copy()
    
    valid_cycles = valid_cycles.sort_values(by="Stab", ascending=False).reset_index(drop=True)
    
    return valid_cycles, full_spectrum_df

analyzed_cycles, full_spectrum_df = analyze_cycles(df["Detrended"])

if analyzed_cycles.empty:
    st.error("Not enough data to calculate any cycles.")
    st.stop()

analyzed_cycles.insert(0, "Select", False)
analyzed_cycles.loc[0:min(2, len(analyzed_cycles)-1), "Select"] = True

# ---------------------------------------------------------
# Layout & UI
# ---------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Cycle Spectrum Data")
    
    display_cols = ["Select", "Len", "Amp", "Strg", "Stab", "Bullish"]
    df_display = analyzed_cycles[display_cols]

    # --- UPDATED: Bulletproof Matrix Styling for Streamlit ---
    def build_style_matrix(style_df):
        matrix = pd.DataFrame('', index=style_df.index, columns=style_df.columns)
        for idx in style_df.index:
            if style_df.loc[idx, 'Bullish']:
                matrix.loc[idx, 'Len'] = 'background-color: #4CAF50; color: white; font-weight: bold; text-align: center;'
            else:
                matrix.loc[idx, 'Len'] = 'background-color: #F44336; color: white; font-weight: bold; text-align: center;'
        return matrix

    styled_table = df_display.style.apply(build_style_matrix, axis=None)

    edited_cycles = st.data_editor(
        styled_table,
        hide_index=True,
        disabled=["Len", "Amp", "Strg", "Stab", "Bullish"], # Disabling columns ensures styles stick
        column_config={
            "Select": st.column_config.CheckboxColumn("✔️", default=False), 
            "Bullish": None # Hidden from UI
        },
        use_container_width=True
    )

# ---------------------------------------------------------
# Composite Projection Math
# ---------------------------------------------------------
selected_rows = edited_cycles[edited_cycles["Select"] == True]
active_cycles = analyzed_cycles[analyzed_cycles["Len"].isin(selected_rows["Len"])]

future_bars = 100
total_bars = len(df) + future_bars
composite_wave = np.zeros(total_bars)

for _, row in active_cycles.iterrows():
    length = row["Len"]
    amp = row["Amp"]
    phase = row["Phase"] 
    omega = 2 * np.pi / length
    x_range = np.arange(total_bars) - (len(df) - 1) 
    wave = amp * np.cos(omega * x_range + phase)
    composite_wave += wave

if len(active_cycles) > 0:
    comp_min, comp_max = composite_wave.min(), composite_wave.max()
    price_min, price_max = df["Price"].min(), df["Price"].max()
    
    if comp_max != comp_min:
        price_mid = (price_max + price_min) / 2
        comp_mid = (comp_max + comp_min) / 2
        scale_factor = (price_max - price_min) * 0.85 / (comp_max - comp_min)
        composite_wave_scaled = ((composite_wave - comp_mid) * scale_factor) + price_mid
    else:
        composite_wave_scaled = np.full(total_bars, (price_max + price_min) / 2)
else:
    composite_wave_scaled = np.full(total_bars, np.nan)

last_date = df["Date"].iloc[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_bars)
all_dates = pd.concat([df["Date"], pd.Series(future_dates)]).reset_index(drop=True)

# ---------------------------------------------------------
# Chart Rendering
# ---------------------------------------------------------
with col1:
    chart_title = uploaded_file.name if uploaded_file else ticker
    fig_main = go.Figure()
    
    fig_main.add_trace(go.Scatter(
        x=df["Date"], y=df["Price"], mode='lines', name='Price', 
        line=dict(color='#2B4A6F', width=1.5)
    ))
    
    fig_main.add_trace(go.Scatter(
        x=all_dates, y=composite_wave_scaled, mode='lines', name='Composite Projection', 
        line=dict(color='#D81B60', width=2, shape='spline') 
    ))
    
    fig_main.update_layout(
        title=f"{chart_title} Price and Composite Cycle", 
        xaxis_title="Date", yaxis_title="Price", 
        template="plotly_white", height=500, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
        margin=dict(l=20, r=20, t=60, b=10)
    )
    fig_main.add_vline(x=last_date.timestamp() * 1000, line_width=1, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_main, use_container_width=True)

    fig_spectrum = go.Figure()

    fig_spectrum.add_trace(go.Scatter(
        x=full_spectrum_df["Len"], 
        y=full_spectrum_df["Amp"],
        fill='tozeroy', 
        mode='lines',
        line=dict(color='#2B4A6F', width=1),
        fillcolor='rgba(140, 158, 186, 0.4)',
        name='Full Spectrum'
    ))

    bullish_peaks = analyzed_cycles[analyzed_cycles["Bullish"] == True]
    if not bullish_peaks.empty:
        fig_spectrum.add_trace(go.Scatter(
            x=bullish_peaks["Len"], y=bullish_peaks["Amp"], 
            mode='markers', name='Bullish Cycle', 
            marker=dict(symbol='triangle-up', color='#4CAF50', size=12, line=dict(width=1, color='darkgreen'))
        ))

    bearish_peaks = analyzed_cycles[analyzed_cycles["Bullish"] == False]
    if not bearish_peaks.empty:
        fig_spectrum.add_trace(go.Scatter(
            x=bearish_peaks["Len"], y=bearish_peaks["Amp"], 
            mode='markers', name='Bearish Cycle', 
            marker=dict(symbol='triangle-down', color='#F44336', size=12, line=dict(width=1, color='darkred'))
        ))

    fig_spectrum.update_layout(
        title="Cycle Spectrum Periodogram",
        xaxis_title="Cycle Length (Bars)", 
        yaxis_title="Amplitude",
        template="plotly_white", 
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 420]) 
    )
    
    st.plotly_chart(fig_spectrum, use_container_width=True)

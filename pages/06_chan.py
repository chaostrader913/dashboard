import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fft import rfft, rfftfreq
import statsmodels.api as sm

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Cycle Scanner")
st.title("Cycle Scanner Dashboard")

# ---------------------------------------------------------
# 1. Data Fetching & Preprocessing
# ---------------------------------------------------------
col_input1, col_input2, col_input3 = st.columns(3)
with col_input1:
    ticker = st.text_input("Ticker Symbol (yfinance)", value="^GSPC")
with col_input2:
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
with col_input3:
    end_date = st.date_input("End Date", pd.to_datetime("today"))

@st.cache_data(ttl=3600)
def fetch_data(t, start, end):
    df = yf.download(t, start=start, end=end)
    if df.empty:
        return pd.DataFrame()
    
    # Handle multi-index columns if present in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        # df['Close'] is ALREADY a DataFrame. We isolate the first column 
        # (the ticker) to avoid length mismatch errors.
        df = df['Close'].iloc[:, [0]].copy() 
        df.columns = ['Close']
    else:
        df = df[['Close']]
        
    df = df.dropna()
    df.reset_index(inplace=True)
    
    # yfinance sometimes names the reset index "Date" or "index". 
    # This safely handles renaming it to our standard.
    df.rename(columns={'Close': 'Price', 'index': 'Date'}, inplace=True)
    
    return df
df = fetch_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data found for the given ticker and dates.")
    st.stop()

# Step 1: Detrending using Hodrick-Prescott (HP) filter
# lambda = 14400 is standard for daily frequency data
cycle_comp, trend_comp = sm.tsa.filters.hpfilter(df["Price"], lamb=14400)
df["Detrended"] = cycle_comp

# ---------------------------------------------------------
# 2, 3, 4. Cycle Detection, Validation & Ranking
# ---------------------------------------------------------
def analyze_cycles(data, min_len=10, max_len=300):
    n = len(data)
    yf_val = rfft(data.values)
    xf_val = rfftfreq(n, 1) # 1 bar frequency
    
    amplitudes = np.abs(yf_val) / n
    
    cycles = []
    # Cap max_len to ensure we have at least 3 chunks for stability testing
    actual_max_len = min(max_len, n // 3)
    
    for i in range(1, len(xf_val)):
        freq = xf_val[i]
        length = int(round(1 / freq))
        
        if min_len <= length <= actual_max_len:
            amp = amplitudes[i]
            
            # --- Step 3: Cycle Validation (Stability / Genuine %) ---
            # Proxy for Bartels Test using Phase Synchronization (Phase Locking Value)
            # We split the data into chunks of size 'length' and measure phase consistency
            n_chunks = n // length
            if n_chunks >= 3:
                chunk_phases = []
                for c in range(n_chunks):
                    chunk = data.values[c * length : (c + 1) * length]
                    chunk_fft = rfft(chunk)
                    # The fundamental frequency of a chunk of size L is exactly at index 1
                    chunk_phases.append(np.angle(chunk_fft[1]))
                
                # Phase Locking Value calculation
                plv = np.abs(np.sum(np.exp(1j * np.array(chunk_phases)))) / n_chunks
                stability = plv
            else:
                stability = 0.0
            
            # --- End of Dataset Phase Extraction ---
            # To project accurately, we need the "current" phase of the last available cycle bar
            # as outlined in Step 2 of the whitepaper, not the average phase over the whole series.
            last_chunk = data.values[-length:]
            last_chunk_fft = rfft(last_chunk)
            current_phase = np.angle(last_chunk_fft[1])
            
            # --- Step 4: Strength Calculation ---
            strength = amp / length
            
            cycles.append({
                "Len": length,
                "Amp": round(amp, 2),
                "Strg": round(strength, 4),
                "Stab": round(stability, 2),
                "Phase": current_phase # Saved for projection math
            })
            
    if not cycles:
        return pd.DataFrame()
    
    cycles_df = pd.DataFrame(cycles)
    
    # Remove duplicates (group by length, take the one with highest stability/amplitude)
    cycles_df = cycles_df.loc[cycles_df.groupby("Len")["Stab"].idxmax()]
    
    # Filter by Genuine % (Stability > 0.49) as requested in Whitepaper Step 3
    valid_cycles = cycles_df[cycles_df["Stab"] > 0.49].copy()
    
    # Rank by Strength (Dominant driving force per bar) as requested in Whitepaper Step 4
    valid_cycles = valid_cycles.sort_values(by="Strg", ascending=False).reset_index(drop=True)
    
    return valid_cycles

analyzed_cycles = analyze_cycles(df["Detrended"])

if analyzed_cycles.empty:
    st.warning("No significant cycles passed the > 49% Stability (Genuine) threshold.")
    st.stop()

# Add UI Checkbox column (Default top 3 selected)
analyzed_cycles.insert(0, "Select", False)
analyzed_cycles.loc[0:min(2, len(analyzed_cycles)-1), "Select"] = True

# ---------------------------------------------------------
# Layout & UI
# ---------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Cycle Spectrum")
    # Interactive dataframe editor for checkboxes
    edited_cycles = st.data_editor(
        analyzed_cycles[["Select", "Len", "Amp", "Strg", "Stab"]],
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn("✔️")},
        use_container_width=True
    )

# ---------------------------------------------------------
# Composite Projection
# ---------------------------------------------------------
selected_rows = edited_cycles[edited_cycles["Select"] == True]
active_cycles = analyzed_cycles[analyzed_cycles["Len"].isin(selected_rows["Len"])]

# Build composite wave extending 100 bars into the future
future_bars = 100
total_bars = len(df) + future_bars

# We project backwards over the data and forward into the future using the *current* phase
composite_wave = np.zeros(total_bars)

# The x_axis mapping. The last data point is index len(df) - 1. 
# Our "current_phase" was calculated such that the end of the data represents the end of the wave.
for _, row in active_cycles.iterrows():
    length = row["Len"]
    amp = row["Amp"]
    phase = row["Phase"] 
    
    omega = 2 * np.pi / length
    # Adjust x mapping so that at x = len(df)-1, the phase aligns with the 'current_phase'
    x_range = np.arange(total_bars) - (len(df) - 1) 
    
    # Cosine wave based on amplitude, frequency, and localized current phase
    wave = amp * np.cos(omega * x_range + phase)
    composite_wave += wave

# Normalize the composite wave to roughly overlay the price visually
if len(active_cycles) > 0:
    comp_min, comp_max = composite_wave.min(), composite_wave.max()
    price_min, price_max = df["Price"].min(), df["Price"].max()
    
    # Scale composite wave to fit within the lower half of the price chart to match your image
    if comp_max != comp_min:
        scale_factor = (price_max - price_min) / (comp_max - comp_min) * 0.5
        composite_wave_scaled = (composite_wave - comp_min) * scale_factor + price_min
    else:
        composite_wave_scaled = np.full(total_bars, price_min)
else:
    composite_wave_scaled = np.full(total_bars, np.nan)

# Generate future dates for the x-axis
last_date = df["Date"].iloc[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_bars)
all_dates = pd.concat([df["Date"], pd.Series(future_dates)]).reset_index(drop=True)

# ---------------------------------------------------------
# Chart Rendering
# ---------------------------------------------------------
with col1:
    fig = go.Figure()

    # Original Price Line
    fig.add_trace(go.Scatter(
        x=df["Date"], 
        y=df["Price"], 
        mode='lines', 
        name='Price',
        line=dict(color='#2B4A6F', width=1.5)
    ))

    # Composite Projection Curve
    fig.add_trace(go.Scatter(
        x=all_dates, 
        y=composite_wave_scaled, 
        mode='lines', 
        name='Composite Projection',
        line=dict(color='#D32F2F', width=2, shape='spline')
    ))

    # Formatting
    fig.update_layout(
        title=f"{ticker} Price and Composite Cycle Projection",
        xaxis_title="Date",
        yaxis_title="Price / Cycle Amplitude",
        template="plotly_white",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Vertical line indicating "Today" / Projection Start
    fig.add_vline(x=last_date.timestamp() * 1000, line_width=1, line_dash="dash", line_color="grey")

    st.plotly_chart(fig, use_container_width=True)

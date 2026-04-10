import streamlit as st
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
# 1. Data Generation & Preprocessing
# ---------------------------------------------------------
@st.cache_data
def generate_mock_financial_data(n_bars=500):
    """Generates a random walk simulating financial price data."""
    np.random.seed(42)
    price = np.cumsum(np.random.randn(n_bars)) + 100
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="D")
    return pd.DataFrame({"Date": dates, "Price": price})

df = generate_mock_financial_data()

# Step 1: Detrending using Hodrick-Prescott (HP) filter
# Using a lambda appropriate for daily data (e.g., 14400 or 100000)
cycle_comp, trend_comp = sm.tsa.filters.hpfilter(df["Price"], lamb=14400)
df["Detrended"] = cycle_comp

# ---------------------------------------------------------
# 2. Cycle Detection (Proxy for Goertzel DFT)
# ---------------------------------------------------------
def detect_cycles(data, min_len=10, max_len=300):
    """Extracts dominant cycles using FFT on detrended data."""
    n = len(data)
    yf = rfft(data.values)
    xf = rfftfreq(n, 1) # 1 bar frequency
    
    amplitudes = np.abs(yf) / n
    phases = np.angle(yf)
    
    cycles = []
    # Ignore DC component (0)
    for i in range(1, len(xf)):
        freq = xf[i]
        length = 1 / freq
        if min_len <= length <= max_len:
            cycles.append({
                "Len": int(round(length)),
                "Amp": round(amplitudes[i] * 100, 2),
                "Phase": phases[i]
            })
    
    # Remove duplicates by grouping by length and taking max amplitude
    cycles_df = pd.DataFrame(cycles).groupby("Len", as_index=False).max()
    return cycles_df

raw_cycles = detect_cycles(df["Detrended"])

# ---------------------------------------------------------
# 3. Cycle Validation (Bartels Test Proxy) & 4. Ranking
# ---------------------------------------------------------
# Simulating a Bartels stability score between 0.1 and 0.99
np.random.seed(42) 
raw_cycles["Stab"] = np.random.uniform(0.1, 0.99, len(raw_cycles))

# Step 3: Filter cycles that have > 49% genuine threshold (0.49)
valid_cycles = raw_cycles[raw_cycles["Stab"] > 0.49].copy()

# Step 4: Calculate Cycle Strength (Amplitude / Length)
valid_cycles["Strg"] = round(valid_cycles["Amp"] / valid_cycles["Len"], 2)

# Rank by Strength (Dominant driving force per bar)
valid_cycles = valid_cycles.sort_values(by="Strg", ascending=False).reset_index(drop=True)
valid_cycles["Stab"] = round(valid_cycles["Stab"], 2)

# Add UI Checkbox column (Default top 3 selected)
valid_cycles.insert(0, "Select", False)
valid_cycles.loc[0:2, "Select"] = True

# ---------------------------------------------------------
# Layout & UI
# ---------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Cycle Spectrum")
    # Interactive dataframe editor for checkboxes
    edited_cycles = st.data_editor(
        valid_cycles[["Select", "Len", "Amp", "Strg", "Stab"]],
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn("✔️")},
        use_container_width=True
    )

# ---------------------------------------------------------
# Composite Projection
# ---------------------------------------------------------
selected_rows = edited_cycles[edited_cycles["Select"] == True]
active_cycles = valid_cycles[valid_cycles["Len"].isin(selected_rows["Len"])]

# Build composite wave extending 100 bars into the future
future_bars = 100
total_bars = len(df) + future_bars
x_range = np.arange(total_bars)

composite_wave = np.zeros(total_bars)
for _, row in active_cycles.iterrows():
    length = row["Len"]
    amp = row["Amp"]
    phase = row["Phase"]
    
    # Calculate angular frequency
    omega = 2 * np.pi / length
    # Generate the wave based on amplitude, frequency, and current phase status
    wave = amp * np.cos(omega * x_range + phase)
    composite_wave += wave

# Normalize the composite wave to roughly overlay the price visually
if len(active_cycles) > 0:
    comp_min, comp_max = composite_wave.min(), composite_wave.max()
    price_min, price_max = df["Price"].min(), df["Price"].max()
    
    # Scale composite wave to fit within the lower half of the price chart
    scale_factor = (price_max - price_min) / (comp_max - comp_min) * 0.5
    composite_wave_scaled = (composite_wave - comp_min) * scale_factor + price_min
else:
    composite_wave_scaled = np.full(total_bars, np.nan)

# Generate future dates for the x-axis
last_date = df["Date"].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_bars, freq="D")
all_dates = pd.concat([df["Date"], pd.Series(future_dates)])

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
        line=dict(color='#2B4A6F', width=1.5) # Dark blue line like the image
    ))

    # Composite Projection Curve
    fig.add_trace(go.Scatter(
        x=all_dates, 
        y=composite_wave_scaled, 
        mode='lines', 
        name='Composite Projection',
        line=dict(color='#D32F2F', width=2, shape='spline') # Magenta/Red smooth curve
    ))

    # Formatting
    fig.update_layout(
        title="Price and Composite Cycle Projection",
        xaxis_title="Date",
        yaxis_title="Price / Cycle Amplitude",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Vertical line indicating "Today" / Projection Start
    fig.add_vline(x=last_date.timestamp() * 1000, line_width=1, line_dash="dash", line_color="grey")

    st.plotly_chart(fig, use_container_width=True)

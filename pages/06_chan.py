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

st_threshold = st.slider("Stability Threshold (Genuine %)", min_value=0.0, max_value=1.0, value=0.49, step=0.01)
st.divider()

@st.cache_data(ttl=3600)
def fetch_data(t, start, end):
    df = yf.download(t, start=start, end=end)
    if df.empty:
        return pd.DataFrame()
    
    # Handle multi-index columns if present in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close'].iloc[:, [0]].copy() 
        df.columns = ['Close']
    else:
        df = df[['Close']]
        
    df = df.dropna()
    df.reset_index(inplace=True)
    df.rename(columns={'Close': 'Price', 'index': 'Date'}, inplace=True)
    return df

df = fetch_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data found for the given ticker and dates.")
    st.stop()

# Detrending using Hodrick-Prescott (HP) filter
cycle_comp, trend_comp = sm.tsa.filters.hpfilter(df["Price"], lamb=1e10)
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
    spectrum = []
    
    actual_max_len = min(max_len, n // 3)
    
    for i in range(1, len(xf_val)):
        freq = xf_val[i]
        length = int(round(1 / freq))
        
        if min_len <= length <= max_len:
            amp = amplitudes[i]
            
            # --- Capture the full spectrum for the subchart ---
            spectrum.append({"Len": length, "Amp": amp})
            
            # --- Cycle Validation & Details ---
            if length <= actual_max_len:
                n_chunks = n // length
                if n_chunks >= 3:
                    chunk_phases = []
                    for c in range(n_chunks):
                        chunk = data.values[c * length : (c + 1) * length]
                        chunk_fft = rfft(chunk)
                        chunk_phases.append(np.angle(chunk_fft[1]))
                    
                    plv = np.abs(np.sum(np.exp(1j * np.array(chunk_phases)))) / n_chunks
                    stability = plv
                else:
                    stability = 0.0
                
                last_chunk = data.values[-length:]
                last_chunk_fft = rfft(last_chunk)
                current_phase = np.angle(last_chunk_fft[1])
                
                # Determine Direction (Bullish vs Bearish)
                # Calculate current bar value vs next bar value based on the wave equation
                curr_val = amp * np.cos(current_phase)
                next_val = amp * np.cos((2 * np.pi / length) * 1 + current_phase)
                is_bullish = next_val > curr_val
                
                strength = amp / length
                
                cycles.append({
                    "Len": length,
                    "Amp": round(amp, 2),
                    "Strg": round(strength, 4),
                    "Stab": round(stability, 2),
                    "Phase": current_phase,
                    "Bullish": is_bullish
                })
            
    spectrum_df = pd.DataFrame(spectrum).groupby("Len", as_index=False).max()
            
    if not cycles:
        return pd.DataFrame(), spectrum_df
    
    cycles_df = pd.DataFrame(cycles)
    cycles_df = cycles_df.loc[cycles_df.groupby("Len")["Stab"].idxmax()]
    
    valid_cycles = cycles_df[cycles_df["Stab"] >= st_threshold].copy()
    
    if valid_cycles.empty:
        st.warning(f"No cycles passed the {st_threshold*100}% threshold. Displaying the top 5 most stable cycles instead.")
        valid_cycles = cycles_df.sort_values(by="Stab", ascending=False).head(5).copy()
    
    valid_cycles = valid_cycles.sort_values(by="Strg", ascending=False).reset_index(drop=True)
    
    return valid_cycles, spectrum_df

analyzed_cycles, full_spectrum_df = analyze_cycles(df["Detrended"])

if analyzed_cycles.empty:
    st.error("Not enough data to calculate any cycles. Try expanding your date range.")
    st.stop()

# Checkboxes
analyzed_cycles.insert(0, "Select", False)
analyzed_cycles.loc[0:min(2, len(analyzed_cycles)-1), "Select"] = True

# ---------------------------------------------------------
# Layout & UI
# ---------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Cycle Spectrum Data")
    
    # ---------------------------------------------------------
    # Styling the DataFrame to color 'Len' Green or Red
    # ---------------------------------------------------------
    def highlight_len_col(styled_df):
        # Create a style DataFrame of the same shape, empty by default
        styles = pd.DataFrame('', index=styled_df.index, columns=styled_df.columns)
        # Apply CSS color based on the hidden 'Bullish' column
        styles['Len'] = np.where(styled_df['Bullish'], 'color: #00C853; font-weight: bold;', 'color: #D32F2F; font-weight: bold;')
        return styles

    display_cols = ["Select", "Len", "Amp", "Strg", "Stab", "Bullish"]
    styled_table = analyzed_cycles[display_cols].style.apply(highlight_len_col, axis=None)

    edited_cycles = st.data_editor(
        styled_table,
        hide_index=True,
        # Hide the 'Bullish' boolean column from the user, but keep it for logic
        column_config={
            "Select": st.column_config.CheckboxColumn("✔️"),
            "Bullish": None 
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
        scale_factor = (price_max - price_min) / (comp_max - comp_min) * 0.5
        composite_wave_scaled = (composite_wave - comp_min) * scale_factor + price_min
    else:
        composite_wave_scaled = np.full(total_bars, price_min)
else:
    composite_wave_scaled = np.full(total_bars, np.nan)

last_date = df["Date"].iloc[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_bars)
all_dates = pd.concat([df["Date"], pd.Series(future_dates)]).reset_index(drop=True)

# ---------------------------------------------------------
# Chart Rendering
# ---------------------------------------------------------
with col1:
    # --- 1. Main Price & Composite Chart ---
    fig_main = go.Figure()

    fig_main.add_trace(go.Scatter(
        x=df["Date"], y=df["Price"], mode='lines', name='Price',
        line=dict(color='#2B4A6F', width=1.5)
    ))

    fig_main.add_trace(go.Scatter(
        x=all_dates, y=composite_wave_scaled, mode='lines', name='Composite Projection',
        line=dict(color='#D32F2F', width=2, shape='spline')
    ))

    fig_main.update_layout(
        title=f"{ticker} Price and Composite Cycle Projection",
        xaxis_title="Date", yaxis_title="Price / Cycle Amplitude",
        template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=10)
    )
    fig_main.add_vline(x=last_date.timestamp() * 1000, line_width=1, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_main, use_container_width=True)

    # --- 2. Cycle Spectrum Subchart ---
    fig_spectrum = go.Figure()

    # Fill Area for the full spectrum
    fig_spectrum.add_trace(go.Scatter(
        x=full_spectrum_df["Len"], y=full_spectrum_df["Amp"],
        fill='tozeroy', mode='lines',
        line=dict(color='#8C9EBA', width=1),
        fillcolor='rgba(140, 158, 186, 0.4)',
        name='Full Spectrum'
    ))

    # Overlay Bullish valid peaks
    bullish_peaks = analyzed_cycles[analyzed_cycles["Bullish"] == True]
    if not bullish_peaks.empty:
        fig_spectrum.add_trace(go.Scatter(
            x=bullish_peaks["Len"], y=bullish_peaks["Amp"],
            mode='markers', name='Bullish Cycle',
            marker=dict(symbol='triangle-up', color='#00C853', size=12, line=dict(width=1, color='darkgreen'))
        ))

    # Overlay Bearish valid peaks
    bearish_peaks = analyzed_cycles[analyzed_cycles["Bullish"] == False]
    if not bearish_peaks.empty:
        fig_spectrum.add_trace(go.Scatter(
            x=bearish_peaks["Len"], y=bearish_peaks["Amp"],
            mode='markers', name='Bearish Cycle',
            marker=dict(symbol='triangle-down', color='#D32F2F', size=12, line=dict(width=1, color='darkred'))
        ))

    fig_spectrum.update_layout(
        title="Cycle Spectrum Periodogram",
        xaxis_title="Cycle Length (Bars)", yaxis_title="Amplitude",
        template="plotly_white", height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    # Highlight the stability filter cutoff line visually
    threshold_line = analyzed_cycles["Amp"].min() if not analyzed_cycles.empty else 0
    if threshold_line > 0:
        fig_spectrum.add_hline(y=threshold_line, line_width=1, line_dash="dash", line_color="rgba(255, 0, 0, 0.5)")

    st.plotly_chart(fig_spectrum, use_container_width=True)

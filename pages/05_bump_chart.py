import streamlit as st
import yfinance as yf
import stumpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set page configuration
st.set_page_config(
    page_title="Stumpy Pattern Matching Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Time Series Pattern Matching Dashboard")
st.markdown("Find historical price analogs and evaluate their forward trajectories using Stumpy's z-normalized Euclidean distance.")

# --- 1. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Dashboard Controls")

ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()

col1, col2 = st.sidebar.columns(2)
with col1:
    interval = st.selectbox("Timeframe", options=["1d", "1wk", "1mo", "1h"], index=0)
with col2:
    period = st.selectbox("Data Range", options=["2y", "5y", "10y", "max"], index=2)

lookback = st.sidebar.slider("Query Lookback Window (m bars)", min_value=20, max_value=300, value=65, step=5)
num_matches = st.sidebar.slider("Number of Matches to Show", min_value=1, max_value=10, value=5, step=1)
forward_bars = st.sidebar.slider("Forward Projection (Future Bars)", min_value=0, max_value=100, value=30, step=5)

# --- 2. DATA FETCHING (CACHED) ---
@st.cache_data(ttl=3600)
def get_data(ticker_symbol, time_interval, time_period):
    try:
        df = yf.download(ticker_symbol, period=time_period, interval=time_interval, progress=False)
        if df.empty:
            return None
        # Handle multi-index columns from recent yfinance updates
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        else:
            df = df[['Close']]
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

with st.spinner(f"Fetching {ticker} data..."):
    df = get_data(ticker, interval, period)

if df is None or len(df) < (lookback * 2 + forward_bars):
    st.error(f"Not enough data points ({len(df) if df is not None else 0}) for the selected lookback ({lookback}) and forward projection ({forward_bars}). Please increase the data range or decrease the window sizes.")
    st.stop()

# Extract price array and dates
prices = df.values.flatten()
dates = df.index

# --- 3. STUMPY PATTERN MATCHING LOGIC ---
m = lookback
Q_df = prices[-m:]  # Query subsequence (latest m bars)

# Calculate distance profile against historical data (excluding current window)
search_space = prices[:-m]
distance_profile = stumpy.mass(Q_df, search_space)

# Get k initial candidates and sort by actual distance
k = max(50, num_matches * 5)
idxs = np.argpartition(distance_profile, k)[:k]
sorted_idxs = idxs[np.argsort(distance_profile[idxs])]

# Greedy non-overlapping filter
non_overlapping_matches = []
min_distance = m  # Ensure matches do not overlap by at least m bars

for idx in sorted_idxs:
    is_overlapping = any(abs(idx - existing) < min_distance for existing in non_overlapping_matches)
    if not is_overlapping:
        non_overlapping_matches.append(idx)
        if len(non_overlapping_matches) >= num_matches:
            break

# --- 4. VISUALIZATIONS ---
tab1, tab2 = st.tabs(["📊 Overview & Forward Projections", "🔍 Individual Subplot Breakdown"])

colors = ['#2ca02c', '#9467bd', '#ff7f0e', '#1f77b4', '#e377c2', '#8c564b', '#bcbd22', '#17becf', '#7f7f7f', '#d62728']

with tab1:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 1.2]})
    
    # Top Plot: Full Time Series with Highlighted Matches
    ax1.plot(dates, prices, color='gray', alpha=0.5, label='Full History', lw=1)
    ax1.plot(dates[-m:], Q_df, color='red', lw=2.5, label=f'Current Pattern (Last {m} bars)')
    
    for i, idx in enumerate(non_overlapping_matches):
        match_dates = dates[idx : idx + m]
        match_prices = prices[idx : idx + m]
        ax1.plot(match_dates, match_prices, color=colors[i % len(colors)], lw=2, label=f'Match {i+1} (Dist: {distance_profile[idx]:.2f})')
    
    ax1.set_title(f"{ticker} Full Price History & Identified Analogs", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Bottom Plot: Z-Normalized Overlay + Forward Trajectory
    Q_z_norm = stumpy.core.z_norm(Q_df)
    x_range_pattern = range(m)
    x_range_forward = range(m - 1, m + forward_bars)
    
    # Plot Query Subsequence
    ax2.plot(x_range_pattern, Q_z_norm, color="black", lw=3.5, label="Current Pattern (Q)", zorder=10)
    ax2.axvline(x=m - 1, color='red', linestyle='--', alpha=0.7, label="Present Day / Match End")
    
    for i, idx in enumerate(non_overlapping_matches):
        color = colors[i % len(colors)]
        
        # Determine how much forward data is available
        end_idx = min(idx + m + forward_bars, len(prices))
        actual_forward = end_idx - (idx + m)
        
        # Correct Z-Normalization using ONLY the match window's mean and std
        match_slice = prices[idx : idx + m]
        mu = np.mean(match_slice)
        sigma = np.std(match_slice)
        if sigma == 0:
            sigma = 1e-5  # Prevent division by zero
            
        extended_slice = prices[idx : end_idx]
        extended_z_norm = (extended_slice - mu) / sigma
        
        # Split into match pattern and forward trajectory
        match_z = extended_z_norm[:m]
        forward_z = extended_z_norm[m - 1:]  # Overlap by 1 point for continuous line
        
        ax2.plot(x_range_pattern, match_z, color=color, lw=1.8, alpha=0.8, label=f'Match {i+1} ({dates[idx].strftime("%Y-%m-%d")})')
        if actual_forward > 0:
            ax2.plot(range(m - 1, m - 1 + len(forward_z)), forward_z, color=color, lw=1.8, linestyle=':', alpha=0.8)

    ax2.set_title("Z-Normalized Pattern Alignment & Historical Forward Trajectory (Dotted)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Bars Since Start of Pattern")
    ax2.set_ylabel("Standard Deviations (Z-Score)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.markdown("### Detailed Match Breakdowns")
    fig_sub, axes = plt.subplots(len(non_overlapping_matches), 1, figsize=(14, 3.5 * len(non_overlapping_matches)), sharex=False)
    
    if len(non_overlapping_matches) == 1:
        axes = [axes]
        
    for i, idx in enumerate(non_overlapping_matches):
        ax = axes[i]
        color = colors[i % len(colors)]
        
        end_idx = min(idx + m + forward_bars, len(prices))
        plot_dates = dates[max(0, idx - 20) : end_idx]
        plot_prices = prices[max(0, idx - 20) : end_idx]
        
        # Plot local context
        ax.plot(plot_dates, plot_prices, color='gray', alpha=0.6, label='Local Context')
        
        # Highlight matched pattern
        ax.plot(dates[idx : idx + m], prices[idx : idx + m], color=color, lw=2.5, label=f'Match {i+1}')
        
        # Highlight forward trajectory if available
        if end_idx > idx + m:
            ax.plot(dates[idx + m - 1 : end_idx], prices[idx + m - 1 : end_idx], color=color, lw=2, linestyle='--', label='Forward Performance')
            
        ax.set_title(f'Match {i+1} — Started: {dates[idx].strftime("%Y-%m-%d")} | Euclidean Distance: {distance_profile[idx]:.3f}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend(loc="best")
        ax.grid(True, linestyle='--', alpha=0.4)
        
    plt.tight_layout()
    st.pyplot(fig_sub)

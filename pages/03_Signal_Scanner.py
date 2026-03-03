import streamlit as st
import datetime
import random

# 1. Page Config
st.set_page_config(
    page_title="QUANT // TERMINAL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapse sidebar by default for max screen real estate
)

# 2. Terminal CSS Overrides
st.markdown("""
<style>
    /* Tighten up the top padding so the dashboard hits the top of the screen */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Force monospace font for all metrics and numbers to align decimals */
    div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] {
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Style the containers to look like terminal panels */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #2B3040 !important;
        border-radius: 4px !important;
        background-color: #12161F !important;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
    }
    
    /* Custom Header Styling */
    h1, h2, h3 {
        color: #E0E6ED !important;
        font-weight: 600 !important;
        letter-spacing: 1px;
    }
    
    /* Make the divider look sharper */
    hr {
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        border-color: #2B3040;
    }
</style>
""", unsafe_allow_html=True)

# 3. Top Command Bar (Header + Live Clock)
head_col1, head_col2 = st.columns([4, 1])
with head_col1:
    st.markdown("### ⚡ SYS.TERMINAL // V.2.0.4")
with head_col2:
    # Right-aligned live clock feel
    st.markdown(f"<div style='text-align: right; color: #00FFAA; font-family: monospace; font-size: 1.2rem;'>{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>", unsafe_allow_html=True)

st.divider()

# 4. Live Market Tape (Simulated High-Density Metrics)
st.markdown("#### OVERVIEW // LIVE MARKETS")
cols = st.columns(6)
tickers = ["BTC/USD", "ETH/USD", "SPY", "QQQ", "DXY", "VIX"]
prices = [64230.50, 3450.20, 512.45, 438.90, 104.20, 13.45]

for i, col in enumerate(cols):
    with col:
        # Randomizing delta just for the visual effect
        delta = round(random.uniform(-2.5, 2.5), 2)
        st.metric(
            label=tickers[i], 
            value=f"{prices[i]:.2f}", 
            delta=f"{delta}%", 
            delta_color="normal" if tickers[i] != "VIX" else "inverse" # VIX goes up when market goes down
        )

st.write("")
st.write("")

# 5. Core Application Modules (The Command Center Grid)
col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.markdown("#### 📊 MODULE: CHART GRID")
        st.caption("SYNCHRONIZED MULTI-ASSET VISUALIZATION")
        st.write("Monitor up to 4 distinct tickers simultaneously with linked crosshairs, overlay indicators, and real-time tick updates.")
        if st.button("EXECUTE // CHART_GRID", use_container_width=True, type="primary"):
            st.switch_page("pages/02_Chart_Grid.py")

    with st.container(border=True):
        st.markdown("#### 📡 MODULE: SIGNAL SCANNER")
        st.caption("REAL-TIME ALGORITHMIC DETECTION")
        st.write("Scan the universe of assets for technical breakouts, RSI divergences, volume anomalies, and custom webhook alerts.")
        if st.button("EXECUTE // SCANNER", use_container_width=True, type="primary"):
            st.switch_page("pages/03_Signal_Scanner.py")

with col_right:
    with st.container(border=True):
        st.markdown("#### ⏱️ MODULE: MTF ANALYSIS")
        st.caption("SINGLE ASSET MULTIPLE TIMEFRAME")
        st.write("Deep dive into a single ticker across the 1m, 5m, 1H, and Daily charts to align macro trends with micro entries.")
        if st.button("EXECUTE // MTF_GRID", use_container_width=True):
             st.warning("ERR: MODULE OFFLINE (IN DEVELOPMENT)")

    with st.container(border=True):
        st.markdown("#### ⚙️ MODULE: BACKTEST ENGINE")
        st.caption("HISTORICAL SIMULATION & OPTIMIZATION")
        st.write("Test custom logic against historical tick data. Calculate Max Drawdown, Sharpe ratios, and optimize strategy parameters via grid search.")
        if st.button("EXECUTE // BACKTESTER", use_container_width=True):
            st.warning("ERR: MODULE OFFLINE (IN DEVELOPMENT)")

import streamlit as st
import datetime

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Trading Terminal | Home",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Header Section
st.title("⚡ Algorithmic Trading Terminal")
st.markdown(
    "Welcome to the platform. Select a module from the sidebar to begin analysis, "
    "or review the current system status below."
)
st.divider()

# 3. Simulated System Status (Adds to the "Terminal" aesthetic)
st.subheader("System Status")
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric(label="Market Data Feed", value="Connected", delta="0ms latency", delta_color="normal")
with metric_cols[1]:
    st.metric(label="Active Strategies", value="4", delta="1 newly deployed", delta_color="normal")
with metric_cols[2]:
    st.metric(label="Last Database Sync", value=datetime.datetime.now().strftime("%H:%M:%S"), delta="-2 mins", delta_color="off")
with metric_cols[3]:
    st.metric(label="Scanner Status", value="Idle", delta="Awaiting input", delta_color="off")

st.write("") # Spacer

# 4. Platform Modules (The 5 Segments)
st.subheader("Platform Modules")

# Row 1 of Modules
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### 📈 1. Chart Grid")
        st.write("Synchronized multi-asset visualization. Monitor up to 4 distinct tickers simultaneously with linked crosshairs and real-time data updates.")
        # Optional: Add a button to navigate (requires the exact file path from your st.navigation setup)
        if st.button("Launch Chart Grid", use_container_width=True):
            st.switch_page("pages/02_Chart_Grid.py")

with col2:
    with st.container(border=True):
        st.markdown("### 📡 2. Signal Scanner")
        st.write("Real-time algorithmic pattern detection. Scan the universe of assets for technical breakouts, RSI divergences, and volume anomalies.")
        if st.button("Launch Scanner", use_container_width=True):
            st.switch_page("pages/03_Signal_Scanner.py")

# Row 2 of Modules
col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.markdown("### ⏱️ 3. MTF Grid (Single Asset)")
        st.write("Multiple Timeframe analysis. Deep dive into a single ticker across the 1m, 5m, 1H, and Daily charts to align macro trends with micro entries.")
        if st.button("Launch MTF Grid", use_container_width=True):
             # Update this path when you create the file
             st.switch_page("pages/04_MTF_Analyzer.py")

with col4:
    with st.container(border=True):
        st.markdown("### ⚙️ 4. Strategy Backtesting & Optimizer")
        st.write("Historical simulation engine. Test your custom logic against historical data, calculate Drawdown/Sharpe ratios, and optimize parameters.")
        if st.button("Launch Backtester", use_container_width=True):
            st.warning("Navigate via sidebar (Link pending)")

# Row 3 (TBC)
with st.container(border=True):
    st.markdown("### 🧪 5. Alpha Labs (TBC)")
    st.write("Experimental models, machine learning sentiment analysis, and order book heatmaps. *Currently in active development.*")
    st.progress(30, text="Development Progress: 30%")


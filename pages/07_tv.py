import streamlit as st
import streamlit.components.v1 as components

st.markdown("### 🎛️ MODULE: TRADINGVIEW ADVANCED WIDGET")
st.divider()

# --- 1. Basic UI Controls ---
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    # TradingView uses specific exchange prefixes, e.g., BINANCE:BTCUSD or NASDAQ:AAPL
    ticker = st.text_input("TARGET ASSET", value="BINANCE:BTCUSD").upper()
with col2:
    # TradingView widget interval formats: "1", "15", "60", "D", "W"
    timeframe = st.selectbox("TIMEFRAME", options=["D", "60", "15"], index=0)
with col3:
    theme = st.selectbox("THEME", ["dark", "light"], index=0)

# --- 2. TradingView Widget HTML/JS ---
# We use string formatting to pass the Streamlit variables into the JS widget config
tradingview_html = f"""
<div class="tradingview-widget-container" style="height:100%;width:100%">
  <div id="tradingview_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {{
  "autosize": true,
  "symbol": "{ticker}",
  "interval": "{timeframe}",
  "timezone": "Etc/UTC",
  "theme": "{theme}",
  "style": "1",
  "locale": "en",
  "enable_publishing": false,
  "backgroundColor": "{'#131722' if theme == 'dark' else '#FFFFFF'}",
  "gridColor": "{'#363C4E' if theme == 'dark' else '#E1E5EA'}",
  "hide_top_toolbar": false,
  "hide_legend": false,
  "save_image": false,
  "container_id": "tradingview_chart"
}}
  );
  </script>
</div>
"""

# --- 3. Render in Streamlit ---
# Adjust the height as needed. 600px gives a good viewing area.
with st.container(border=True):
    components.html(tradingview_html, height=600)

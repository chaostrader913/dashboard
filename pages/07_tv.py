import streamlit as st
import streamlit.components.v1 as components

# 1. Force Streamlit to use the full width of your monitor
st.set_page_config(layout="wide") 

st.markdown("### 🎛️ MODULE: TRADINGVIEW ADVANCED WIDGET")
st.divider()

# --- Basic UI Controls ---
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    ticker = st.text_input("TARGET ASSET", value="BINANCE:BTCUSD").upper()
with col2:
    timeframe = st.selectbox("TIMEFRAME", options=["D", "60", "15"], index=0)
with col3:
    theme = st.selectbox("THEME", ["dark", "light"], index=0)

# --- 2. TradingView Widget HTML/JS ---
# Added custom CSS to force the HTML body and divs to 100% height
tradingview_html = f"""
<style>
  html, body {{
    margin: 0;
    padding: 0;
    height: 100%;
    overflow: hidden;
  }}
  .tradingview-widget-container {{
    height: 100%;
    width: 100%;
  }}
  #tradingview_chart {{
    height: 100%;
    width: 100%;
  }}
</style>

<div class="tradingview-widget-container">
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
# Bumped the height from 600 to 800 for a larger vertical canvas
with st.container(border=True):
    components.html(tradingview_html, height=800)

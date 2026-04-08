import streamlit as st

# 1. Define your pages using st.Page
# The first argument is the exact file path. 
# You can optionally add a clean title and an emoji icon for the sidebar.
page_intro = st.Page("pages/01_intro.py", title="Introduction", icon="🏠")
page_chart = st.Page("pages/02_Chart_Grid.py", title="Chart Grid", icon="📊")
page_scanner = st.Page("pages/03_Signal_Scanner.py", title="Signal Scanner", icon="📡")
page_mtf = st.Page("pages/04_MTF_Analyzer.py", title="MTF Analyzer", icon="🚀")
page_bump = st.Page("pages/05_bump_chart.py", title="Sector Rotation", icon="📈")
page_chan = st.Page("pages/05_chan.py", title="Chan Theory", icon="📈")

# 2. Initialize the navigation
# Passing them in a list dictates the order they appear in the sidebar
pg = st.navigation([page_intro, page_chart, page_scanner,page_mtf,page_bump,page_chan])

# 3. Run the selected page
pg.run()



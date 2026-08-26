import streamlit as st
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import the vectorized matrix calculator
from utils.renko_matrix import run_relative_strength_matrix

# --- Page Setup ---
st.set_page_config(page_title="Renko Matrix", page_icon="🥊", layout="wide")

st.markdown("### 🥊 MODULE: RENKO RELATIVE-STRENGTH MATRIX")
st.caption("VECTORIZED PAIRWISE TREND SCORING")

with st.expander("📖 How this works", expanded=False):
    st.markdown("""
    **The Methodology:**
    This matrix ranks a universe of assets by "relative strength." For every possible pair of assets (A and B) in the chosen group, we build a synthetic ratio (Asset A Price / Asset B Price).
    
    We then calculate a mathematical **Renko brick-walk** for that ratio. 
    * If the most recent brick is **UP**, Asset A gets a point.
    * If the most recent brick is **DOWN**, Asset B gets a point.
    
    The final score represents how many peers the asset is currently outperforming based on pure price-action trends.
    """)

# --- Institutional Ticker Database ---
TICKER_GROUPS = {
    'Indices (US)': {
        '^GSPC': 'S&P 500', '^DJI': 'Dow Jones', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000',
        'MTUM': 'US Momentum', 'VLUE': 'US Value', 'QUAL': 'US Quality', 'USMV': 'US Min Vol'
    },
    'Sectors (US)': {
        'XLK': 'Technology', 'XLV': 'Healthcare', 'XLF': 'Financials', 'XLE': 'Energy', 
        'XLI': 'Industrials', 'XLY': 'Cons. Disc.', 'XLP': 'Cons. Staples', 
        'XLU': 'Utilities', 'XLB': 'Materials', 'XLRE': 'Real Estate', 'XLC': 'Comm. Svcs'
    },
    'Themes (US)': {
        'SMH': 'Semiconductors', 'IGV': 'Software', 'XBI': 'Biotech', 'ARKK': 'Innovation', 
        'TAN': 'Solar', 'URA': 'Uranium', 'LIT': 'Lithium', 'PAVE': 'Infrastructure'
    },
    'International': {
        'VEA': 'Dev ex-US', 'VWO': 'Emerging Mkts', 'EWJ': 'Japan', 
        'FXI': 'China Large', 'INDA': 'India', 'EWG': 'Germany', 'EWU': 'UK', 'EWZ': 'Brazil'
    },
    'Fixed Income ETFs': {
        'SHY': '1-3Y Treas', 'IEF': '7-10Y Treas', 'TLT': '20Y+ Treas',
        'LQD': 'Inv. Grade', 'HYG': 'High Yield', 'BND': 'Total Bond', 
        'MBB': 'MBS ETF', 'TIP': 'TIPS Bond'
    },
    'Commodity, Currencies & Crypto': {
        'GLD': 'Gold', 'SLV': 'Silver', 'USO': 'Crude Oil', 'UUP': 'US Dollar', 
        'FXE': 'Euro', 'FXY': 'Jap Yen', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum'
    },
    'Major Stocks by Market Cap': {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'GOOGL': 'Alphabet',
        'AMZN': 'Amazon', 'META': 'Meta', 'BRK-B': 'Berkshire', 'TSLA': 'Tesla'
    }
}

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ MATRIX CONTROLS")
    
    selected_group = st.selectbox(
        "ASSET UNIVERSE", 
        options=list(TICKER_GROUPS.keys()), 
        index=6 # Defaults to Major Stocks
    )
    
    period_sel = st.selectbox(
        'CALCULATION PERIOD', 
        options=['3mo', '6mo', '1y', '2y', '5y'], 
        index=2,
        help="How far back to fetch data to calculate the Renko brick sizes and trends."
    )
    
    st.divider()
    st.markdown("#### CURRENT UNIVERSE")
    # Display the list of tickers in the sidebar for reference
    current_tickers = TICKER_GROUPS[selected_group]
    for ticker, name in current_tickers.items():
        st.text(f"• {name} ({ticker})")

# --- Main App Execution ---
st.subheader(f"Leaderboard: {selected_group}")

with st.spinner(f"Crunching the math for {len(current_tickers)} assets ({len(current_tickers) * (len(current_tickers)-1)} unique pairs)..."):
    try:
        # Run the vectorized calculation
        matrix_df = run_relative_strength_matrix(
            tickers_dict=current_tickers, 
            period=period_sel
        )
        
        if not matrix_df.empty:
            # Display using Streamlit's native dataframe with dynamic height
            st.dataframe(
                matrix_df, 
                use_container_width=True,
                height= (len(matrix_df) + 1) * 38 # Auto-adjust height so no internal scrollbar is needed
            )
        else:
            st.warning("Not enough data to calculate the matrix. Try a longer period or different assets.")
            
    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")

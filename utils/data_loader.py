import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data(ttl=300) # Cache clears every 5 mins
def fetch_data(ticker, interval="1d", period="1y"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if df.empty:
        return None
        
    # Flatten yfinance multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Drop rows with NaN values in core columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    return df
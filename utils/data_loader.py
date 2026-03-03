import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data(ttl=300) 
def fetch_data(ticker, interval="1d", period="1y", custom_days=None):
    """
    Centralized data fetcher. Handles standard intervals and custom business-day resampling.
    """
    # If the user wants custom days, we must fetch 1-day data first to build from
    fetch_i = '1d' if interval == 'Custom Days' else interval
    
    df = yf.download(ticker, period=period, interval=fetch_i, progress=False)
    
    if df.empty:
        return None
        
    # Flatten yfinance multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.index = pd.to_datetime(df.index)
    
    # --- Execute Custom Resampling ---
    if interval == 'Custom Days' and custom_days is not None:
        logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        # Resample by business days ('B')
        df = df.resample(f'{custom_days}B').apply(logic).dropna()
        
    # Drop rows with NaN values in core columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df.index.name = 'Date'
    
    return df

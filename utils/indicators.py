import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def apply_td_sequential(df):
    df = df.copy()
    # Simple squeeze to handle the yfinance 1-column-dataframe quirk
    close = df['Close'].squeeze()
    low = df['Low'].squeeze()
    high = df['High'].squeeze()

    # Setup Logic
    df['Close_vs_Close4'] = np.where(close > close.shift(4), 1, 
                                     np.where(close < close.shift(4), -1, 0))
    
    setup_count = 0
    setup_direction = 0
    df['Setup_Signal'] = 0
    
    for i in range(4, len(df)):
        current_dir = df['Close_vs_Close4'].iloc[i]
        if current_dir != 0 and current_dir == setup_direction:
            setup_count += 1
        else:
            setup_count = 1 if current_dir != 0 else 0
            setup_direction = current_dir
        
        if setup_count == 9:
            df.iloc[i, df.columns.get_loc('Setup_Signal')] = 1 if setup_direction == -1 else -1
            setup_count = 0
    return df

def apply_rsi_divergence(df, rsi_period=14, lookback=20):
    df = df.copy()
    close = df['Close'].squeeze()
    low = df['Low'].squeeze()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=rsi_period-1, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(com=rsi_period-1, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Troughs for Bullish Div
    troughs = argrelextrema(low.values, np.less_equal, order=lookback)[0]
    df['Signal'] = 0
    
    for i in range(len(troughs)):
        if i == 0: continue
        curr, prev = troughs[i], troughs[i-1]
        if low.iloc[curr] < low.iloc[prev] and df['RSI'].iloc[curr] > df['RSI'].iloc[prev]:
            if df['RSI'].iloc[curr] < 40:
                df.iloc[curr, df.columns.get_loc('Signal')] = 1
    return df

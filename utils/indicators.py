import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# --- 1. Standard Indicators (Native Pandas for Stability) ---

def apply_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    close_series = df['Close'].squeeze()
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def apply_bollinger_bands(df, length=20, std=2):
    df = df.copy()
    close_series = df['Close'].squeeze()
    df['BB_Mid'] = close_series.rolling(window=length).mean()
    rolling_std = close_series.rolling(window=length).std()
    df['BB_Upper'] = df['BB_Mid'] + (rolling_std * std)
    df['BB_Lower'] = df['BB_Mid'] - (rolling_std * std)
    return df

# --- 2. Advanced Logic (With Atomic 1D Protection) ---

def apply_rsi_divergence(df, rsi_period=14, lookback=20):
    df = df.copy()
    # Force 1D arrays immediately
    close_np = df['Close'].values.flatten()
    low_np = df['Low'].values.flatten()

    delta = pd.Series(close_np).diff()
    gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs.values))
    
    df['Trough'] = 0
    troughs = argrelextrema(low_np, np.less_equal, order=lookback)[0]
    df.loc[df.index[troughs], 'Trough'] = 1
    
    df['Signal'] = 0
    last_trough_idx = None
    
    for i in range(len(df)):
        if df['Trough'].iloc[i] == 1:
            if last_trough_idx is not None:
                if low_np[i] < low_np[last_trough_idx] and df['RSI'].iloc[i] > df['RSI'].iloc[last_trough_idx]:
                    if df['RSI'].iloc[i] < 40: 
                        df.iloc[i, df.columns.get_loc('Signal')] = 1
            last_trough_idx = i
    return df

def apply_td_sequential(df):
    df = df.copy()
    # Atomic 1D conversion
    close_np = df['Close'].values.flatten()
    low_np = df['Low'].values.flatten()
    high_np = df['High'].values.flatten()
    
    df['TD_Setup'] = 0
    df['TD_Countdown'] = 0
    df['Setup_Signal'] = 0      
    df['Countdown_Signal'] = 0  
    
    # Calculate relative changes using shifted numpy arrays
    close_shift4 = pd.Series(close_np).shift(4).values
    df['Close_vs_Close4'] = np.where(close_np > close_shift4, 1, 
                                     np.where(close_np < close_shift4, -1, 0))
    
    setup_count = 0
    setup_direction = 0
    countdown_count = 0
    active_countdown_dir = 0 

    for i in range(4, len(df)):
        current_dir = df['Close_vs_Close4'].iloc[i]
        if current_dir != 0 and current_dir == setup_direction:
            setup_count += 1
        else:
            setup_count = 1 if current_dir != 0 else 0
            setup_direction = current_dir
            
        df.iloc[i, df.columns.get_loc('TD_Setup')] = setup_count * setup_direction
        
        if setup_count == 9:
            signal_dir = 1 if setup_direction == -1 else -1
            df.iloc[i, df.columns.get_loc('Setup_Signal')] = signal_dir
            active_countdown_dir = signal_dir
            countdown_count = 0 
            setup_count = 0 
            
        if active_countdown_dir != 0 and i >= 2:
            if active_countdown_dir == 1 and close_np[i] <= low_np[i-2]:
                countdown_count += 1
            elif active_countdown_dir == -1 and close_np[i] >= high_np[i-2]:
                countdown_count += 1
                
            df.iloc[i, df.columns.get_loc('TD_Countdown')] = countdown_count * active_countdown_dir

            if countdown_count == 13:
                df.iloc[i, df.columns.get_loc('Countdown_Signal')] = active_countdown_dir
                active_countdown_dir = 0 
                countdown_count = 0
    return df

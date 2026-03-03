import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# --- 1. Standard Indicators ---

def apply_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    close_series = df['Close'].squeeze()
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    df['MACD'] = (ema_fast - ema_slow).values
    df['MACD_Signal'] = (df['MACD'].ewm(span=signal, adjust=False).mean()).values
    df['MACD_Hist'] = (df['MACD'] - df['MACD_Signal']).values
    return df

def apply_bollinger_bands(df, length=20, std=2):
    df = df.copy()
    close_series = df['Close'].squeeze()
    df['BB_Mid'] = close_series.rolling(window=length).mean().values
    rolling_std = close_series.rolling(window=length).std()
    df['BB_Upper'] = (df['BB_Mid'] + (rolling_std * std)).values
    df['BB_Lower'] = (df['BB_Mid'] - (rolling_std * std)).values
    return df

# --- 2. Advanced Logic with Length-Match Protection ---

def apply_rsi_divergence(df, rsi_period=14, lookback=20):
    df = df.copy()
    # Flatten to raw 1D numpy arrays
    close_np = df['Close'].values.flatten()
    low_np = df['Low'].values.flatten()

    delta = pd.Series(close_np).diff()
    gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
    
    rs = gain / loss
    # Assign using .values to strip pandas index and avoid length mismatch errors
    df['RSI'] = (100 - (100 / (1 + rs))).values
    
    df['Trough'] = 0
    troughs = argrelextrema(low_np, np.less_equal, order=lookback)[0]
    df.loc[df.index[troughs], 'Trough'] = 1
    
    signals = np.zeros(len(df))
    last_trough_idx = None
    
    for i in range(len(df)):
        if df['Trough'].iloc[i] == 1:
            if last_trough_idx is not None:
                if low_np[i] < low_np[last_trough_idx] and df['RSI'].iloc[i] > df['RSI'].iloc[last_trough_idx]:
                    if df['RSI'].iloc[i] < 40: 
                        signals[i] = 1
            last_trough_idx = i
            
    df['Signal'] = signals
    return df

def apply_td_sequential(df):
    df = df.copy()
    close_np = df['Close'].values.flatten()
    low_np = df['Low'].values.flatten()
    high_np = df['High'].values.flatten()
    
    # Calculate setup direction in a temporary 1D numpy array
    close_shift4 = pd.Series(close_np).shift(4).values
    # Ensure result is exactly the same length as the dataframe
    direction_results = np.zeros(len(df))
    valid_mask = ~np.isnan(close_shift4)
    direction_results[valid_mask] = np.where(close_np[valid_mask] > close_shift4[valid_mask], 1, 
                                             np.where(close_np[valid_mask] < close_shift4[valid_mask], -1, 0))
    
    df['Close_vs_Close4'] = direction_results
    
    setup_vals = np.zeros(len(df))
    setup_sig_vals = np.zeros(len(df))
    cd_vals = np.zeros(len(df))
    cd_sig_vals = np.zeros(len(df))

    setup_count = 0
    setup_direction = 0
    countdown_count = 0
    active_countdown_dir = 0 

    for i in range(len(df)):
        current_dir = direction_results[i]
        
        if current_dir != 0 and current_dir == setup_direction:
            setup_count += 1
        else:
            setup_count = 1 if current_dir != 0 else 0
            setup_direction = current_dir
            
        setup_vals[i] = setup_count * setup_direction
        
        if setup_count == 9:
            sig_dir = 1 if setup_direction == -1 else -1
            setup_sig_vals[i] = sig_dir
            active_countdown_dir = sig_dir
            countdown_count = 0 
            setup_count = 0 
            
        if active_countdown_dir != 0 and i >= 2:
            if active_countdown_dir == 1 and close_np[i] <= low_np[i-2]:
                countdown_count += 1
            elif active_countdown_dir == -1 and close_np[i] >= high_np[i-2]:
                countdown_count += 1
                
            cd_vals[i] = countdown_count * active_countdown_dir

            if countdown_count == 13:
                cd_sig_vals[i] = active_countdown_dir
                active_countdown_dir = 0 
                countdown_count = 0
                
    # Final Assignment using raw numpy arrays to bypass length/index checks
    df['TD_Setup'] = setup_vals
    df['Setup_Signal'] = setup_sig_vals
    df['TD_Countdown'] = cd_vals
    df['Countdown_Signal'] = cd_sig_vals
    
    return df

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# --- 1. Standard Indicators ---

def apply_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    # Force 1D
    c = df['Close'].values.flatten()
    c_ser = pd.Series(c)
    
    ema_fast = c_ser.ewm(span=fast, adjust=False).mean()
    ema_slow = c_ser.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    
    df['MACD'] = macd.values
    df['MACD_Signal'] = macd_sig.values
    df['MACD_Hist'] = (macd - macd_sig).values
    return df

def apply_bollinger_bands(df, length=20, std=2):
    df = df.copy()
    c = df['Close'].values.flatten()
    c_ser = pd.Series(c)
    
    mid = c_ser.rolling(window=length).mean()
    rstd = c_ser.rolling(window=length).std()
    
    df['BB_Mid'] = mid.values
    df['BB_Upper'] = (mid + (rstd * std)).values
    df['BB_Lower'] = (mid - (rstd * std)).values
    return df

# --- 2. Advanced Logic (Indestructible Arrays) ---

def apply_rsi_divergence(df, rsi_period=14, lookback=20):
    df = df.copy()
    c_np = df['Close'].values.flatten()
    l_np = df['Low'].values.flatten()

    # RSI Math
    delta = pd.Series(c_np).diff()
    gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
    rs = gain / loss
    rsi_vals = 100 - (100 / (1 + rs))
    df['RSI'] = rsi_vals.values
    
    # Trough detection
    troughs = argrelextrema(l_np, np.less_equal, order=lookback)[0]
    trough_mask = np.zeros(len(df))
    trough_mask[troughs] = 1
    df['Trough'] = trough_mask
    
    signals = np.zeros(len(df))
    last_trough = None
    
    for i in range(len(df)):
        if trough_mask[i] == 1:
            if last_trough is not None:
                # Direct comparison on raw numpy values
                if l_np[i] < l_np[last_trough] and rsi_vals.values[i] > rsi_vals.values[last_trough]:
                    if rsi_vals.values[i] < 40: 
                        signals[i] = 1
            last_trough = i
    df['Signal'] = signals
    return df

def apply_td_sequential(df):
    df = df.copy()
    # Strip everything to raw numpy 1D
    c = df['Close'].values.flatten()
    l = df['Low'].values.flatten()
    h = df['High'].values.flatten()
    
    # Create shifted array manually to ensure shape match
    c_shift4 = np.full_like(c, np.nan)
    c_shift4[4:] = c[:-4]
    
    # Initialize results
    dir_results = np.zeros(len(c))
    
    # Vectorized comparison on raw numpy (No masks needed)
    # We use np.nan_to_num to handle the first 4 bars
    dir_results = np.where(c > np.nan_to_num(c_shift4, nan=c), 1, 
                           np.where(c < np.nan_to_num(c_shift4, nan=c), -1, 0))
    # Reset first 4 bars to 0
    dir_results[:4] = 0
    
    df['Close_vs_Close4'] = dir_results
    
    setup_vals = np.zeros(len(c))
    setup_sig = np.zeros(len(c))
    cd_vals = np.zeros(len(c))
    cd_sig = np.zeros(len(c))

    setup_count = 0
    setup_dir = 0
    cd_count = 0
    active_cd_dir = 0

    for i in range(len(c)):
        curr_dir = dir_results[i]
        
        if curr_dir != 0 and curr_dir == setup_dir:
            setup_count += 1
        else:
            setup_count = 1 if curr_dir != 0 else 0
            setup_dir = curr_dir
            
        setup_vals[i] = setup_count * setup_dir
        
        if setup_count == 9:
            sig_dir = 1 if setup_dir == -1 else -1
            setup_sig[i] = sig_dir
            active_cd_dir = sig_dir
            cd_count = 0 
            setup_count = 0 
            
        if active_cd_dir != 0 and i >= 2:
            if active_cd_dir == 1 and c[i] <= l[i-2]:
                cd_count += 1
            elif active_cd_dir == -1 and c[i] >= h[i-2]:
                cd_count += 1
                
            cd_vals[i] = cd_count * active_cd_dir

            if cd_count == 13:
                cd_sig[i] = active_cd_dir
                active_cd_dir = 0 
                cd_count = 0
                
    df['TD_Setup'] = setup_vals
    df['Setup_Signal'] = setup_sig
    df['TD_Countdown'] = cd_vals
    df['Countdown_Signal'] = cd_sig
    
    return df

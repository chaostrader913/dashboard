import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def apply_rsi_divergence(df, rsi_period=14, lookback=20):
    """Calculates RSI and finds bullish divergences."""
    df = df.copy()
    
    # Squeeze ensures we are working with a 1D Series
    close_series = df['Close'].squeeze()
    low_series = df['Low'].squeeze()

    delta = close_series.diff()
    gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Detect Troughs
    df['Trough'] = 0
    troughs = argrelextrema(low_series.values, np.less_equal, order=lookback)[0]
    df.loc[df.index[troughs], 'Trough'] = 1
    
    df['Signal'] = 0
    last_trough_idx = None
    
    for i in range(len(df)):
        if df['Trough'].iloc[i] == 1:
            if last_trough_idx is not None:
                # Comparison logic
                if low_series.iloc[i] < low_series.iloc[last_trough_idx] and \
                   df['RSI'].iloc[i] > df['RSI'].iloc[last_trough_idx]:
                    if df['RSI'].iloc[i] < 40: 
                        df.iloc[i, df.columns.get_loc('Signal')] = 1
            last_trough_idx = i
    return df

def apply_td_sequential(df):
    """Calculates TD Setup (9) and TD Countdown (13)."""
    df = df.copy()
    close_s = df['Close'].squeeze()
    low_s = df['Low'].squeeze()
    high_s = df['High'].squeeze()
    
    df['TD_Setup'] = 0
    df['TD_Countdown'] = 0
    df['Setup_Signal'] = 0      
    df['Countdown_Signal'] = 0  
    
    # Setup Logic
    df['Close_vs_Close4'] = np.where(close_s > close_s.shift(4), 1, 
                                     np.where(close_s < close_s.shift(4), -1, 0))
    
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
            
        # Countdown Logic
        if active_countdown_dir != 0 and i >= 2:
            if active_countdown_dir == 1 and close_s.iloc[i] <= low_s.iloc[i-2]:
                countdown_count += 1
            elif active_countdown_dir == -1 and close_s.iloc[i] >= high_s.iloc[i-2]:
                countdown_count += 1
                
            df.iloc[i, df.columns.get_loc('TD_Countdown')] = countdown_count * active_countdown_dir

            if countdown_count == 13:
                df.iloc[i, df.columns.get_loc('Countdown_Signal')] = active_countdown_dir
                active_countdown_dir = 0 
                countdown_count = 0

    return df

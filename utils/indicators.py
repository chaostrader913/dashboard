import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import argrelextrema

# --- 1. Standard Indicators (Powered by pandas-ta) ---

def apply_macd(df, fast=12, slow=26, signal=9):
    """Calculates MACD using pandas-ta and standardizes column names."""
    df = df.copy()
    df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
    df.rename(columns={
        f"MACD_{fast}_{slow}_{signal}": "MACD",
        f"MACDh_{fast}_{slow}_{signal}": "MACD_Hist",
        f"MACDs_{fast}_{slow}_{signal}": "MACD_Signal"
    }, inplace=True, errors='ignore')
    return df

def apply_bollinger_bands(df, length=20, std=2):
    """Calculates Bollinger Bands using pandas-ta."""
    df = df.copy()
    df.ta.bbands(length=length, std=std, append=True)
    df.rename(columns={
        f"BBL_{length}_{float(std)}": "BB_Lower",
        f"BBM_{length}_{float(std)}": "BB_Mid",
        f"BBU_{length}_{float(std)}": "BB_Upper"
    }, inplace=True, errors='ignore')
    return df

# --- 2. Advanced / Custom Logic (With Squeeze Protection) ---

def apply_rsi_divergence(df, rsi_period=14, lookback=20):
    """
    Calculates RSI and finds hidden/regular divergences using local extrema.
    Protected against 2D yfinance array shape errors.
    """
    df = df.copy()
    
    # THE FIX: Squeeze core columns to strict 1D Series
    close_series = df['Close'].squeeze()
    low_series = df['Low'].squeeze()

    # 1. Calculate RSI using the squeezed 1D series
    delta = close_series.diff()
    gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. Find local minima (troughs) for Bullish Divergence
    df['Trough'] = 0
    # Scipy requires a strict 1D numpy array
    troughs = argrelextrema(low_series.values, np.less_equal, order=lookback)[0]
    df.loc[df.index[troughs], 'Trough'] = 1
    
    df['Signal'] = 0
    last_trough_idx = None
    
    # 3. Detect Divergence logic
    for i in range(len(df)):
        if df['Trough'].iloc[i] == 1:
            if last_trough_idx is not None:
                # Extract scalars safely in case of rogue nested DataFrames
                current_low = low_series.iloc[i]
                past_low = low_series.iloc[last_trough_idx]
                if isinstance(current_low, pd.Series): current_low = current_low.iloc[0]
                if isinstance(past_low, pd.Series): past_low = past_low.iloc[0]
                
                current_rsi = df['RSI'].iloc[i]
                past_rsi = df['RSI'].iloc[last_trough_idx]
                if isinstance(current_rsi, pd.Series): current_rsi = current_rsi.iloc[0]
                if isinstance(past_rsi, pd.Series): past_rsi = past_rsi.iloc[0]

                # Bullish Regular Divergence: Price makes Lower Low, RSI makes Higher Low
                if current_low < past_low and current_rsi > past_rsi:
                    # Ensure RSI is actually in oversold territory for validity
                    if current_rsi < 40: 
                        df.iloc[i, df.columns.get_loc('Signal')] = 1
                        
            last_trough_idx = i
            
    return df

def apply_td_sequential(df):
    """
    Calculates both TD Setup (9) and TD Countdown (13).
    Protected against 2D yfinance array shape errors.
    """
    df = df.copy()
    
    # THE FIX: Squeeze core columns
    close_series = df['Close'].squeeze()
    low_series = df['Low'].squeeze()
    high_series = df['High'].squeeze()
    
    df['TD_Setup'] = 0
    df['TD_Countdown'] = 0
    df['Setup_Signal'] = 0      
    df['Countdown_Signal'] = 0  
    
    df['Close_vs_Close4'] = np.where(close_series > close_series.shift(4), 1, 
                                     np.where(close_series < close_series.shift(4), -1, 0))
    
    setup_count = 0
    setup_direction = 0
    countdown_count = 0
    active_countdown_dir = 0 

    for i in range(4, len(df)):
        # --- 1. SETUP PHASE ---
        current_dir = df['Close_vs_Close4'].iloc[i]
        if isinstance(current_dir, pd.Series): current_dir = current_dir.iloc[0]
            
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
            
        # --- 2. COUNTDOWN PHASE ---
        if active_countdown_dir != 0 and i >= 2:
            current_close = close_series.iloc[i]
            if isinstance(current_close, pd.Series): current_close = current_close.iloc[0]
                
            if active_countdown_dir == 1 and current_close <= low_series.iloc[i-2]:
                countdown_count += 1
            elif active_countdown_dir == -1 and current_close >= high_series.iloc[i-2]:
                countdown_count += 1
                
            df.iloc[i, df.columns.get_loc('TD_Countdown')] = countdown_count * active_countdown_dir

            if countdown_count == 13:
                df.iloc[i, df.columns.get_loc('Countdown_Signal')] = active_countdown_dir
                active_countdown_dir = 0 
                countdown_count = 0

    return df

def apply_advanced_trendlines(df, window=5, pct_limit=5.0, breaks_limit=2, max_lines=3):
    """
    Translates the Amibroker Auto Trendlines logic.
    Identifies valid trendlines, scores them by length/significance, 
    and returns only the top N lines to prevent chart clutter.
    """
    df = df.copy()
    
    # THE FIX: Squeeze core columns
    close_series = df['Close'].squeeze()
    high_series = df['High'].squeeze()
    low_series = df['Low'].squeeze()
    
    highs = argrelextrema(high_series.values, np.greater_equal, order=window)[0]
    lows = argrelextrema(low_series.values, np.less_equal, order=window)[0]
    
    valid_upper_lines = []
    valid_lower_lines = []
    
    last_close = close_series.iloc[-1]
    if isinstance(last_close, pd.Series): last_close = last_close.iloc[0]
    
    # --- Upper Trendlines (Resistance) ---
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            idx1, idx2 = highs[i], highs[j]
            y1, y2 = high_series.iloc[idx1], high_series.iloc[idx2]
            if isinstance(y1, pd.Series): y1 = y1.iloc[0]
            if isinstance(y2, pd.Series): y2 = y2.iloc[0]
            
            slope = (y2 - y1) / (idx2 - idx1)
            end_y = y1 + slope * ((len(df) - 1) - idx1)
            
            if abs(1 - (end_y / last_close)) * 100 <= pct_limit:
                breaks = 0
                for k in range(idx2 + 1, len(df)):
                    projected_y = y1 + slope * (k - idx1)
                    curr_close = close_series.iloc[k]
                    if isinstance(curr_close, pd.Series): curr_close = curr_close.iloc[0]
                        
                    if curr_close > projected_y:
                        breaks += 1
                    if breaks > breaks_limit:
                        break 
                        
                if breaks <= breaks_limit:
                    line_length = (len(df) - 1) - idx1
                    valid_upper_lines.append((line_length, ((df.index[idx1], y1), (df.index[-1], end_y))))
                    
    # --- Lower Trendlines (Support) ---
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            idx1, idx2 = lows[i], lows[j]
            y1, y2 = low_series.iloc[idx1], low_series.iloc[idx2]
            if isinstance(y1, pd.Series): y1 = y1.iloc[0]
            if isinstance(y2, pd.Series): y2 = y2.iloc[0]
            
            slope = (y2 - y1) / (idx2 - idx1)
            end_y = y1 + slope * ((len(df) - 1) - idx1)
            
            if abs(1 - (end_y / last_close)) * 100 <= pct_limit:
                breaks = 0
                for k in range(idx2 + 1, len(df)):
                    projected_y = y1 + slope * (k - idx1)
                    curr_close = close_series.iloc[k]
                    if isinstance(curr_close, pd.Series): curr_close = curr_close.iloc[0]
                        
                    if curr_close < projected_y:
                        breaks += 1
                    if breaks > breaks_limit:
                        break 
                        
                if breaks <= breaks_limit:
                    line_length = (len(df) - 1) - idx1
                    valid_lower_lines.append((line_length, ((df.index[idx1], y1), (df.index[-1], end_y))))
                    
    valid_upper_lines.sort(key=lambda x: x[0], reverse=True)
    valid_lower_lines.sort(key=lambda x: x[0], reverse=True)
    
    top_upper = [coords for length, coords in valid_upper_lines[:max_lines]]
    top_lower = [coords for length, coords in valid_lower_lines[:max_lines]]
    
    return top_upper, top_lower

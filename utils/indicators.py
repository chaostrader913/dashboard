import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# --- 1. EMA Crossover (Existing) ---
def apply_ema_crossover(df, fast_len=9, slow_len=21):
    df = df.copy()
    df['EMA_Fast'] = df['Close'].ewm(span=fast_len, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_len, adjust=False).mean()
    df['Signal'] = 0
    
    bull_cross = (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
    bear_cross = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))
    
    df.loc[bull_cross, 'Signal'] = 1
    df.loc[bear_cross, 'Signal'] = -1
    return df

# --- 2. DeMark Setup (TD Sequential) ---
def apply_td_sequential(df):
    """
    Calculates both TD Setup (9) and TD Countdown (13).
    """
    df = df.copy()
    
    # --- THE FIX: Force columns to 1D Series to prevent shape errors ---
    close_series = df['Close'].squeeze()
    low_series = df['Low'].squeeze()
    high_series = df['High'].squeeze()
    
    df['TD_Setup'] = 0
    df['TD_Countdown'] = 0
    df['Setup_Signal'] = 0      
    df['Countdown_Signal'] = 0  
    
    # Use the squeezed 1D series for the math
    df['Close_vs_Close4'] = np.where(close_series > close_series.shift(4), 1, 
                                     np.where(close_series < close_series.shift(4), -1, 0))
    
    setup_count = 0
    setup_direction = 0
    countdown_count = 0
    active_countdown_dir = 0 

    for i in range(4, len(df)):
        # --- 1. SETUP PHASE ---
        current_dir = df['Close_vs_Close4'].iloc[i]
        
        # Extract single scalar value to prevent ambiguity
        if isinstance(current_dir, pd.Series):
            current_dir = current_dir.iloc[0]
            
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
            # Use the squeezed series for comparisons
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
# --- 3. RSI & Automated Divergence ---
def apply_rsi_divergence(df, rsi_period=14, lookback=20):
    """
    Calculates RSI and finds hidden/regular divergences using local extrema (peaks/troughs).
    """
    df = df.copy()
    
    # 1. Calculate RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
    loss = (-1 * delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. Find local minima (troughs) for Bullish Divergence
    # Looks for points that are the lowest within the 'lookback' window
    df['Trough'] = 0
    troughs = argrelextrema(df['Low'].values, np.less_equal, order=lookback)[0]
    df.loc[df.index[troughs], 'Trough'] = 1
    
    df['Signal'] = 0
    
    last_trough_idx = None
    
    # 3. Detect Divergence logic
    for i in range(len(df)):
        if df['Trough'].iloc[i] == 1:
            if last_trough_idx is not None:
                # Bullish Regular Divergence: Price makes Lower Low, RSI makes Higher Low
                if df['Low'].iloc[i] < df['Low'].iloc[last_trough_idx] and df['RSI'].iloc[i] > df['RSI'].iloc[last_trough_idx]:
                    # Ensure RSI is actually in oversold territory for validity
                    if df['RSI'].iloc[i] < 40: 
                        df.iloc[i, df.columns.get_loc('Signal')] = 1
            last_trough_idx = i
            
    # Note: You can mirror this logic with argrelextrema(np.greater_equal) for Bearish Divergence
    return df

# --- 4. Automatic Pivot Points (Basis for Trendlines) ---
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

def apply_advanced_trendlines(df, window=5, pct_limit=5.0, breaks_limit=2, max_lines=3):
    """
    Identifies valid trendlines, scores them by length/significance, 
    and returns only the top N lines to prevent chart clutter.
    """
    df = df.copy()
    
    # Increase the window to find only major swing points (macro fractals)
    highs = argrelextrema(df['High'].values, np.greater_equal, order=window)[0]
    lows = argrelextrema(df['Low'].values, np.less_equal, order=window)[0]
    
    valid_upper_lines = []
    valid_lower_lines = []
    last_close = df['Close'].iloc[-1]
    
    # --- Upper Trendlines (Resistance) ---
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            idx1, idx2 = highs[i], highs[j]
            y1, y2 = df['High'].iloc[idx1], df['High'].iloc[idx2]
            
            slope = (y2 - y1) / (idx2 - idx1)
            end_y = y1 + slope * ((len(df) - 1) - idx1)
            
            if abs(1 - (end_y / last_close)) * 100 <= pct_limit:
                breaks = 0
                for k in range(idx2 + 1, len(df)):
                    projected_y = y1 + slope * (k - idx1)
                    if df['Close'].iloc[k] > projected_y:
                        breaks += 1
                    if breaks > breaks_limit:
                        break 
                        
                if breaks <= breaks_limit:
                    line_length = (len(df) - 1) - idx1 # Score by how long the line has survived
                    valid_upper_lines.append((line_length, ((df.index[idx1], y1), (df.index[-1], end_y))))
                    
    # --- Lower Trendlines (Support) ---
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            idx1, idx2 = lows[i], lows[j]
            y1, y2 = df['Low'].iloc[idx1], df['Low'].iloc[idx2]
            
            slope = (y2 - y1) / (idx2 - idx1)
            end_y = y1 + slope * ((len(df) - 1) - idx1)
            
            if abs(1 - (end_y / last_close)) * 100 <= pct_limit:
                breaks = 0
                for k in range(idx2 + 1, len(df)):
                    projected_y = y1 + slope * (k - idx1)
                    if df['Close'].iloc[k] < projected_y:
                        breaks += 1
                    if breaks > breaks_limit:
                        break 
                        
                if breaks <= breaks_limit:
                    line_length = (len(df) - 1) - idx1
                    valid_lower_lines.append((line_length, ((df.index[idx1], y1), (df.index[-1], end_y))))
                    
    # Sort by length (descending) and keep only the top `max_lines`
    valid_upper_lines.sort(key=lambda x: x[0], reverse=True)
    valid_lower_lines.sort(key=lambda x: x[0], reverse=True)
    
    # Strip out the length score, returning just the coordinates for Plotly
    top_upper = [coords for length, coords in valid_upper_lines[:max_lines]]
    top_lower = [coords for length, coords in valid_lower_lines[:max_lines]]
    
    return top_upper, top_lower
    # Add to utils/indicators.py
def apply_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    df['MACD'] = df['Close'].ewm(span=fast, adjust=False).mean() - df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def apply_bollinger_bands(df, window=20, std=2):
    df = df.copy()
    df['BB_Mid'] = df['Close'].rolling(window=window).mean()
    df['BB_Upper'] = df['BB_Mid'] + (df['Close'].rolling(window=window).std() * std)
    df['BB_Lower'] = df['BB_Mid'] - (df['Close'].rolling(window=window).std() * std)
    return df




import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import argrelextrema

# --- 1. Standard Indicators (Powered by pandas-ta) ---

def apply_macd(df, fast=12, slow=26, signal=9):
    """Calculates MACD using pandas-ta and standardizes column names."""
    # append=True automatically adds the columns to your dataframe
    df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
    
    # Rename the dynamic pandas-ta columns to match our UI expectations
    df.rename(columns={
        f"MACD_{fast}_{slow}_{signal}": "MACD",
        f"MACDh_{fast}_{slow}_{signal}": "MACD_Hist",
        f"MACDs_{fast}_{slow}_{signal}": "MACD_Signal"
    }, inplace=True)
    return df

def apply_bollinger_bands(df, length=20, std=2):
    """Calculates Bollinger Bands using pandas-ta."""
    df.ta.bbands(length=length, std=std, append=True)
    
    df.rename(columns={
        f"BBL_{length}_{float(std)}": "BB_Lower",
        f"BBM_{length}_{float(std)}": "BB_Mid",
        f"BBU_{length}_{float(std)}": "BB_Upper"
    }, inplace=True)
    return df

# --- 2. Advanced / Custom Logic (Kept manual as they are not standard) ---

def apply_rsi_divergence(df, rsi_period=14, lookback=20):
    """
    Uses pandas-ta for the base RSI calculation, but retains our custom 
    scipy geometry logic to find the actual divergence signals.
    """
    # 1. Use pandas-ta for the clean RSI calculation
    df.ta.rsi(length=rsi_period, append=True)
    df.rename(columns={f"RSI_{rsi_period}": "RSI"}, inplace=True)
    
    # 2. Custom Divergence Geometry Logic
    df['Trough'] = 0
    troughs = argrelextrema(df['Low'].values, np.less_equal, order=lookback)[0]
    df.loc[df.index[troughs], 'Trough'] = 1
    
    df['Signal'] = 0
    last_trough_idx = None
    
    for i in range(len(df)):
        if df['Trough'].iloc[i] == 1:
            if last_trough_idx is not None:
                # Bullish Regular Divergence
                if df['Low'].iloc[i] < df['Low'].iloc[last_trough_idx] and df['RSI'].iloc[i] > df['RSI'].iloc[last_trough_idx]:
                    if df['RSI'].iloc[i] < 40: 
                        df.iloc[i, df.columns.get_loc('Signal')] = 1
            last_trough_idx = i
            
    return df

def apply_td_sequential(df):
    """
    Calculates both TD Setup (9) and TD Countdown (13).
    """
    df = df.copy()
    df['TD_Setup'] = 0
    df['TD_Countdown'] = 0
    
    # We now separate the signals so the UI can plot '9' and '13' independently
    df['Setup_Signal'] = 0      # Will trigger 1 (Buy 9) or -1 (Sell 9)
    df['Countdown_Signal'] = 0  # Will trigger 1 (Buy 13) or -1 (Sell 13)
    
    # Setup Logic: Close vs Close 4 bars ago
    df['Close_vs_Close4'] = np.where(df['Close'] > df['Close'].shift(4), 1, 
                                     np.where(df['Close'] < df['Close'].shift(4), -1, 0))
    
    setup_count = 0
    setup_direction = 0
    
    countdown_count = 0
    active_countdown_dir = 0 # Tracks if we are looking for Buy 13s or Sell 13s

    for i in range(4, len(df)):
        # --- 1. SETUP PHASE (Looking for 9) ---
        current_dir = df['Close_vs_Close4'].iloc[i]
        
        if current_dir != 0 and current_dir == setup_direction:
            setup_count += 1
        else:
            setup_count = 1 if current_dir != 0 else 0
            setup_direction = current_dir
            
        df.iloc[i, df.columns.get_loc('TD_Setup')] = setup_count * setup_direction
        
        # When Setup hits 9, record signal and activate the Countdown phase
        if setup_count == 9:
            # 1 = Buy Setup (Price dropping), -1 = Sell Setup (Price rising)
            signal_dir = 1 if setup_direction == -1 else -1
            df.iloc[i, df.columns.get_loc('Setup_Signal')] = signal_dir
            
            # Start/Restart the Countdown phase
            active_countdown_dir = signal_dir
            countdown_count = 0 
            setup_count = 0 # Reset setup to look for fresh 9s
            
        # --- 2. COUNTDOWN PHASE (Looking for 13) ---
        if active_countdown_dir != 0 and i >= 2:
            # Buy Countdown: Close must be <= True Low 2 bars prior
            if active_countdown_dir == 1 and df['Close'].iloc[i] <= df['Low'].iloc[i-2]:
                countdown_count += 1
            # Sell Countdown: Close must be >= True High 2 bars prior
            elif active_countdown_dir == -1 and df['Close'].iloc[i] >= df['High'].iloc[i-2]:
                countdown_count += 1
                
            df.iloc[i, df.columns.get_loc('TD_Countdown')] = countdown_count * active_countdown_dir

            # When Countdown hits 13, record signal and reset
            if countdown_count == 13:
                df.iloc[i, df.columns.get_loc('Countdown_Signal')] = active_countdown_dir
                active_countdown_dir = 0 # Reset after completion
                countdown_count = 0

    return df

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

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

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
def get_pivot_points(df, window=5):
    """
    Identifies Fractal Pivot Highs and Lows. 
    These coordinates can be passed to Plotly to draw automated trendlines.
    """
    df = df.copy()
    df['Pivot_High'] = np.nan
    df['Pivot_Low'] = np.nan
    
    # Find local max/min
    highs = argrelextrema(df['High'].values, np.greater_equal, order=window)[0]
    lows = argrelextrema(df['Low'].values, np.less_equal, order=window)[0]
    
    df.loc[df.index[highs], 'Pivot_High'] = df['High'].iloc[highs]
    df.loc[df.index[lows], 'Pivot_Low'] = df['Low'].iloc[lows]
    

    return df

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


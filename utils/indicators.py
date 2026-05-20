import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import argrelextrema
from numba import njit

# ==========================================
# --- NUMBA COMPILED CORE ENGINE MATH ---
# ==========================================

@njit
def _calc_cqwma_iterative(close_arr, work_arr, v1_arr):
    """Numba-optimized iterative calculation for Corrected QWMA"""
    n = len(close_arr)
    qwma = np.full(n, np.nan)
    
    # Find the first valid index to start the sequence
    first_valid = 0
    for i in range(n):
        if not np.isnan(work_arr[i]):
            first_valid = i
            break
            
    if first_valid < n:
        qwma[first_valid] = work_arr[first_valid]
        for i in range(first_valid + 1, n):
            prev = qwma[i-1]
            cur_work = work_arr[i]
            cur_v1 = v1_arr[i]

            if np.isnan(prev) or np.isnan(cur_work) or np.isnan(cur_v1):
                qwma[i] = cur_work if not np.isnan(cur_work) else prev
                continue

            v2 = (prev - cur_work) ** 2
            if v2 < cur_v1 or v2 == 0:
                c = 0.0
            else:
                c = 1.0 - (cur_v1 / v2)

            qwma[i] = prev + c * (cur_work - prev)
            
    return qwma

@njit
def _calc_dma_iterative(series_arr, alpha_arr):
    """Numba-optimized iterative calculation for Dynamic Moving Average"""
    n = len(series_arr)
    dma = np.full(n, np.nan)
    
    if n > 0:
        dma[0] = series_arr[0]
        for i in range(1, n):
            # EMA style smoothing applying the dynamic alpha
            dma[i] = alpha_arr[i] * series_arr[i] + (1.0 - alpha_arr[i]) * dma[i-1]
            
    return dma

# ==========================================
# --- 1. Standard Indicators ---
# ==========================================

def apply_macd(df, fast=12, slow=26, signal=9):
    """Calculates MACD bypassing pandas-ta to prevent EMA initialization lagging."""
    df = df.copy()
    close = df['Close']
    
    fast_ema = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    
    df['MACD'] = fast_ema - slow_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, min_periods=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df

def apply_corrected_qwma(df, ma_period=25, ma_speed=2, correction_period=0, fl_period=25, fl_up=90, fl_down=10):
    """Translates Loxx's exact PineScript Corrected QWMA to Pandas."""
    df = df.copy()
    close = df['Close']

    # 1. Calculate uncorrected QWMA
    weights = np.array([i**ma_speed for i in range(1, ma_period)] + [2 * (ma_period**ma_speed)])
    sum_weights = weights.sum()

    def calc_qwma(x):
        return np.sum(x * weights) / sum_weights

    work = close.rolling(window=ma_period).apply(calc_qwma, raw=True)

    # 2. Calculate Variance (v1)
    dev_period = correction_period if correction_period > 0 else (0 if correction_period < 0 else ma_period)
    if dev_period > 0:
        v1 = close.rolling(window=dev_period).std(ddof=0) ** 2
    else:
        v1 = pd.Series(0, index=df.index)

    # 3. Corrected QWMA Iterative Calculation (Passed to Numba Engine)
    qwma = _calc_cqwma_iterative(close.values.astype(np.float64), 
                                 work.values.astype(np.float64), 
                                 v1.values.astype(np.float64))

    df['CQWMA'] = qwma

    # 4. Floating Levels
    qwma_s = df['CQWMA']
    min_fl = qwma_s.rolling(window=fl_period).min()
    max_fl = qwma_s.rolling(window=fl_period).max()
    rng = max_fl - min_fl

    df['CQWMA_Up'] = min_fl + (fl_up * rng / 100.0)
    df['CQWMA_Down'] = min_fl + (fl_down * rng / 100.0)
    df['CQWMA_Mid'] = (df['CQWMA_Up'] + df['CQWMA_Down']) * 0.5

    # 5. Signal/Color Logic
    color_state = np.zeros(len(qwma))
    fup_arr = df['CQWMA_Up'].values
    fdn_arr = df['CQWMA_Down'].values

    for i in range(1, len(qwma)):
        if np.isnan(qwma[i]) or np.isnan(fup_arr[i]) or np.isnan(fdn_arr[i]):
            continue

        if qwma[i] > fup_arr[i]:
            color_state[i] = 1
        elif qwma[i] < fdn_arr[i]:
            color_state[i] = 2
        elif qwma[i] == qwma[i-1]:
            color_state[i] = color_state[i-1]
        else:
            color_state[i] = 0

    df['CQWMA_Color'] = color_state
    return df
    
# ==========================================
# --- 2. Advanced / Custom Logic ---
# ==========================================

def apply_rsi_divergence(df, rsi_period=14, lookback=20):
    """Base RSI + scipy geometry logic to find actual divergence signals."""
    df.ta.rsi(length=rsi_period, append=True)
    df.rename(columns={f"RSI_{rsi_period}": "RSI"}, inplace=True)
    
    df['Trough'] = 0
    troughs = argrelextrema(df['Low'].values, np.less_equal, order=lookback)[0]
    df.loc[df.index[troughs], 'Trough'] = 1
    
    df['Signal'] = 0
    last_trough_idx = None
    
    for i in range(len(df)):
        if df['Trough'].iloc[i] == 1:
            if last_trough_idx is not None:
                if df['Low'].iloc[i] < df['Low'].iloc[last_trough_idx] and df['RSI'].iloc[i] > df['RSI'].iloc[last_trough_idx]:
                    if df['RSI'].iloc[i] < 40: 
                        df.iloc[i, df.columns.get_loc('Signal')] = 1
            last_trough_idx = i
            
    return df

def apply_td_sequential(df):
    """Calculates both TD Setup (9) and TD Countdown (13)."""
    df = df.copy()
    df['TD_Setup'] = 0
    df['TD_Countdown'] = 0
    
    df['Setup_Signal'] = 0      
    df['Countdown_Signal'] = 0  
    
    df['Close_vs_Close4'] = np.where(df['Close'] > df['Close'].shift(4), 1, 
                                     np.where(df['Close'] < df['Close'].shift(4), -1, 0))
    
    setup_count, setup_direction = 0, 0
    countdown_count, active_countdown_dir = 0, 0

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
            if active_countdown_dir == 1 and df['Close'].iloc[i] <= df['Low'].iloc[i-2]:
                countdown_count += 1
            elif active_countdown_dir == -1 and df['Close'].iloc[i] >= df['High'].iloc[i-2]:
                countdown_count += 1
                
            df.iloc[i, df.columns.get_loc('TD_Countdown')] = countdown_count * active_countdown_dir

            if countdown_count == 13:
                df.iloc[i, df.columns.get_loc('Countdown_Signal')] = active_countdown_dir
                active_countdown_dir = 0 
                countdown_count = 0

    return df

def apply_jma(df, length=7, phase=0):
    """Calculates the Jurik Moving Average (JMA) utilizing pandas-ta."""
    df = df.copy()
    df.ta.jma(length=length, phase=phase, append=True)
    jma_cols = [col for col in df.columns if col.startswith('JMA_')]
    if jma_cols:
        df.rename(columns={jma_cols[-1]: 'JMA'}, inplace=True)
    return df

def apply_natural_moving_average(df, length=40):
    """Calculates Jim Sloman's Ocean Natural Moving Average (NMA)."""
    df = df.copy()
    close = df['Close']
    
    weights = np.sqrt(np.arange(1, length + 1)) - np.sqrt(np.arange(0, length))
    weights_reversed = weights[::-1]
    
    delta = close.diff()
    abs_delta = delta.abs()
    
    def calc_weighted_sum(arr):
        return np.sum(arr * weights_reversed)
        
    nmr = delta.rolling(window=length).apply(calc_weighted_sum, raw=True).abs()
    total_delta = abs_delta.rolling(window=length).apply(calc_weighted_sum, raw=True)
    
    ratio = nmr / total_delta.replace(0, np.nan)
    ratio = ratio.fillna(0) 
    
    nma = np.full(len(close), np.nan)
    close_arr = close.values
    ratio_arr = ratio.values
    
    first_valid = ratio.notna().idxmax()
    if pd.notna(first_valid) and first_valid in df.index:
        start_idx = df.index.get_loc(first_valid)
        nma[start_idx] = close_arr[start_idx]
        
        for i in range(start_idx + 1, len(close)):
            alpha = ratio_arr[i]
            nma[i] = nma[i-1] + alpha * (close_arr[i] - nma[i-1])
            
    df['NMA'] = nma
    return df

def apply_natural_market_channel(df, nma_length=40, atr_length=14, multiplier=1.5):
    """Builds adaptive volatility bands around the NMA."""
    df = df.copy()
    if 'NMA' not in df.columns:
        df = apply_natural_moving_average(df, length=nma_length)
        
    df.ta.atr(length=atr_length, append=True)
    atr_col = [col for col in df.columns if col.startswith('ATRr_')][-1]
    
    df['NMC_Upper'] = df['NMA'] + (df[atr_col] * multiplier)
    df['NMC_Lower'] = df['NMA'] - (df[atr_col] * multiplier)
    return df

def apply_dma(df, base_length=10, smoothing=2, col_name='DMA'):
    """Dynamic Moving Average: adjusts length based on price momentum (ROC)."""
    df = df.copy()
    series = df['Close']
    
    roc = series.pct_change(periods=base_length) * 100
    norm_roc = np.clip(abs(roc) / 5, 0.2, 1.0)
    dyn_len = base_length * (1.5 - norm_roc)
    dyn_len = np.clip(dyn_len.fillna(base_length), 5, base_length * 2)
    
    alpha = smoothing / (dyn_len + 1)
    
    # Pass to Numba Engine
    dma = _calc_dma_iterative(series.values.astype(np.float64), 
                              alpha.values.astype(np.float64))
    
    df[col_name] = pd.Series(dma, index=series.index)
    return df

def apply_dma_bands(df, dma_length=10, atr_length=14, multiplier=2.5):
    """DMA Bands: Dynamic bands using exponential ATR around the faster DMA."""
    df = df.copy()
    if 'DMA' not in df.columns:
        df = apply_dma(df, base_length=dma_length)
        
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.DataFrame(index=df.index)
    tr['h_l'] = high - low
    tr['h_pc'] = abs(high - close.shift(1))
    tr['l_pc'] = abs(low - close.shift(1))
    tr['tr'] = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    
    atr = tr['tr'].ewm(span=atr_length, min_periods=atr_length).mean()
    
    df['DMA_Upper'] = df['DMA'] + (multiplier * atr)
    df['DMA_Lower'] = df['DMA'] - (multiplier * atr)
    df['DMA_Mid'] = df['DMA']
    return df

import numpy as np
import pandas as pd
from numba import njit

@njit
def rolling_max(arr, window):
    n = len(arr)
    res = np.full(n, np.nan)
    for i in range(window, n):
        res[i] = np.max(arr[i-window:i])
    return res

@njit
def rolling_min(arr, window):
    n = len(arr)
    res = np.full(n, np.nan)
    for i in range(window, n):
        res[i] = np.min(arr[i-window:i])
    return res

@njit
def calc_dwave_state(highs, lows, closes, dir_up=True):
    n = len(closes)
    # This will hold the active wave state at every given bar
    state = np.zeros(n, dtype=np.int32) 
    
    wave_sizes = np.array([21, 13, 8, 21, 13, 34, 13, 8, 21])
    thresholds = np.zeros((9, n))
    
    for w in range(9):
        is_odd = (w % 2 != 0)
        is_up_search = (dir_up and is_odd) or (not dir_up and not is_odd)
        if is_up_search:
            thresholds[w] = rolling_max(highs, wave_sizes[w])
        else:
            thresholds[w] = rolling_min(lows, wave_sizes[w])
            
    current_wave = 0
    
    for i in range(1, n):
        if np.isnan(thresholds[current_wave, i-1]):
            state[i] = current_wave
            continue
            
        is_odd = (current_wave % 2 != 0)
        is_up_search = (dir_up and is_odd) or (not dir_up and not is_odd)
        
        triggered = False
        if is_up_search:
            if highs[i] > thresholds[current_wave, i-1]:
                triggered = True
        else:
            if lows[i] < thresholds[current_wave, i-1]:
                triggered = True
                
        if triggered:
            current_wave += 1
            if current_wave > 8:
                current_wave = 0
                
        # Record the current state
        state[i] = current_wave
                
    return state

def apply_dwave(df):
    highs = df['High'].to_numpy()
    lows = df['Low'].to_numpy()
    closes = df['Close'].to_numpy()
    
    # 0 = Trend Start, 1-5 = Impulse, 6 = A, 7 = B, 8 = C
    df['DWave_Up_State'] = calc_dwave_state(highs, lows, closes, dir_up=True)
    df['DWave_Dn_State'] = calc_dwave_state(highs, lows, closes, dir_up=False)
    
    return df

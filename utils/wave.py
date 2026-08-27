import numpy as np
import pandas as pd
from numba import njit

# --- NUMBA OPTIMIZED ROLLING FUNCTIONS ---
@njit
def rolling_max(arr, window):
    """Numba-optimized rolling maximum for lookback periods."""
    n = len(arr)
    res = np.full(n, np.nan)
    for i in range(window, n):
        res[i] = np.max(arr[i-window:i])
    return res

@njit
def rolling_min(arr, window):
    """Numba-optimized rolling minimum for lookback periods."""
    n = len(arr)
    res = np.full(n, np.nan)
    for i in range(window, n):
        res[i] = np.min(arr[i-window:i])
    return res

# --- CORE STATE MACHINE ---
@njit
def calc_dwave_sequence(highs, lows, closes, dir_up=True):
    """
    Calculates the DeMark D-Wave sequence using a flattened state machine.
    dir_up = True calculates the Up-Wave sequence. False calculates Down-Wave.
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int32)
    
    # D-Wave Lookback Levels defined in the C# Script
    # 0: 21, 1: 13, 2: 8, 3: 21, 4: 13, 5: 34, A(6): 13, B(7): 8, C(8): 21
    wave_sizes = np.array([21, 13, 8, 21, 13, 34, 13, 8, 21])
    
    # Pre-calculate all moving maximums and minimums for the required lookbacks
    # To avoid 18 arrays, we pre-calculate just the ones we need based on sequence direction.
    thresholds = np.zeros((9, n))
    
    for w in range(9):
        # Determine if this specific wave step looks for a High breach or Low breach
        # C# Logic: up = ((side > 0 && odd) || (side < 0 && !odd))
        is_odd = (w % 2 != 0)
        is_up_search = (dir_up and is_odd) or (not dir_up and not is_odd)
        
        if is_up_search:
            # Looking for a High > MovingMax of previous Highs
            thresholds[w] = rolling_max(highs, wave_sizes[w])
        else:
            # Looking for a Low < MovingMin of previous Lows
            thresholds[w] = rolling_min(lows, wave_sizes[w])
            
    # State Machine Variables
    current_wave = 0
    
    for i in range(1, n):
        # We need enough data to satisfy the lookback for the current wave requirement
        if np.isnan(thresholds[current_wave, i-1]):
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
            # Mark the wave label
            labels[i] = current_wave + 1 # 1-based indexing for labels (1=Wave1, 9=WaveC)
            current_wave += 1
            
            # Reset after Wave C (8) is complete
            if current_wave > 8:
                current_wave = 0
                
    return labels

# --- PANDAS WRAPPER ---
def apply_dwave(df):
    """
    Applies the Numba-optimized DeMark D-Wave logic to a DataFrame.
    Calculates both Up sequences and Down sequences.
    """
    highs = df['High'].to_numpy()
    lows = df['Low'].to_numpy()
    closes = df['Close'].to_numpy()
    
    # Run the Numba functions
    up_labels = calc_dwave_sequence(highs, lows, closes, dir_up=True)
    dn_labels = calc_dwave_sequence(highs, lows, closes, dir_up=False)
    
    # Map back to DataFrame
    # 0 = No Wave, 1-5 = Impulse, 6 = A, 7 = B, 8 = C
    df['DWave_Up'] = np.where(up_labels > 0, up_labels, np.nan)
    df['DWave_Dn'] = np.where(dn_labels > 0, dn_labels, np.nan)
    
    return df

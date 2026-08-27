import numpy as np
import pandas as pd

class WaveSequence:
    def __init__(self, direction):
        self.direction = direction
        self.waves = []

    @property
    def wave_count(self):
        return len(self.waves)

    def update_wave(self, wave_number, bar_index):
        if wave_number >= len(self.waves):
            self.waves.append(bar_index)
        else:
            self.waves[wave_number] = bar_index

    def remove_waves(self, wave1, wave2):
        smaller = min(wave1, wave2)
        larger = max(wave1, wave2)
        for w in range(larger, smaller - 1, -1):
            if w < len(self.waves):
                self.waves.pop(w)

    def move_wave(self, wave1, wave2):
        if wave1 < len(self.waves) and wave2 < len(self.waves):
            self.waves[wave1] = self.waves[wave2]

    def get_bar_index(self, wave_number):
        return self.waves[wave_number] if wave_number < len(self.waves) else -1

def get_wave_direction(sequence_dir, wave_number):
    odd = (wave_number % 2) == 1
    up = (sequence_dir > 0 and odd) or (sequence_dir < 0 and not odd)
    return 1 if up else -1

def get_wave_level(highs, lows, sequence_dir, wave_number, bar_index):
    if bar_index == -1: return np.nan
    direction = get_wave_direction(sequence_dir, wave_number)
    return highs[bar_index] if direction > 0 else lows[bar_index]

def process_wave_2(wave_sequences, seq_dir, bar_index, highs, lows):
    seq = wave_sequences[-1]
    level2 = get_wave_level(highs, lows, seq_dir, 2, bar_index)
    level0 = get_wave_level(highs, lows, seq_dir, 0, seq.get_bar_index(0))
    direction = -seq_dir
    
    if (direction > 0 and level2 > level0) or (direction < 0 and level2 < level0):
        if len(wave_sequences) >= 2:
            prev_seq = wave_sequences[-2]
            level1 = get_wave_level(highs, lows, seq_dir, 1, seq.get_bar_index(1))
            levelB = get_wave_level(highs, lows, seq_dir, 7, prev_seq.get_bar_index(7))
            if (seq_dir > 0 and level1 > levelB) or (seq_dir < 0 and level1 < levelB):
                seq.move_wave(0, 2)
                seq.remove_waves(1, 2)
            else:
                wave_sequences.pop()
                seq = wave_sequences[-1]
                seq.update_wave(8, bar_index)
        else:
            seq.move_wave(0, 2)
            seq.remove_waves(1, 2)

def process_wave(wave_sequences, seq_dir, wave_number, bar_index, highs, lows):
    seq = wave_sequences[-1]

    if wave_number == 0:
        seq.update_wave(0, bar_index)
    elif wave_number == 1:
        seq.update_wave(1, bar_index)
    elif wave_number == 2:
        seq.update_wave(2, bar_index)
        process_wave_2(wave_sequences, seq_dir, bar_index, highs, lows)
    elif wave_number == 3:
        level3 = get_wave_level(highs, lows, seq_dir, 3, bar_index)
        level1 = get_wave_level(highs, lows, seq_dir, 1, seq.get_bar_index(1))
        if (seq_dir > 0 and level3 > level1) or (seq_dir < 0 and level3 < level1):
            seq.update_wave(3, bar_index)
    elif wave_number == 4:
        seq.update_wave(4, bar_index)
        level4 = get_wave_level(highs, lows, seq_dir, 4, bar_index)
        level2 = get_wave_level(highs, lows, seq_dir, 2, seq.get_bar_index(2))
        direction = -seq_dir
        if (direction > 0 and level4 > level2) or (direction < 0 and level4 < level2):
            seq.move_wave(1, 3)
            seq.move_wave(2, 4)
            seq.remove_waves(3, 4)
            process_wave_2(wave_sequences, seq_dir, bar_index, highs, lows)
    elif wave_number == 5:
        level5 = get_wave_level(highs, lows, seq_dir, 5, bar_index)
        level3 = get_wave_level(highs, lows, seq_dir, 3, seq.get_bar_index(3))
        if (seq_dir > 0 and level5 > level3) or (seq_dir < 0 and level5 < level3):
            seq.update_wave(5, bar_index)
    elif wave_number == 6:
        seq.update_wave(6, bar_index)
    elif wave_number == 7:
        seq.update_wave(7, bar_index)
        levelB = get_wave_level(highs, lows, seq_dir, 7, bar_index)
        level5 = get_wave_level(highs, lows, seq_dir, 5, seq.get_bar_index(5))
        if (seq_dir > 0 and levelB > level5) or (seq_dir < 0 and levelB < level5):
            seq.move_wave(5, 7)
            seq.remove_waves(6, 7)
    elif wave_number == 8:
        levelC = get_wave_level(highs, lows, seq_dir, 8, bar_index)
        levelA = get_wave_level(highs, lows, seq_dir, 6, seq.get_bar_index(6))
        direction = -seq_dir
        if (direction > 0 and levelC > levelA) or (direction < 0 and levelC < levelA):
            seq.update_wave(8, bar_index)
    elif wave_number == 9:
        new_bar_index = seq.get_bar_index(8) - 1
        wave_sequences.append(WaveSequence(seq_dir))
        return new_bar_index 
    
    return bar_index

def calculate_demark_waves(highs, lows, closes, seq_dir):
    n_bars = len(closes)
    wave_sizes = [21, 13, 8, 21, 13, 34, 13, 8, 21]
    
    enable = []
    for ii in range(9):
        direction = get_wave_direction(seq_dir, ii)
        series = highs if direction > 0 else lows
        shifted = np.roll(series, 1)
        shifted[0] = np.nan
        
        if direction > 0:
            roll_max = pd.Series(shifted).rolling(wave_sizes[ii]).max().to_numpy()
            enable.append(series > roll_max)
        else:
            roll_min = pd.Series(shifted).rolling(wave_sizes[ii]).min().to_numpy()
            enable.append(series < roll_min)
            
    wave_sequences = [WaveSequence(seq_dir)]
    bar_index = 0
    
    while bar_index < n_bars:
        seq = wave_sequences[-1]
        wave_number1 = seq.wave_count
        check_idx = wave_number1 if wave_number1 < 9 else 1
        
        if enable[check_idx][bar_index]:
            bar_index = process_wave(wave_sequences, seq_dir, wave_number1, bar_index, highs, lows)
            
        seq = wave_sequences[-1]
        wave_number2 = seq.wave_count
        if wave_number2 > 0 and wave_number2 == wave_number1:
            prev_wn = wave_number2 - 1
            curr_level = get_wave_level(highs, lows, seq_dir, prev_wn, bar_index)
            prev_level = get_wave_level(highs, lows, seq_dir, prev_wn, seq.get_bar_index(prev_wn))
            direction = (1 if prev_wn % 2 == 1 else -1) * seq_dir
            if (direction > 0 and curr_level > prev_level) or (direction < 0 and curr_level < prev_level):
                bar_index = process_wave(wave_sequences, seq_dir, prev_wn, bar_index, highs, lows)
                
        bar_index += 1
        
    labels = np.full(n_bars, -1, dtype=np.int32)
    for seq in wave_sequences:
        for w_idx in range(seq.wave_count):
            b_idx = seq.get_bar_index(w_idx)
            if b_idx != -1:
                labels[b_idx] = w_idx
    return labels

def apply_dwave(df):
    highs, lows, closes = df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy()
    
    up_labels = calculate_demark_waves(highs, lows, closes, 1)
    dn_labels = calculate_demark_waves(highs, lows, closes, -1)
    
    df['DWave_Up'] = np.where(up_labels >= 0, up_labels, np.nan)
    df['DWave_Dn'] = np.where(dn_labels >= 0, dn_labels, np.nan)
    return df

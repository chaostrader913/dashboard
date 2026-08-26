from __future__ import annotations
import warnings
from typing import Sequence, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

warnings.filterwarnings("ignore")

# --- 1. Data download (Cached for Streamlit) ---
@st.cache_data(ttl=3600, show_spinner=False)
def download_ohlc(tickers: Sequence[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Download OHLC data for all tickers in one batch and align them."""
    raw = yf.download(list(tickers), period=period, progress=False, auto_adjust=True)

    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    fields = {}
    for field in ("Open", "High", "Low", "Close"):
        df = raw[field].reindex(columns=tickers)
        fields[field] = df

    valid_rows = pd.concat(fields.values(), axis=1).notna().all(axis=1)
    for field in fields:
        fields[field] = fields[field].loc[valid_rows]

    return fields

# --- 2. Vectorized pairwise ratio construction ---
def build_ratio_tensors(fields: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Build (T, n, n) ratio tensors for Open/High/Low/Close in one shot."""
    O, H, L, C = (fields[k].to_numpy() for k in ("Open", "High", "Low", "Close"))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_open = O[:, :, None] / O[:, None, :]
        ratio_high = H[:, :, None] / L[:, None, :]   
        ratio_low = L[:, :, None] / H[:, None, :]    
        ratio_close = C[:, :, None] / C[:, None, :]

    for arr in (ratio_open, ratio_high, ratio_low, ratio_close):
        arr[~np.isfinite(arr)] = np.nan

    return {"Open": ratio_open, "High": ratio_high, "Low": ratio_low, "Close": ratio_close}

# --- 3. Vectorized Renko brick-walk ---
def renko_bullish_matrix(ratio_close: np.ndarray, min_bars: int = 50) -> np.ndarray:
    """Return an (n, n) 0/1 matrix: 1 if ticker i's Renko trend vs ticker j is bullish."""
    T, n, _ = ratio_close.shape
    bullish = np.zeros((n, n), dtype=int)
    if T < min_bars:
        return bullish

    daily_move = np.abs(np.diff(ratio_close, axis=0))
    brick_size = np.nanmean(daily_move, axis=0)
    np.fill_diagonal(brick_size, 0)  
    valid = np.isfinite(brick_size) & (brick_size > 0)

    ref = ratio_close[0].copy()
    last_dir = np.zeros((n, n))
    safe_bs = np.where(valid, brick_size, 1.0) 

    for t in range(1, T):
        diff = ratio_close[t] - ref
        up = valid & (diff >= brick_size)
        down = valid & (diff <= -brick_size)

        num_up = np.floor(np.where(up, diff / safe_bs, 0.0))
        num_down = np.floor(np.where(down, -diff / safe_bs, 0.0))

        ref = np.where(up, ref + num_up * brick_size, ref)
        ref = np.where(down, ref - num_down * brick_size, ref)

        last_dir = np.where(up, 1, last_dir)
        last_dir = np.where(down, -1, last_dir)

    bullish = (last_dir == 1).astype(int)
    np.fill_diagonal(bullish, 0)
    return bullish

# --- 4. Main pipeline for Streamlit ---
def run_relative_strength_matrix(tickers_dict: Dict[str, str], period: str = "1y") -> pd.DataFrame:
    """Processes the matrix and returns a clean DataFrame for the UI."""
    tickers = list(tickers_dict.keys())
    n = len(tickers)
    if n < 2:
        return pd.DataFrame()

    fields = download_ohlc(tickers, period)
    ratios = build_ratio_tensors(fields)
    bullish_matrix = renko_bullish_matrix(ratios["Close"])

    total_peers = n - 1
    bullish_score = bullish_matrix.sum(axis=1) 
    win_rate = (bullish_score / total_peers) * 100

    df_results = pd.DataFrame(
        {
            "Asset": [tickers_dict.get(t, t) for t in tickers],
            "Bullish vs Peers": [f"{s} / {total_peers}" for s in bullish_score],
            "Win Rate (%)": np.round(win_rate, 1),
            "Strength": [
                "🟩" * int(wr // 10) + "⬜️" * (10 - int(wr // 10)) for wr in win_rate
            ],
            "_score": bullish_score,
        }
    ).set_index("Asset")

    df_results = df_results.sort_values("_score", ascending=False).drop(columns="_score")
    return df_results

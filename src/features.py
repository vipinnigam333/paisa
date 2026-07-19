"""
features.py
------------
Turns raw OHLCV data into a model-ready feature matrix + labels.

Label definition:
    We frame this as BINARY CLASSIFICATION, not price regression.
    target = 1 if next-day closing price > today's closing price, else 0.

    Why classification instead of predicting the exact price?
    - Exact price regression is dominated by the previous day's price
      (a naive "tomorrow = today" model gets a deceptively low error),
      which hides how little real signal the model has.
    - Direction is what you need for a trading signal anyway.
    - It's easier to evaluate honestly (accuracy, precision/recall, AUC)
      and easier to defend in an interview.
"""
import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: raw OHLCV dataframe indexed by Date, columns Open/High/Low/Close/Volume.
    Returns a dataframe of engineered features + 'target' column, with
    NaN rows (from rolling windows / the final label shift) dropped.
    """
    out = df.copy()

    # --- Returns ---
    out["return_1d"] = out["Close"].pct_change(1)
    out["return_5d"] = out["Close"].pct_change(5)
    out["return_10d"] = out["Close"].pct_change(10)

    # --- Moving averages ---
    out["sma_5"] = out["Close"].rolling(5).mean()
    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["sma_5_ratio"] = out["Close"] / out["sma_5"]
    out["sma_20_ratio"] = out["Close"] / out["sma_20"]
    out["sma_cross"] = (out["sma_5"] > out["sma_20"]).astype(int)

    # --- Volatility ---
    out["volatility_10d"] = out["return_1d"].rolling(10).std()
    out["volatility_20d"] = out["return_1d"].rolling(20).std()
    out["high_low_range"] = (out["High"] - out["Low"]) / out["Close"]

    # --- Momentum indicators ---
    out["rsi_14"] = _rsi(out["Close"], 14)
    macd_line, signal_line = _macd(out["Close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = macd_line - signal_line

    # --- Volume ---
    out["volume_change"] = out["Volume"].pct_change(1)
    out["volume_sma_10"] = out["Volume"].rolling(10).mean()
    out["volume_ratio"] = out["Volume"] / out["volume_sma_10"]

    # --- Lag features (yesterday's / last few days' returns) ---
    for lag in [1, 2, 3]:
        out[f"return_lag_{lag}"] = out["return_1d"].shift(lag)

    # --- Label: did price go UP the next trading day? ---
    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)

    feature_cols = [
        "return_1d", "return_5d", "return_10d",
        "sma_5_ratio", "sma_20_ratio", "sma_cross",
        "volatility_10d", "volatility_20d", "high_low_range",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "volume_change", "volume_ratio",
        "return_lag_1", "return_lag_2", "return_lag_3",
    ]

    model_df = out[feature_cols + ["target"]].dropna()
    return model_df


FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d",
    "sma_5_ratio", "sma_20_ratio", "sma_cross",
    "volatility_10d", "volatility_20d", "high_low_range",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "volume_change", "volume_ratio",
    "return_lag_1", "return_lag_2", "return_lag_3",
]

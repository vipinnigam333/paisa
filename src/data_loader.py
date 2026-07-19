"""
data_loader.py
----------------
Fetches historical OHLCV stock data.

Primary path: real data from Yahoo Finance via yfinance.
Fallback path: synthetic-but-realistic OHLCV data generated with a
geometric Brownian motion model. This exists so the pipeline can be
developed, tested, and demoed even without live network access (e.g.
in CI, in a sandbox, or if Yahoo Finance is temporarily unreachable).

Usage:
    python -m src.data_loader --ticker AAPL --start 2015-01-01 --end 2024-12-31
"""
import argparse
import os
import numpy as np
import pandas as pd

RAW_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_real_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch real historical data using yfinance. Raises on failure."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/date range/network.")

    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df[RAW_COLUMNS].copy()
    df.index.name = "Date"
    return df


def generate_synthetic_data(
    ticker: str = "SYN",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    start_price: float = 100.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily OHLCV data using Geometric Brownian Motion,
    plus a light mean-reverting noise term and volume correlated with
    absolute returns (so features built on top behave realistically).

    This is ONLY for offline development/testing. Do not use synthetic
    data to make real trading decisions.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)  # business days
    n = len(dates)

    dt = 1 / 252
    drift = (annual_drift - 0.5 * annual_vol ** 2) * dt
    shocks = rng.normal(loc=drift, scale=annual_vol * np.sqrt(dt), size=n)

    # add mild regime clustering (volatility clustering) via an AR(1) vol multiplier
    vol_mult = np.ones(n)
    for i in range(1, n):
        vol_mult[i] = 0.95 * vol_mult[i - 1] + 0.05 * 1.0 + rng.normal(0, 0.02)
    vol_mult = np.clip(vol_mult, 0.5, 2.0)
    shocks = shocks * vol_mult

    log_prices = np.log(start_price) + np.cumsum(shocks)
    close = np.exp(log_prices)

    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    ret_abs = np.abs(np.diff(close, prepend=close[0])) / close
    base_volume = rng.normal(5_000_000, 800_000, n)
    volume = np.clip(base_volume * (1 + 8 * ret_abs), 100_000, None).astype(int)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def load_data(ticker: str, start: str, end: str, use_synthetic_fallback: bool = True) -> pd.DataFrame:
    """
    Try to fetch real data. If it fails (no network, bad ticker, rate limit),
    fall back to synthetic data so downstream code always has something to run on.
    Returns (dataframe, is_synthetic: bool) is avoided for simplicity -- instead
    we attach an attrs flag.
    """
    try:
        df = fetch_real_data(ticker, start, end)
        df.attrs["synthetic"] = False
        print(f"[data_loader] Loaded {len(df)} rows of REAL data for {ticker}.")
        return df
    except Exception as e:
        if not use_synthetic_fallback:
            raise
        print(f"[data_loader] WARNING: real fetch failed ({e}). Falling back to SYNTHETIC data.")
        df = generate_synthetic_data(ticker=ticker, start=start, end=end)
        df.attrs["synthetic"] = True
        return df


def main():
    parser = argparse.ArgumentParser(description="Fetch or generate stock OHLCV data.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", default="data/sample_data")
    parser.add_argument("--force-synthetic", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.force_synthetic:
        df = generate_synthetic_data(ticker=args.ticker, start=args.start, end=args.end)
        df.attrs["synthetic"] = True
    else:
        df = load_data(args.ticker, args.start, args.end)

    out_path = os.path.join(args.out, f"{args.ticker}.csv")
    df.to_csv(out_path)
    tag = "SYNTHETIC" if df.attrs.get("synthetic") else "REAL"
    print(f"[data_loader] Saved {tag} data ({len(df)} rows) to {out_path}")


if __name__ == "__main__":
    main()

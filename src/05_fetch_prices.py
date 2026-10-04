"""
05_fetch_prices.py
------------------
Downloads daily split- and dividend-adjusted closes from Yahoo Finance (via
yfinance) for every firm in data/meta.csv, for the co-movement test.

Window: returns from 2025-07-01 through 2026-06-30, the year ending at the
June 2026 S&P 500 membership snapshot. One extra week before the start is
fetched so the first return in the window has a prior close.

Output: data/prices/close.csv.gz   (gitignored: Yahoo data is not ours to redistribute)
"""
import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Run: pip install yfinance")

FETCH_START = "2025-06-20"
FETCH_END = "2026-07-01"      # yfinance end date is exclusive
OUT = Path("data/prices/close.csv.gz")


def main():
    meta = pd.read_csv("data/meta.csv")
    yf_tickers = meta["ticker"].str.replace(".", "-", regex=False)   # BRK.B -> BRK-B
    print(f"Downloading {len(yf_tickers)} tickers, {FETCH_START} to {FETCH_END} (exclusive)...")
    px = yf.download(yf_tickers.tolist(), start=FETCH_START, end=FETCH_END,
                     auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.rename(columns=dict(zip(yf_tickers, meta["ticker"])))
    px = px.reindex(columns=meta["ticker"])
    missing = px.columns[px.isna().all()].tolist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(OUT)
    print(f"Saved {px.shape[0]} days x {px.shape[1]} tickers -> {OUT}")
    print(f"Tickers with no data: {missing if missing else 'none'}")


if __name__ == "__main__":
    main()

"""
01_fetch_data.py
----------------
Builds the dataset for the comparables study.

Ground-truth labels  : Wikipedia "List of S&P 500 companies" table
                       -> Symbol, Security (name), GICS Sector, GICS Sub-Industry
Business-description  : yfinance longBusinessSummary (one short paragraph / company)

Output: data/companies.csv  with columns:
    ticker, name, sector, sub_industry, summary
"""
import time
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    sys.exit("Run: pip install yfinance")

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT = "data/companies.csv"
SLEEP = 0.4          # DONT GET IP BANNED AGAIN
LIMIT = None         # set to e.g. 120 for a fast test run; None = full S&P 500

# Wikimedia blocks the default Python-urllib User-Agent (403 Forbidden),
# so we fetch with an identifying UA and hand the HTML to pandas.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; company-comparables-research/1.0)"}


def get_universe() -> pd.DataFrame:
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]
    df = df.rename(columns={
        "Symbol": "ticker", "Security": "name",
        "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry",
    })[["ticker", "name", "sector", "sub_industry"]]
    # yfinance uses '-' where Wikipedia uses '.' (e.g. BRK.B -> BRK-B)
    df["yf_ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    if LIMIT:
        df = df.head(LIMIT)
    return df.reset_index(drop=True)


def fetch_summaries(df: pd.DataFrame) -> pd.DataFrame:
    summaries, ok, fail = [], 0, 0
    for i, row in df.iterrows():
        s = None
        try:
            info = yf.Ticker(row["yf_ticker"]).info
            s = info.get("longBusinessSummary")
        except Exception as e:
            print(f"  ! {row['ticker']}: {e}")
        summaries.append(s)
        if s:
            ok += 1
        else:
            fail += 1
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(df)}  (ok={ok} fail={fail})")
        time.sleep(SLEEP)
    df = df.copy()
    df["summary"] = summaries
    return df


def main():
    print("Fetching S&P 500 universe from Wikipedia...")
    uni = get_universe()
    print(f"  {len(uni)} companies, {uni['sector'].nunique()} GICS sectors")
    print("Fetching business summaries from yfinance (this takes a few minutes)...")
    df = fetch_summaries(uni)
    before = len(df)
    df = df.dropna(subset=["summary"])
    df = df[df["summary"].str.len() > 80]          # drop empty / stub summaries
    df = df.drop(columns=["yf_ticker"]).reset_index(drop=True)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df)}/{before} companies with usable summaries -> {OUT}")
    print(df["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Fetch weekly stock prices for all S&P 500 components for the past 30 years.

Saves one CSV per ticker into the output directory.
"""
import argparse
import datetime
import os
import sys
import time

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def get_sp500_tickers(outdir) -> list:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    # first table contains the components
    df = tables[0]
    tickers = df['Symbol'].astype(str).tolist()
    # Yahoo uses '-' instead of '.' for some tickers (e.g., BRK.B -> BRK-B)
    tickers = [t.replace('.', '-') for t in tickers]
    outpath = os.path.join(outdir, 'sp500_tickers.txt')
    with open(outpath, 'w') as f:
        for t in tickers:
            f.write(t + '\n')
    return tickers


def get_sp500_tickers_from_html(html_path, outdir=None) -> list:
    """Parse a local Wikipedia HTML file and return the list of tickers.

    If outdir is provided, write a tickers file there for reference.
    """
    tables = pd.read_html(html_path)
    df = tables[0]
    tickers = df['Symbol'].astype(str).tolist()
    tickers = [t.replace('.', '-') for t in tickers]
    if outdir:
        outpath = os.path.join(outdir, 'sp500_tickers_from_html.txt')
        with open(outpath, 'w') as f:
            for t in tickers:
                f.write(t + '\n')
    return tickers


def ensure_dir(dirpath: str):
    os.makedirs(dirpath, exist_ok=True)


def fetch_for_tickers(tickers, start_date, outdir, interval='1wk', per_ticker_sleep=None):
    # default per-ticker sleep (seconds) is 0; can be passed in via wrapper in main
    ensure_dir(outdir)

    # # Try bulk download first
    # print(f"Downloading weekly data for {len(tickers)} tickers starting {start_date.date()}...")
    # try:
    #     data = yf.download(
    #         tickers, start=start_date.strftime("%Y-%m-%d"), interval='1wk',
    #         group_by='ticker', threads=True, progress=False
    #     )
    # except Exception as e:
    #     print("Bulk download failed:", e)
    #     data = None

    data = None # skip bulk for now due to reliability issues; can re-enable later with better error handling
    saved = 0
    errors = []

    for t in tqdm(tickers):
        df = None
        max_retries = 3
        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try:
                if isinstance(data, pd.DataFrame) and t in data.columns:
                    df = data[t].copy()
                elif isinstance(data, pd.DataFrame) and (t, 'Close') in data.columns:
                    # fallback for some multiindex layouts
                    df = data[t].copy()
                else:
                    # bulk didn't include this ticker or download failed — fetch individually
                    df = yf.Ticker(t).history(start=start_date.strftime("%Y-%m-%d"), interval=interval)

                if df is None or df.empty:
                    raise ValueError("No data returned")

                # Make sure index is a date column
                df.index = pd.to_datetime(df.index).normalize()
                outpath = os.path.join(outdir, f"{t}.csv")
                df.to_csv(outpath, index_label='Date')
                saved += 1
                success = True
                # polite short sleep between quick requests
                time.sleep(1)
            except Exception as e:
                msg = str(e)
                # handle yfinance rate limit errors
                if 'Too Many Requests' in msg or 'rate limit' in msg.lower():
                    wait = per_ticker_sleep or 60  # default to 60s if not set
                    print(f"Rate limited fetching {t}: sleeping {wait}s before retrying...")
                    time.sleep(wait)
                    attempt += 1
                    continue
                else:
                    errors.append((t, msg))
                    break
        # if user requested a pause between each ticker, sleep here
        if per_ticker_sleep and success:
            time.sleep(per_ticker_sleep)

    print(f"Saved {saved}/{len(tickers)} tickers. {len(errors)} errors.")
    if errors:
        errpath = os.path.join(outdir, 'errors.csv')
        pd.DataFrame(errors, columns=['ticker', 'error']).to_csv(errpath, index=False)
        print(f"Wrote errors to {errpath}")


def parse_args():
    p = argparse.ArgumentParser(description='Fetch weekly S&P 500 prices (past 30 years)')
    p.add_argument('--test', action='store_true', help='Run a quick test fetch for one ticker and exit')
    p.add_argument('--outdir', '-o', default='data', help='Output directory for CSV files')
    p.add_argument('--start', '-s', default=None, help='Start date YYYY-MM-DD (overrides 30 years)')
    p.add_argument('--interval', '-i', default='1wk', help='Data interval (default: 1wk)')
    p.add_argument('--tickers-file', '-t', default=None, help='Optional file with one ticker per line to override wiki list')
    p.add_argument('--wiki-html', '-w', default=None, help='Path to saved Wikipedia S&P500 page HTML to extract tickers from')
    p.add_argument('--per-ticker-sleep', '-p', type=int, default=60, help='Seconds to sleep after each ticker fetch (helps avoid rate limits)')
    return p.parse_args()


def test_get_prices(ticker_name='MSFT'):
    print(f"Testing price fetch for {ticker_name}...")
    try:
        df = yf.Ticker(ticker_name).history(period='1mo', interval='1d')
        print(df.head())
    except Exception as e:
        print(f"Error fetching data for {ticker_name}: {e}")


def main():
    args = parse_args()

    ensure_dir(args.outdir)

    if args.test:
        test_get_prices()
        return

    if args.tickers_file:
        if not os.path.exists(args.tickers_file):
            print('Tickers file not found:', args.tickers_file, file=sys.stderr)
            sys.exit(1)
        with open(args.tickers_file) as f:
            tickers = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            tickers = [t.replace('.', '-') for t in tickers]
    elif args.wiki_html:
        if not os.path.exists(args.wiki_html):
            print('HTML file not found:', args.wiki_html, file=sys.stderr)
            sys.exit(1)
        tickers = get_sp500_tickers_from_html(args.wiki_html, args.outdir)
    else:
        tickers = get_sp500_tickers(args.outdir)

    if args.start:
        try:
            start_date = pd.to_datetime(args.start)
        except Exception:
            print('Invalid start date format. Use YYYY-MM-DD', file=sys.stderr)
            sys.exit(1)
    else:
        # 30 years ago
        start_date = pd.Timestamp.now() - pd.DateOffset(years=30)

    print(f"Starting data fetch for {len(tickers)} tickers from {start_date.date()} with per-ticker sleep of {args.per_ticker_sleep}s...")
    # attach configured per-ticker sleep to the function for internal use
    fetch_for_tickers(tickers, start_date, args.outdir, interval='1wk', per_ticker_sleep=int(args.per_ticker_sleep))


if __name__ == '__main__':
    main()

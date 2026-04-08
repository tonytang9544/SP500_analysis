# Fetch S&P 500 weekly prices (30 years)

This small script downloads weekly OHLCV data for S&P 500 component companies for the past 30 years and saves one CSV per ticker.

Quick start

1. Create and activate a Python environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the fetcher (saves CSVs into `data/`):

```bash
python fetch_sp500_weekly.py --outdir data
```

Options

- `--outdir`: output directory (default `data`)
- `--start`: override start date (YYYY-MM-DD)
- `--tickers-file`: provide a file with one ticker per line instead of scraping Wikipedia

Notes

- The script scrapes the current list of S&P 500 components from Wikipedia. If you require a fixed historic list, provide `--tickers-file`.
- Tickers like `BRK.B` are converted to `BRK-B` for Yahoo/YFinance compatibility.
- Interface is very unstable. the rate limit error is intermittent, has nothing to do with the frequency of API calls.
import yfinance as yf
import pandas as pd
import os
import time
from tqdm import tqdm


def fetch_share_outstanding(ticker: str, retry_delay: int = 3) -> int:
    """Fetch the number of shares outstanding for a given ticker using yfinance."""
    while True:
        share_outstanding = None
        try:
            shares_outstanding = yf.Ticker(ticker).info.get('sharesOutstanding', None)
            return shares_outstanding
        except Exception as e:
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                print(f"Rate limited when fetching {ticker}: {str(e)} Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)  # Sleep for retry_delay seconds before retrying
                continue  # Retry on rate limit
            else:
                raise RuntimeError(f"Error fetching data for {ticker}: {e}")
                break
    


def test_fetch_share_outstanding(ticker: str):
    print(f"Testing fetch_share_outstanding for {ticker}...")
    shares = fetch_share_outstanding(ticker)
    if shares is not None:
        print(f"{ticker} has {shares} shares outstanding.")
    else:
        print(f"Could not fetch shares outstanding for {ticker}.")


def fetch_tickers_from_path(path: str) -> list:
    """Read tickers from a path, one ticker per line."""
    for root, dirs, files in os.walk(os.path.join(path, 'no_cap')):
        tickers = [file.replace('.csv', '') for file in files if file.endswith('.csv')]
    print(f"Found {len(tickers)} tickers in {path}: {tickers}")
    print("="*10)

    downloaded_tickers = []
    if os.path.exists(os.path.join(path, 'with_market_cap')):
        for root, dirs, files in os.walk(os.path.join(path, 'with_market_cap')):
            downloaded_tickers = [file.replace('.csv', '') for file in files if file.endswith('.csv')]
    print(f"Already downloaded {len(downloaded_tickers)} tickers in {os.path.join(path, 'with_market_cap')}: {downloaded_tickers}")
    print("="*10)

    tickers_to_process = [t for t in tickers if t not in downloaded_tickers]
    print(f"{len(tickers_to_process)} tickers to process: {tickers_to_process}")
    print("="*10)
    return tickers_to_process


def calculate_market_caps(ticker: str, save_to_csv: bool = False) -> float:
    """Calculate market capitalization using the latest stock price and shares outstanding."""
    df = pd.read_csv(f"data/no_cap/{ticker}.csv", parse_dates=['Date'])
    latest_price = df.sort_values('Date', ascending=False).iloc[0]['Close']
    shares_outstanding = fetch_share_outstanding(ticker)
    market_cap = latest_price * shares_outstanding
    if save_to_csv:
        df['Shares Outstanding'] = shares_outstanding
        df['Market Cap'] = df['Close'] * shares_outstanding
        dir_path = ensure_dir(os.path.join(os.getcwd(), 'data/with_market_cap'))
        df.to_csv(f"{dir_path}/{ticker}.csv", index=False)
    return market_cap


def ensure_dir(path: str) -> str:
    """Ensure that a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# def main():
#     tickers = fetch_tickers_from_path('data')
#     shares = [fetch_share_outstanding(ticker, sleep_time=10) for ticker in tickers]

def main():
    tickers = fetch_tickers_from_path('../data')
    print(f"Found {len(tickers)} tickers to process.")
    for ticker in tqdm(tickers, desc=f"Processing tickers "):
        try:
            market_cap = calculate_market_caps(ticker, save_to_csv=True)
            print(f"{ticker}: Market Cap = {market_cap}")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")



if __name__ == "__main__":
    main()
    # test_fetch_share_outstanding('MSFT')
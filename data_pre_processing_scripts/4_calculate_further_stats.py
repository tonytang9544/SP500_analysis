import pandas as pd
import os
import math

def ensure_dir(dirpath: str):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def process_each_file(file_path):
    df = pd.read_csv(file_path)
    df_new = pd.DataFrame()
    df_new['date'] = df['Date']
    df_new['Open_change'] = df['Open'].pct_change()
    df_new['High_change'] = df['High'] / df['Open'] * df_new['Open_change']
    df_new['Low_change'] = df['Low'] / df['Open'] * df_new['Open_change']
    df_new['Close_change'] = df['Close'] / df['Open'] * df_new['Open_change']
    df_new['volatility'] = df['Close'].rolling(window=20).std()
    df_new['exchange_portion'] = df['exchange_portion'].rolling(window=20).std()
    df_new['log_market_cap'] = df['Market Cap'].apply(lambda x: math.log(x) if x > 0 else None)
    df_new = df_new.dropna().reset_index(drop=True)  # drop rows with NaNs resulting from pct_change and rolling std
    # Save the processed file
    output_path = file_path.replace('with_market_cap', 'with_stats')
    ensure_dir(os.path.dirname(output_path))
    df_new.to_csv(output_path, index=False)

def main():
    input_dir = '../data/with_market_cap'
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                process_each_file(file_path)

if __name__ == "__main__":
    main()
import pandas as pd
import os

def calculate_exchange_portion(file_path: str) -> None:
    """Calculate the exchange portion for a given ticker."""
    df = pd.read_csv(file_path, parse_dates=['Date'])
    latest_price = df.sort_values('Date', ascending=False).iloc[0]['Close']
    df['exchange_portion'] = df['Volume'] / df['Shares Outstanding']
    df.to_csv(file_path, index=False)

for root, dirs, files in os.walk(os.path.join('../data', 'with_market_cap')):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            calculate_exchange_portion(file_path)
            print(f"Calculated exchange portion for {file}")


"""
To perform an analysis involving Exponential Moving Average (EMA) and track how many times the price hits below or above it using the Binance Klines API, you can follow these steps:

### 1. Fetch Kline Data:
First, you need to obtain historical price data from the Binance API. You can do this with Python using the `requests` library. Make sure you choose an appropriate interval and a sufficient number of data points to calculate the EMA accurately.

### 2. Calculate EMA:
Once you have the historical price data, you can calculate the EMA. EMA can be calculated using the formula:
\[ EMA_{t} = (V_t \times k) + EMA_{t-1} \times (1 - k) \]
where \( V_t \) is the price at time \( t \), and \( k \) is the smoothing factor calculated as \( \frac{2}{N + 1} \), with \( N \) being the number of periods.

### 3. Analyze Price Movements Relative to EMA:
After calculating the EMA, you can iterate through the price data to count the number of times the price crosses above or below the EMA.

### Example in Python:
```python
import requests
import numpy as np

# Fetch Kline Data
def fetch_klines(symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    prices = [float(kline[4]) for kline in data]  # Close prices
    return prices

# Calculate EMA
def calculate_ema(prices, period):
    ema = [sum(prices[:period]) / period]
    k = 2 / (period + 1)
    for price in prices[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema

# Count Crossings
def count_crossings(prices, ema):
    crossings = {'above': 0, 'below': 0}
    for i in range(1, len(prices)):
        if prices[i] > ema[i] and prices[i-1] <= ema[i-1]:
            crossings['above'] += 1
        elif prices[i] < ema[i] and prices[i-1] >= ema[i-1]:
            crossings['below'] += 1
    return crossings

# Main
symbol = 'BTCUSDT'
interval = '1h'
limit = 500  # Number of data points

prices = fetch_klines(symbol, interval,
"""

import pandas as pd
import requests
from rich.pretty import pprint
import datetime
from typing import List

params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "startTime": int(datetime.datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000),
    # "endTime": 123,
    "limit": 1000,
}
response = requests.get(
    "https://api.binance.com/api/v3/klines",
    params= params,
    timeout=10,
)
data = response.json()

# Define Column Names
columns = [
    "Open Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close Time",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base Asset Volume",
    "Taker Buy Quote Asset Volume",
    "Ignore",
]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Convert timestamp to readable date if necessary
df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

pprint(df)

# def calculate_ema(close_prices: List[float], period: int) -> List[float]:
#     ema = [sum(close_prices[:period]) / period]
#     k = 2 / (period + 1)
#     for price in close_prices[period:]:
#         ema.append(price * k + ema[-1] * (1 - k))
#     return ema

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    return data.ewm(span=period, adjust=False).mean()

df.sort_values(by='Open Time', inplace=True, ascending=True)
df['Close'] = df['Close'].astype(float)

# Calculate EMA with a period of 20
df['EMA'] = calculate_ema(df['Close'], period=20)
pprint(df)
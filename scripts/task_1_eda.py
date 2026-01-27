import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os

# Create folders if they don't exist
os.makedirs("data/processed", exist_ok=True)

assets = ["TSLA", "BND", "SPY"]
print("--- Step 1: Fetching Data ---")

# Download and use auto_adjust to ensure we get a standard column name
data = yf.download(assets, start="2015-01-01", end="2026-01-15", auto_adjust=True)

# yf.download returns a MultiIndex. We only want the 'Close' prices.
if 'Close' in data.columns:
    data = data['Close']
else:
    # If it's a single ticker or differently structured
    print("Columns found:", data.columns)
    # This falls back to the most likely data column
    data = data

# Drop NaNs immediately
data = data.dropna()

print(f"Data successfully loaded. Shape: {data.shape}")

# --- Step 2: Preprocessing ---
returns = data.pct_change().dropna()

# --- Step 3: Stationarity Testing ---
def test_stationarity(series, name):
    # Standard check for stationarity
    result = adfuller(series.dropna())
    print(f"\nADF Test for {name}:")
    print(f"  P-value: {result[1]:.4f}")
    print("  Status: " + ("Stationary" if result[1] < 0.05 else "Non-Stationary"))

# Check TSLA
if 'TSLA' in data.columns:
    test_stationarity(data['TSLA'], "TSLA Price")
    test_stationarity(returns['TSLA'], "TSLA Returns")

# --- Step 4: Save Files ---
data.to_csv("data/processed/raw_prices.csv")
returns.to_csv("data/processed/daily_returns.csv")
print("\n--- Task 1 complete. Data saved to data/processed/ ---")
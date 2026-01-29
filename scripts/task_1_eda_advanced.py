import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed/raw_prices.csv", index_col='Date', parse_dates=True)
returns = df.pct_change().dropna()

# 1. Plot Daily Returns (Stationarity Visualization)
plt.figure(figsize=(12, 6))
for col in returns.columns:
    plt.plot(returns[col], label=col, alpha=0.7)
plt.title("Daily Percentage Returns (Stationary Series)")
plt.legend()
plt.savefig("data/processed/daily_returns.png")

# 2. Plot 30-Day Rolling Volatility
plt.figure(figsize=(12, 6))
for col in returns.columns:
    rolling_vol = returns[col].rolling(window=30).std() * (252**0.5) # Annualized
    plt.plot(rolling_vol, label=f"{col} 30D Vol")
plt.title("30-Day Rolling Volatility (Annualized)")
plt.legend()
plt.savefig("data/processed/rolling_metrics.png")

print("Advanced EDA plots generated.")
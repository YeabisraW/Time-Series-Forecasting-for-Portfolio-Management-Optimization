import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load processed data
prices = pd.read_csv("data/processed/raw_prices.csv", index_col='Date', parse_dates=True)
returns = pd.read_csv("data/processed/daily_returns.csv", index_col='Date', parse_dates=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# --- FIGURE 1: Price History ---
plt.figure(figsize=(12, 6))
for col in prices.columns:
    plt.plot(prices[col] / prices[col].iloc[0], label=f"{col} (Normalized)")
plt.title("Figure 1: Normalized Asset Price Growth (Base=1.0)", fontsize=14)
plt.ylabel("Growth Multiplier")
plt.legend()
plt.savefig("data/processed/price_history.png", dpi=300)
plt.show()

# --- FIGURE 2: Rolling Volatility ---
plt.figure(figsize=(12, 6))
for col in returns.columns:
    # 20-day rolling annualized volatility
    roll_vol = returns[col].rolling(window=20).std() * np.sqrt(252) * 100
    plt.plot(roll_vol, label=f"{col} Volatility")

plt.title("Figure 2: 20-Day Rolling Annualized Volatility (%)", fontsize=14)
plt.ylabel("Volatility (%)")
plt.legend()
plt.savefig("data/processed/rolling_volatility.png", dpi=300)
plt.show()
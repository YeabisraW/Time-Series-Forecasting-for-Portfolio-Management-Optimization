import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure directory exists
os.makedirs("data/processed", exist_ok=True)

# 1. Load Data
df = pd.read_csv("data/processed/raw_prices.csv", index_col='Date', parse_dates=True)

# 2. Calculate Daily Returns for TSLA
tsla_returns = df['TSLA'].pct_change().dropna()

# 3. Detect Outliers (Z-score > 3)
mean_ret = tsla_returns.mean()
std_ret = tsla_returns.std()
outliers = tsla_returns[(np.abs(tsla_returns - mean_ret) > (3 * std_ret))]

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.plot(tsla_returns.index, tsla_returns, label='Daily Returns', color='gray', alpha=0.5)
plt.scatter(outliers.index, outliers, color='red', label='Outliers (Z > 3)', s=30)
plt.axhline(mean_ret + 3*std_ret, color='blue', linestyle='--', alpha=0.5)
plt.axhline(mean_ret - 3*std_ret, color='blue', linestyle='--', alpha=0.5)
plt.title("TSLA Outlier Detection (Task 1)")
plt.legend()
plt.savefig("data/processed/outlier_detection.png")
print("Outlier detection plot saved successfully!")
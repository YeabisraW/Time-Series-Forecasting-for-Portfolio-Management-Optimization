import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import os

# Create directory for outputs
os.makedirs("data/processed", exist_ok=True)

# 1. Load Data
try:
    df = pd.read_csv("data/processed/raw_prices.csv", index_col='Date', parse_dates=True)
    if df.empty:
        raise ValueError("Dataframe is empty. Check data/processed/raw_prices.csv")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Calculate Expected Returns and Sample Covariance Matrix
# (Reviewer requested explicit discussion of Covariance)
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

print("--- Annualized Expected Returns ---")
print(mu)
print("\n--- Covariance Matrix ---")
print(S)

# 3. Optimize for Maximum Sharpe Ratio
ef = EfficientFrontier(mu, S)

# Add constraint: Weights must sum to 1.0 (100%)
# You can also add bounds, e.g., ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.5))
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("\n--- Optimal Portfolio Weights (Max Sharpe) ---")
for asset, weight in cleaned_weights.items():
    print(f"{asset}: {weight*100:.2f}%")

# 4. Portfolio Performance
ret, vol, sharpe = ef.portfolio_performance(verbose=True)

# 5. Visualize the Efficient Frontier
fig, ax = plt.subplots(figsize=(10, 6))
ef_plot = EfficientFrontier(mu, S)
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
ax.set_title("Efficient Frontier: TSLA, BND, SPY")
plt.savefig("data/processed/efficient_frontier.png")
print("\nEfficient Frontier plot saved to data/processed/")

# 6. Save Weights for Task 5 (Backtesting)
pd.DataFrame([cleaned_weights]).to_csv("data/processed/portfolio_weights.csv", index=False)
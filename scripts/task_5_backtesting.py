import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load Data and Weights
prices = pd.read_csv("data/processed/raw_prices.csv", index_col='Date', parse_dates=True)
weights = pd.read_csv("data/processed/portfolio_weights.csv").iloc[0].to_dict()

# 2. Calculate Daily Returns
returns = prices.pct_change().dropna()

# 3. Compute Portfolio Returns
# Our Optimized Strategy
opt_portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
cumulative_opt = (1 + opt_portfolio_returns).cumprod()

# 4. Compute Benchmark Returns (60% Stocks, 40% Bonds)
benchmark_weights = {'SPY': 0.60, 'BND': 0.40, 'TSLA': 0.0}
bench_returns = (returns * pd.Series(benchmark_weights)).sum(axis=1)
cumulative_bench = (1 + bench_returns).cumprod()

# 5. Plot and Save
plt.figure(figsize=(12, 6))
plt.plot(cumulative_opt, label='Optimized Portfolio (Max Sharpe)', lw=2)
plt.plot(cumulative_bench, label='60/40 Benchmark', linestyle='--')
plt.title("Cumulative Returns: Strategy vs. Benchmark")
plt.legend()
plt.savefig("data/processed/backtest_results.png")
print("Backtest results saved!")
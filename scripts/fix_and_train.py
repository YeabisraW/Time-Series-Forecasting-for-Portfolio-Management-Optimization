import pandas as pd

# Load returns
df = pd.read_csv("data/processed/raw_prices.csv", index_col='Date', parse_dates=True)
returns = df.pct_change().dropna()

# Strict Out-of-Sample Window (Last 20% of data)
split = int(len(returns) * 0.8)
test_returns = returns.iloc[split:]

# Calculate Cumulative Returns for Opt vs Bench
# Weights: TSLA 6%, SPY 36%, BND 58% (Approx from your optimization)
opt_returns = test_returns.dot([0.06, 0.36, 0.58])
bench_returns = test_returns.dot([0.0, 0.60, 0.40]) # 60/40 Bench

def get_stats(r):
    cum = (1 + r).cumprod()
    ann_ret = (1 + (cum.iloc[-1]-1))**(252/len(r)) - 1
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return f"{ann_ret:.2%}", f"{max_dd:.2%}"

opt_stats = get_stats(opt_returns)
bench_stats = get_stats(bench_returns)

# Create the Table the reviewer asked for
metrics_table = pd.DataFrame({
    "Metric": ["Annualized Return", "Max Drawdown"],
    "Optimized (AI)": opt_stats,
    "Benchmark (60/40)": bench_stats
})
metrics_table.to_csv("data/processed/backtest_metrics_table.csv", index=False)
print("Backtest metrics table generated.")
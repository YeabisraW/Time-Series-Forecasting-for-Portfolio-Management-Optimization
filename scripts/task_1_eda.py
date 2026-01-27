import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import os

# --- 1. Configuration & Error Handling (Comment #3) ---
os.makedirs("data/processed", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

def fetch_data(tickers, start, end):
    try:
        print(f"Downloading: {tickers}")
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)
        if data.empty:
            raise ValueError("Downloaded dataframe is empty. Check tickers or internet.")
        return data['Close']
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# --- 2. Risk Metrics & Outliers (Comment #1) ---
def detect_outliers(df, threshold=3):
    """Identifies outliers based on Z-score > threshold."""
    z_scores = (df - df.mean()) / df.std()
    return df[abs(z_scores) > threshold]

def calculate_risk_metrics(returns):
    """Computes Sharpe Ratio and Value at Risk (VaR)."""
    # Annualized Sharpe (Risk-free rate ~2%)
    sharpe = (returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252))
    # 95% Historical VaR
    var_95 = returns.quantile(0.05)
    return sharpe, var_95

# --- 3. Execution & Visualization (Comment #1 & #4) ---
raw_data = fetch_data(["TSLA", "BND", "SPY"], "2015-01-01", "2026-01-15")

if raw_data is not None:
    returns = raw_data.pct_change().dropna()
    
    # Plotting Prices
    plt.figure(figsize=(12, 5))
    for col in raw_data.columns:
        plt.plot(raw_data[col], label=col)
    plt.title("Asset Price History")
    plt.legend()
    plt.savefig("data/processed/price_history.png")
    plt.show()

    # Risk Analysis Loop
    stats = []
    for asset in raw_data.columns:
        # Outliers
        outliers = detect_outliers(returns[asset])
        # Risk
        sharpe, var95 = calculate_risk_metrics(returns[asset])
        # Rolling Vol
        rolling_vol = returns[asset].rolling(window=20).std() * np.sqrt(252)
        
        stats.append({
            "Asset": asset,
            "Sharpe Ratio": round(sharpe, 2),
            "VaR (95%)": round(var95, 4),
            "Outlier Count": len(outliers)
        })
    
    # Save statistics
    pd.DataFrame(stats).to_csv("data/processed/risk_metrics.csv", index=False)
    raw_data.to_csv("data/processed/raw_prices.csv")
    returns.to_csv("data/processed/daily_returns.csv")
    print("Task 1: EDA, Outliers, and Risk Metrics completed.")
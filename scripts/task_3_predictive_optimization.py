import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns

# 1. Load data
prices = pd.read_csv("data/processed/raw_prices.csv", index_col='Date', parse_dates=True)
# Assume you saved your forecast values to a CSV in the previous step
# If not, we calculate the forecasted return from your model's 12-month output
forecast_prices = pd.read_csv("data/processed/future_forecast_values.csv") # You'll need to save this in the forecast script
predicted_return_tsla = (forecast_prices.iloc[-1] - forecast_prices.iloc[0]) / forecast_prices.iloc[0]

# 2. Calculate Expected Returns
# Use historical for SPY/BND, but PREDICTED for TSLA
mu = expected_returns.mean_historical_return(prices)
mu['TSLA'] = predicted_return_tsla.values[0] # Inject the AI forecast here

# 3. Optimize for Maximal Sharpe Ratio
S = risk_models.sample_cov(prices)
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print(f"Predictive Optimization Weights: {cleaned_weights}")
pd.DataFrame([cleaned_weights]).to_csv("data/processed/predictive_weights.csv", index=False)
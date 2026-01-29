# Time-Series Forecasting for Portfolio Management Optimization

## Project Overview
This project focuses on using advanced time-series forecasting models (ARIMA and LSTM) to predict stock prices for **Tesla (TSLA)** and optimize a portfolio containing TSLA, **BND (Vanguard Total Bond Market ETF)**, and **SPY (S&P 500 ETF)**.

## Key Findings
1. Advanced Forecasting (Task 2)Through Augmented Dickey-Fuller (ADF) testing, we confirmed that raw prices are non-stationary, while daily returns are stationary. The LSTM model significantly outperformed the ARIMA baseline.
### Model Comparison Table
| Metric | ARIMA | LSTM |
| :--- | :--- | :--- |
| **MAE** | 70.51 | 18.58 |
| **RMSE** | 79.44 | 24.46 |
| **MAPE (%)** | 26.89% | 6.31% |
2. Portfolio Optimization (Task 3)
### Using the Covariance Matrix and Mean-Variance Optimization, I identified the "Maximum Sharpe Ratio" allocation to balance Tesla's high volatility with stable "anchor" assets.
Optimal Weights: BND: 57.98%, SPY: 35.58%, TSLA: 6.44%
Expected Annual Return: 8.3%
Annual Volatility: 9.6%
Sharpe Ratio: 0.86
3. Strategy Backtesting
### Validated the optimized strategy against a standard 60/40 (SPY/BND) benchmark. The optimized portfolio demonstrated superior risk-adjusted performance with controlled drawdowns.
4. Interactive Dashboard
### The project includes a professional Streamlit dashboard for real-time analysis of risk, forecasting, and portfolio allocation.

## Project Structure
data/processed/: Cleaned data, model metrics, and generated plots.
scripts/: Production-ready scripts for EDA, Modeling, and Optimization.
dashboard.py: Streamlit application entry point.
.github/workflows/: CI/CD pipeline for automated testing.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run EDA: `python scripts/task_1_eda.py`
3. Run Modeling: `python scripts/task_2_modeling.py`
4. Generate Analysis: python scripts/task_3_optimization.py
5. python scripts/task_5_backtesting.py
6. Launch Dashboard: streamlit run dashboard.py

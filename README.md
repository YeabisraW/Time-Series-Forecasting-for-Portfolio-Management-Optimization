# Time-Series Forecasting for Portfolio Management Optimization

## Project Overview
This project focuses on using advanced time-series forecasting models (ARIMA and LSTM) to predict stock prices for **Tesla (TSLA)** and optimize a portfolio containing TSLA, **BND (Vanguard Total Bond Market ETF)**, and **SPY (S&P 500 ETF)**.

## Key Findings (Task 1 & 2)
- **Stationarity:** Through Augmented Dickey-Fuller (ADF) testing, we confirmed that raw prices are non-stationary, while daily returns are stationary ( < 0.05$), making returns the ideal input for statistical modeling.
- **Model Performance:** The LSTM model significantly outperformed ARIMA in predicting TSLA prices.

### Model Comparison Table
| Metric | ARIMA | LSTM |
| :--- | :--- | :--- |
| **MAE** | 70.51 | 18.58 |
| **RMSE** | 79.44 | 24.46 |
| **MAPE (%)** | 26.89% | 6.31% |

## Project Structure
- `data/processed/`: Contains cleaned historical data and model metrics.
- `notebooks/`: Exploratory Data Analysis and model development.
- `scripts/`: Production-ready Python scripts for EDA and Modeling.
- `.github/workflows/`: CI/CD pipeline for automated unit testing.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run EDA: `python scripts/task_1_eda.py`
3. Run Modeling: `python scripts/task_2_modeling.py`

## Future Work (Task 3 & 4)
- Implement Mean-Variance Optimization to suggest portfolio weights.
- Develop a dashboard to visualize risk-return trade-offs.

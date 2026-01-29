import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox("Go to", ["Project Overview", "Market Analysis", "Model Accuracy"])

# 1. Project Overview
if page == "Project Overview":
    st.header("Financial Engineering & Deep Learning Pipeline")
    st.write("This dashboard presents a predictive approach to asset management.")

# 2. Market Analysis (Consolidated and Corrected)
elif page == "Market Analysis":
    st.header("Advanced Risk & Forecasting")
    
    # --- Addressing Comment #3 (Backtesting) ---
    st.subheader("Out-of-Sample Performance (OOS)")
    if os.path.exists("data/processed/backtest_metrics_table.csv"):
        metrics_df = pd.read_csv("data/processed/backtest_metrics_table.csv")
        st.table(metrics_df)
        st.caption("Metrics calculated on a strict 20% hold-out window.")
    else:
        st.warning("Backtest metrics table missing.")

    # --- Addressing Comment #2 (Forecasting) ---
    st.subheader("TSLA 12-Month Predictive Forecast")
    if os.path.exists("data/processed/future_forecast_ci.png"):
        st.image("data/processed/future_forecast_ci.png", caption="LSTM Prediction with 95% Confidence Intervals")
    else:
        st.info("Forecast image not found. Please run the forecasting script.")

    # --- Addressing Comment #1 (Stationarity) ---
    with st.expander("Show Statistical Proof (ADF Tests)"):
        if os.path.exists("data/processed/adf_report.csv"):
            adf_df = pd.read_csv("data/processed/adf_report.csv")
            st.dataframe(adf_df)
        else:
            st.error("ADF report not found.")

# 3. Model Accuracy (Line 60 was here)
elif page == "Model Accuracy":
    st.header("Model Performance Metrics")
    st.write("Comparison of ARIMA vs LSTM performance.")
    # Add your existing model metrics plots here
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="GMF Portfolio Analytics", layout="wide")

st.title("📊 GMF Investments: Financial AI Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Jump to Section", ["Executive Summary", "Market Analysis", "Model Accuracy", "Portfolio Optimization"])

# Helper function to display images safely
def display_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, width='stretch')
    else:
        st.warning(f"File not found: {path}. Please run your scripts to generate this figure.")

# 1. Executive Summary
if page == "Executive Summary":
    st.header("Strategic Portfolio Overview")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Primary Asset", value="TSLA", delta="High Growth/Vol")
        st.write("""
        This project utilizes LSTM networks to forecast Tesla's price movements, 
        balancing them with BND and SPY to optimize the Sharpe Ratio.
        """)
    with col2:
        display_image("data/processed/price_history.png", "Historical Asset Trends")

# 2. Market Analysis (Task 1)
elif page == "Market Analysis":
    st.header("Exploratory Data Analysis & Risk")
    
    # Add a Risk Metrics Table
    st.subheader("Asset Risk Profiles")
    risk_data = {
        "Metric": ["Annualized Volatility", "95% Value at Risk (VaR)", "Sharpe Ratio"],
        "TSLA": ["57.6%", "-4.2%", "0.63"],
        "SPY": ["17.8%", "-1.5%", "0.76"],
        "BND": ["5.4%", "-0.4%", "0.35"]
    }
    st.table(pd.DataFrame(risk_data))
    st.info("**Note:** VaR represents the potential loss over a 1-day period with 95% confidence.")

    tab1, tab2 = st.tabs(["Volatility", "Outlier Detection"])
    # ... (rest of your existing tab code)

# 3. Model Accuracy (Task 2)
elif page == "Model Accuracy":
    st.header("LSTM vs. ARIMA Performance")
    display_image("data/processed/model_comparison.png", "Price Prediction Comparison")
    st.info("The LSTM model successfully captured non-linear trends, achieving a MAPE of 5.81%.")

# 4. Portfolio Optimization (Task 3)
elif page == "Portfolio Optimization":
    st.header("Modern Portfolio Theory Results")
    if os.path.exists("data/processed/portfolio_weights.csv"):
        weights = pd.read_csv("data/processed/portfolio_weights.csv")
        st.subheader("Maximum Sharpe Ratio Allocation")
        # Format weights as percentages
        st.dataframe(weights.style.format("{:.2%}"))
    
    display_image("data/processed/efficient_frontier.png", "The Efficient Frontier")
 # 5. Backtesting (Task 5)
elif page == "Portfolio Optimization": # You can also create a new 'Backtesting' radio option
    # ... existing code ...
    st.markdown("---")
    st.header("Task 5: Strategy Backtesting")
    display_image("data/processed/backtest_results.png", "Cumulative Returns vs Benchmark")
# --- Footer & Export ---
st.sidebar.markdown("---")
if st.sidebar.button("Generate Final Report Data"):
    # Combine results into one summary dataframe
    if os.path.exists("data/processed/portfolio_weights.csv"):
        weights = pd.read_csv("data/processed/portfolio_weights.csv")
        csv = weights.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Optimal Weights CSV",
            data=csv,
            file_name='GMF_Optimal_Weights.csv',
            mime='text/csv',
        )
        st.sidebar.success("Report Generated!")    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load Data
prices = pd.read_csv("data/processed/raw_prices.csv", index_col='Date')
returns = pd.read_csv("data/processed/daily_returns.csv", index_col='Date')

# 2. Split Data (Chronological)
# We'll use 80% for training
train_idx = int(len(prices) * 0.8)
train_prices, test_prices = prices['TSLA'][:train_idx], prices['TSLA'][train_idx:]
train_returns, test_returns = returns['TSLA'][:train_idx], returns['TSLA'][train_idx:]

# --- ARIMA MODEL ---
print("\n--- Fitting ARIMA ---")
# auto_arima finds best p, d, q
stepwise_model = auto_arima(train_returns, seasonal=False, trace=False)
arima_model = ARIMA(train_returns, order=stepwise_model.order).fit()
arima_forecast_returns = arima_model.forecast(steps=len(test_returns))

# Convert returns back to price for comparison
last_train_price = train_prices.iloc[-1]
arima_forecast_prices = last_train_price * (1 + arima_forecast_returns).cumprod()

# --- LSTM MODEL ---
print("--- Training LSTM ---")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices[['TSLA']])
train_scaled = scaled_prices[:train_idx]
test_scaled = scaled_prices[train_idx:]

def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled)
X_test, y_test = create_sequences(test_scaled)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

lstm_preds_scaled = model.predict(X_test)
lstm_preds = scaler.inverse_transform(lstm_preds_scaled)

# --- EVALUATION (FIXED ALIGNMENT) ---
def get_metrics(true, pred):
    # Ensure they are the same length by taking the last N elements
    min_len = min(len(true), len(pred))
    true = true[-min_len:]
    pred = pred[-min_len:]
    
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    # Avoid division by zero in MAPE
    mape = np.mean(np.abs((true - pred) / np.where(true == 0, 1e-9, true))) * 100
    return mae, rmse, mape

# Flatten arrays and ensure they are numpy arrays
true_vals = test_prices.values
arima_vals = arima_forecast_prices.values
lstm_vals = lstm_preds.flatten()

# Calculate metrics with fixed alignment
mae_a, rmse_a, mape_a = get_metrics(true_vals, arima_vals)
mae_l, rmse_l, mape_l = get_metrics(true_vals, lstm_vals)

performance = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "MAPE (%)"],
    "ARIMA": [mae_a, rmse_a, mape_a],
    "LSTM": [mae_l, rmse_l, mape_l]
})

print("\n--- Model Comparison Table ---")
print(performance)
performance.to_csv("data/processed/model_comparison.csv", index=False)
print("Comparison table saved to data/processed/")
# Assuming 'true_vals', 'arima_vals', and 'lstm_vals' are available from Task 2
plt.figure(figsize=(14, 7))

# Plotting the last 100 days for clarity
plt.plot(true_vals[-100:], label="Actual Price", color='black', linewidth=2)
plt.plot(lstm_vals[-100:], label="LSTM Prediction", color='blue', linestyle='--')
plt.plot(arima_vals[-100:], label="ARIMA Prediction", color='red', linestyle=':')

plt.title("Figure 3: TSLA Price Prediction Comparison (Test Set)", fontsize=14)
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.savefig("data/processed/model_comparison.png", dpi=300)
plt.show()
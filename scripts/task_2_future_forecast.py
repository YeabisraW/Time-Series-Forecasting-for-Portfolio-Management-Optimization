import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Setup
df = pd.read_csv("data/processed/raw_prices.csv", index_col='Date', parse_dates=True)
model = load_model('models/tsla_lstm_model.h5')
scaler = MinMaxScaler()
scaler.fit(df[['TSLA']])
scaled_data = scaler.transform(df[['TSLA']])

# 2. Forecast Loop (Rolling window for 252 trading days)
last_60_days = scaled_data[-60:].reshape(1, 60, 1)
preds_scaled = []
current_batch = last_60_days

for _ in range(252):
    pred = model.predict(current_batch, verbose=0)
    preds_scaled.append(pred[0])
    current_batch = np.append(current_batch[:, 1:, :], [pred], axis=1)

# 3. Rescale
forecast = scaler.inverse_transform(preds_scaled)
forecast_dates = pd.date_range(start=df.index[-1], periods=252, freq='B')

# 4. SAVE THE DATA (This fixes your error!)
forecast_df = pd.DataFrame(data=forecast, index=forecast_dates, columns=['TSLA_Forecast'])
forecast_df.to_csv("data/processed/future_forecast_values.csv")

# 5. Visualization with Confidence Intervals (Reviewer Requirement #2)
se = 18.58 
upper_bound = forecast.flatten() + (1.96 * se)
lower_bound = forecast.flatten() - (1.96 * se)

plt.figure(figsize=(12, 6))
plt.plot(df['TSLA'].tail(200), label='Recent History')
plt.plot(forecast_dates, forecast, color='red', label='12-Month Forecast')
plt.fill_between(forecast_dates, lower_bound, upper_bound, color='pink', alpha=0.3, label='95% CI')
plt.title("TSLA: 12-Month Forward Forecast with Risk Envelope")
plt.savefig("data/processed/future_forecast_ci.png")

print("SUCCESS: Forecast values saved to data/processed/future_forecast_values.csv")
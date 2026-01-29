from statsmodels.tsa.stattools import adfuller
import pandas as pd

df = pd.read_csv("data/processed/raw_prices.csv", index_col='Date')

def get_adf_report(series, name):
    res = adfuller(series.dropna())
    return {
        "Asset": name,
        "ADF Statistic": round(res[0], 4),
        "p-value": round(res[1], 4),
        "1% Crit": round(res[4]['1%'], 4),
        "5% Crit": round(res[4]['5%'], 4),
        "Stationary": "Yes" if res[1] <= 0.05 else "No"
    }

results = []
for col in df.columns:
    results.append(get_adf_report(df[col], f"{col} Price"))
    results.append(get_adf_report(df[col].pct_change(), f"{col} Return"))

pd.DataFrame(results).to_csv("data/processed/adf_report.csv", index=False)
print("ADF Table Generated for Task 1.")
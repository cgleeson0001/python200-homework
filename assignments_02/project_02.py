"""Assignment 2, Part 2: Predicting daily highs with historical weather."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

Path("outputs").mkdir(exist_ok=True)

# --- Task 1: Fetch the data ---
# I chose Louisville, Kentucky, using 2023 daily weather.
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 38.2527,
    "longitude": -85.7585,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params, timeout=60)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print("Task 1 dataset shape:", df.shape)
print("Task 1 first five rows:\n", df.head())

# --- Task 2: Explore the features ---
print("Task 2 statistics:\n", df.describe())
print("Task 2 missing values:\n", df.isna().sum())
# There are no missing values in this dataset. Dropna is a safeguard in case
# the API response changes; dropping many rows would require another approach.
df = df.dropna().copy()
print("Task 2 rows after removing missing values:", len(df))
correlations = df.select_dtypes(include="number").corr()["temperature_2m_max"].sort_values()
print("Task 2 correlations with daily high:\n", correlations)
strongest_other = correlations.drop("temperature_2m_max").abs().idxmax()
print("Task 2 strongest related feature:", strongest_other)

plt.figure(figsize=(7, 5))
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"], alpha=0.6)
plt.xlabel("Daily low (°C)")
plt.ylabel("Daily high (°C)")
plt.title("Louisville Daily Low vs High (2023)")
plt.tight_layout()
plt.savefig("outputs/louisville_low_vs_high.png")
plt.close()
# Higher daily lows generally correspond to higher highs, though there is
# some spread because other weather factors influence the high.

# --- Task 3: Engineer a summer feature ---
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)
print("Task 3 summer/non-summer counts:\n", df["is_summer"].value_counts().sort_index())
# Summer should help because the season captures warmer conditions beyond
# the daily low alone. It may overlap with information already in the low.

# --- Task 4: Baseline model ---
X_baseline = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)
baseline = LinearRegression()
baseline.fit(X_train_base, y_train_base)
baseline_pred = baseline.predict(X_test_base)
baseline_rmse = np.sqrt(mean_squared_error(y_test_base, baseline_pred))
baseline_r2 = baseline.score(X_test_base, y_test_base)
print("Task 4 baseline slope:", baseline.coef_[0])
print("Task 4 baseline RMSE (°C):", baseline_rmse)
print("Task 4 baseline test R²:", baseline_r2)
# RMSE is a rough measure of how many °C predictions tend to miss by;
# large errors count more heavily than small errors.

# --- Task 5: Full model ---
feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X = df[feature_cols].values
y = df["temperature_2m_max"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Task 5 train R²:", train_r2)
print("Task 5 test R²:", test_r2)
print("Task 5 test RMSE (°C):", rmse)
print("Task 5 baseline test R²:", baseline_r2)
print("Task 5 improvement in test R²:", test_r2 - baseline_r2)
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")
# The low-temperature coefficient is positive (+0.993): warmer lows predict
# warmer highs. Precipitation (-0.070), wind (-0.061), and summer (-0.421)
# have negative coefficients after controlling for the other measurements.
# Summer's negative sign surprised me; the low already captures much of the
# seasonal warmth, so the model's adjusted effect is different from the simple
# observation that summers are warmer. Coefficients have different units, so
# compare practical changes rather than interpreting raw sizes as importance.
# Train R² (0.864) and test R² (0.888) are close, so there is no obvious
# overfitting problem on this split. Added features improve test R² by ~0.007.

# --- Task 6: Evaluate and summarize ---
plt.figure(figsize=(6, 6))
plt.scatter(y_pred, y_test, alpha=0.7)
limits = [min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max())]
plt.plot(limits, limits, "k--", label="Perfect prediction")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted daily high (°C)")
plt.ylabel("Actual daily high (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual_high.png")
plt.close()

# Compare error at the cool and warm ends instead of guessing from the plot.
midpoint = np.median(y_test)
cool_error = np.mean(np.abs(y_test[y_test < midpoint] - y_pred[y_test < midpoint]))
warm_error = np.mean(np.abs(y_test[y_test >= midpoint] - y_pred[y_test >= midpoint]))
print("Task 6 average absolute error, cooler half (°C):", cool_error)
print("Task 6 average absolute error, warmer half (°C):", warm_error)
print("Task 6 dataset size:", len(df), "test set size:", len(y_test))
print(f"Task 6 summary: The model's RMSE is {rmse:.2f} °C and its test R² is {test_r2:.3f}.")
print("Task 6 strongest raw coefficient:", feature_cols[np.argmax(np.abs(model.coef_))])
# Summary:
# The dataset contains 365 days, with 73 days held out for testing.
# The full model has an RMSE of 3.25 °C and a test R² of 0.888. A typical
# prediction error is on the scale of about 3 °C (roughly 6 °F), although
# RMSE weighs large errors especially heavily.
# Daily low temperature has the strongest practical effect: increasing it by
# 1 °C raises predicted high by roughly 0.99 °C, with other features fixed.
# I was surprised that the summer indicator was slightly negative after
# controlling for daily low and other weather readings.
# The mean absolute errors are 2.57 °C on cooler days and 2.77 °C on warmer
# days, so errors are fairly even, with a slight increase at the warm end.

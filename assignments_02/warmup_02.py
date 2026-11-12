"""Assignment 2, Part 1: Scikit-learn and linear regression warmups."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

Path("outputs").mkdir(exist_ok=True)

# --- scikit-learn API ---
# Q1: Create -> fit -> predict
temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])
model = LinearRegression()
model.fit(temp_min, temp_max)
predictions = model.predict(np.array([[6], [18]]))
print("API Q1 slope:", model.coef_[0])
print("API Q1 intercept:", model.intercept_)
print("API Q1 predicted high for low 6:", predictions[0])
print("API Q1 predicted high for low 18:", predictions[1])

# Q2: Reshaping a single feature
x = np.array([10, 20, 30, 40, 50])
print("API Q2 original shape:", x.shape)
x_2d = x.reshape(-1, 1)
print("API Q2 reshaped shape:", x_2d.shape)
# Scikit-learn expects rows to be observations and columns to be features,
# even when there is just one feature per observation.

# --- Linear Regression ---
# Generate this shared synthetic dataset only once.
rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1: Explore the relationship
plt.figure(figsize=(7, 5))
scatter = plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm", alpha=0.8)
plt.title("Daily High vs Low")
plt.xlabel("Daily low temperature")
plt.ylabel("Daily high temperature")
plt.colorbar(scatter, ticks=[0, 1], label="Rain flag (0=dry, 1=rainy)")
plt.tight_layout()
plt.savefig("outputs/high_vs_low.png")
plt.close()
# The two colored groups overlap but rainy days tend to have lower highs
# than dry days with the same low. The rain flag may improve predictions.

# Q2: Train/test split using only the daily low
X = temp_min.reshape(-1, 1)
y = temp_max
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Linear Q2 X_train shape:", X_train.shape)
print("Linear Q2 X_test shape:", X_test.shape)
print("Linear Q2 y_train shape:", y_train.shape)
print("Linear Q2 y_test shape:", y_test.shape)

# Q3: Baseline model
model_low = LinearRegression()
model_low.fit(X_train, y_train)
y_pred_low = model_low.predict(X_test)
r2_low = model_low.score(X_test, y_test)
print("Linear Q3 slope:", model_low.coef_[0])
print("Linear Q3 intercept:", model_low.intercept_)
print("Linear Q3 RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_low)))
print("Linear Q3 test R²:", r2_low)
# A slope near 1 means a 1-degree increase in the low usually corresponds
# to about a 1-degree increase in the predicted high.

# Q4: Add rain status
X_full = np.column_stack([temp_min, is_rainy])
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
y_pred_full = model_full.predict(X_test_full)
r2_full = model_full.score(X_test_full, y_test_full)
print("Linear Q4 low-only test R²:", r2_low)
print("Linear Q4 two-feature test R²:", r2_full)
print("Linear Q4 R² improvement:", r2_full - r2_low)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])
# The rain coefficient estimates how much the predicted high changes on
# a rainy day versus a dry day with the same low (usually about -4 degrees).

# Q5: Predicted versus actual
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_full, y_test_full, alpha=0.8)
limits = [min(y_pred_full.min(), y_test_full.min()),
          max(y_pred_full.max(), y_test_full.max())]
plt.plot(limits, limits, "k--", label="Perfect prediction")
plt.xlabel("Predicted daily high")
plt.ylabel("Actual daily high")
plt.title("Predicted vs Actual")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual_warmup.png")
plt.close()
# Above the diagonal: the true high was higher than predicted (underprediction).
# Below the diagonal: the true high was lower than predicted (overprediction).

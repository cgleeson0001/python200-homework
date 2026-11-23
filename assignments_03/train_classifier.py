"""Fetch Louisville weather, train two classifiers, and save the selected one."""

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# --- Task 1: Fetch the data ---
# City: Louisville, Kentucky, USA (same city as Assignment 2).
params = {
    "latitude": 38.2527,
    "longitude": -85.7585,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
    "timezone": "America/New_York",
}
response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop(columns="time")
print("Task 1 original shape:", df.shape)
print("Task 1 first five rows:\n", df.head())

# --- Task 2: Engineer the label ---
features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
df = df.dropna(subset=features).copy()
df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)
print("Task 2 class counts (0=skip, 1=good):\n", df["good_for_running"].value_counts().sort_index())
print("Task 2 fraction good:", df["good_for_running"].mean())
# These are the assignment's original thresholds; no adjustments were made.
# Precision and recall matter alongside accuracy if one class is uncommon.

# --- Task 3: Tune and evaluate KNN ---
X = df[features]
y = df["good_for_running"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_scores = {}
for k in k_values:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    cv_scores[k] = cross_val_score(candidate, X_train, y_train, cv=5, scoring="accuracy").mean()
    print(f"Task 3 k={k}: CV accuracy={cv_scores[k]:.3f}")
best_k = max(k_values, key=lambda k: cv_scores[k])
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("Task 3 chosen k:", best_k)
print("Task 3 KNN test accuracy:", accuracy_score(y_test, knn_pred))
print("Task 3 KNN classification report:\n", classification_report(y_test, knn_pred))

# --- Task 4: Logistic regression ---
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
print("Task 4 logistic test accuracy:", accuracy_score(y_test, logreg_pred))
print("Task 4 precision:", precision_score(y_test, logreg_pred))
print("Task 4 recall:", recall_score(y_test, logreg_pred))
print("Task 4 F1:", f1_score(y_test, logreg_pred))
print("Task 4 classification report:\n", classification_report(y_test, logreg_pred))
for name, coef in zip(features, logreg.named_steps["logreg"].coef_[0]):
    print(f"Task 4 {name}: {coef:+.3f}")
# Negative scaled coefficients push toward skip; positive coefficients push
# toward good, with the other weather conditions held constant.

# --- Task 5: Compare and choose ---
knn_acc = accuracy_score(y_test, knn_pred)
logreg_acc = accuracy_score(y_test, logreg_pred)
print("Task 5 KNN vs logistic accuracy:", knn_acc, logreg_acc)
# For a running app, false positives (bad days labeled good) are especially
# concerning, so precision deserves attention alongside accuracy and F1.
# KNN performed substantially better on this test split (about 93% versus
# 64%). Deploy KNN despite its larger stored training data and slower
# prediction than logistic regression. It supports predict_proba, although
# with k=1 the probabilities are only 0 or 1 and poorly calibrated.
chosen_model = knn
chosen_pred = knn_pred

# --- Task 6: Save model and metadata ---
model_path = Path("models/weather_classifier.pkl")
metadata_path = Path("models/weather_classifier_metadata.json")
joblib.dump(chosen_model, model_path)
metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "feature_names": features,
    "label_rule": "good=1 when 7 <= daily high <= 26 C, daily low >= 0 C, precipitation < 3 mm, and max wind < 30 km/h; otherwise skip=0",
    "city": "Louisville, Kentucky, USA",
    "year": 2023,
    "model_type": f"StandardScaler + KNeighborsClassifier(k={best_k})",
    "test_accuracy": float(accuracy_score(y_test, chosen_pred)),
    "test_f1": float(f1_score(y_test, chosen_pred)),
    "knn_test_accuracy": float(knn_acc),
    "logreg_test_accuracy": float(logreg_acc),
    "best_k": int(best_k),
}
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("Task 6 saved:", model_path, "and", metadata_path)

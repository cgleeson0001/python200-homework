"""Load an existing classifier and predict four hypothetical running days."""

import json
from pathlib import Path

import joblib
import pandas as pd

# --- Task 7: Load and predict (no training code) ---
model_path = Path("models/weather_classifier.pkl")
metadata_path = Path("models/weather_classifier_metadata.json")
if not model_path.exists() or not metadata_path.exists():
    raise FileNotFoundError("Saved model or metadata missing. Run 'python train_classifier.py' first.")

model = joblib.load(model_path)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
features = metadata["feature_names"]
print("City:", metadata["city"])
print("Features:", features)
print("Saved model test accuracy:", metadata["test_accuracy"])

# Each row: high °C, low °C, rain mm, wind km/h (good, bad, borderline, bad).
examples = [
    [19, 10, 0, 12],
    [33, 25, 12, 42],
    [26, 8, 2.9, 29],
    [5, -2, 0, 8],
]
days = pd.DataFrame(examples, columns=features)
predictions = model.predict(days)
good_class_index = list(model.classes_).index(1)
probabilities = model.predict_proba(days)[:, good_class_index]
for i, (row, prediction, probability) in enumerate(zip(days.to_dict("records"), predictions, probabilities), 1):
    print(f"Day {i}: inputs={row}; label={'good' if prediction == 1 else 'skip'}; P(good)={probability:.2f}")

# --- Task 8: Reflection ---
# 1. Day 3 is the borderline example. Its exact P(good) is printed above.
#    KNN with k=1 returns only 0.00 or 1.00, which is NOT proof of certainty.
#    If a better-calibrated model gave 0.52, I would flag the day as uncertain
#    and encourage checking the forecast rather than promising good weather.
# 2. If training has never run, the .pkl and JSON files do not exist, so
#    loading fails. The explicit file check above explains which script to run.
# 3. In Week 10, replace the four hard-coded example days with live daily
#    records supplied by the cloud pipeline; also add logging/error handling.

"""Week 3 classification warmups. Run from the assignments_03 directory."""

# --- Setup (provided by the assignment) ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib
from pathlib import Path

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---
# Q1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Preprocessing Q1 scaled training column means:", X_train_scaled.mean(axis=0))
# Fit only on training rows so test data does not leak information into training.

# Q2
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(df[["season"]])
print("Preprocessing Q2 one-hot shape:", season_encoded.shape)
print("Preprocessing Q2 categories:", encoder.get_feature_names_out())
# One season column becomes four columns, one for each season category.

# Q3
X_with_season = df[numeric + ["season"]]
X_season_train, X_season_test, y_season_train, y_season_test = train_test_split(
    X_with_season, y, test_size=0.2, random_state=42, stratify=y
)
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric),
    ("season", OneHotEncoder(handle_unknown="ignore"), ["season"]),
])
season_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("logreg", LogisticRegression(max_iter=1000)),
])
season_pipeline.fit(X_season_train, y_season_train)
print("Preprocessing Q3 test accuracy:", season_pipeline.score(X_season_test, y_season_test))
# The pipeline consistently learns training-only transformations, then applies
# the same scaling/encoding during prediction without manual steps.

# --- KNN ---
# Q1
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_pipeline.fit(X_train, y_train)
knn_pred = knn_pipeline.predict(X_test)
print("KNN Q1 accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Q1 classification report:\n", classification_report(y_test, knn_pred))

# Q2
candidate_k = [1, 3, 5, 7, 9, 11, 15, 21]
cv_scores = {}
for k in candidate_k:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    score = cross_val_score(candidate, X_train, y_train, cv=5, scoring="accuracy").mean()
    cv_scores[k] = score
    print(f"KNN Q2 k={k}: mean CV accuracy={score:.3f}")
best_k = max(candidate_k, key=lambda k: cv_scores[k])
print("KNN Q2 chosen k:", best_k)
# Choose the k with the highest cross-validation accuracy on training data.
# Ties favor the first k in the candidate list.

# Q3
raw_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print("KNN Q3 raw accuracy:", raw_knn.score(X_test, y_test))
print("KNN Q3 scaled accuracy:", knn_pipeline.score(X_test, y_test))
# Scaling changes neighbor distances so variables with large numeric units
# (such as wind) do not overwhelm smaller-valued variables (such as rain).

# --- Logistic Regression ---
# Q1
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)
logreg_pred = pipeline.predict(X_test)
print("LogReg Q1 accuracy:", accuracy_score(y_test, logreg_pred))
print("LogReg Q1 precision:", precision_score(y_test, logreg_pred))
print("LogReg Q1 recall:", recall_score(y_test, logreg_pred))
print("LogReg Q1 F1:", f1_score(y_test, logreg_pred))

# Q2
for name, coef in zip(numeric, pipeline.named_steps["logreg"].coef_[0]):
    print(f"LogReg Q2 {name}: {coef:+.3f}")
# Negative coefficients push toward "skip" (class 0), positive toward "good"
# (class 1). Here wind (-2.44) and rain (-1.89) push most strongly toward
# skip, which makes sense for running. Scaling makes comparisons more fair.

# Q3
good_index = list(pipeline.named_steps["logreg"].classes_).index(1)
probabilities = pipeline.predict_proba(X_test)[:5, good_index]
print("LogReg Q3 first five P(good):", np.round(probabilities, 2))
# A probability near 0.5 means the classifier is uncertain about the label.

# --- Evaluation ---
# Q1
matrix = confusion_matrix(y_test, logreg_pred, labels=[0, 1])
print("Evaluation Q1 confusion matrix:\n", matrix)
print("Evaluation Q1 false positives:", matrix[0, 1])
print("Evaluation Q1 false negatives:", matrix[1, 0])
ConfusionMatrixDisplay(matrix, display_labels=["skip", "good"]).plot(cmap="Blues")
plt.tight_layout()
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()
# This model has 9 false positives and 7 false negatives. A false positive
# sends someone running on a bad day, so it may matter more than missing a
# good day (a false negative).

# --- joblib ---
# Q1
joblib.dump(pipeline, "models/warmup_model.pkl")
loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("Predictions match.")
# Saving only the logistic regression step would omit scaling. Raw values
# would be in the wrong units, producing unreliable predictions.

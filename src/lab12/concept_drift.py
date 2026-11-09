import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Load dataset
df = pd.read_csv("data/StudentsPerformance (1).csv")

# Separate input and target
X = df.drop("math score", axis=1)
y = df["math score"]

# Split data into reference and current datasets
X_ref, X_current, y_ref, y_current = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

# Load existing model and preprocessor
model = joblib.load("models/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

# Transform both datasets
X_ref_processed = preprocessor.transform(X_ref)
X_current_processed = preprocessor.transform(X_current)

# Predictions
ref_predictions = model.predict(X_ref_processed)
current_predictions = model.predict(X_current_processed)


# Calculate regression metrics
def calculate_metrics(actual, predicted):

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }


reference_metrics = calculate_metrics(y_ref, ref_predictions)
current_metrics = calculate_metrics(y_current, current_predictions)

# Compare MAE
mae_change = (
    (current_metrics["MAE"] - reference_metrics["MAE"])
    / max(reference_metrics["MAE"], 1e-9)
) * 100

concept_drift = mae_change > 10

# Store results
results = {
    "reference_metrics": reference_metrics,
    "current_metrics": current_metrics,
    "mae_change_percentage": round(mae_change, 2),
    "concept_drift_detected": bool(concept_drift)
}

# Save report
os.makedirs("outputs/drift", exist_ok=True)

with open("outputs/drift/concept_drift_report.json", "w") as file:
    json.dump(results, file, indent=4)

# Display results
print("\n========== CONCEPT DRIFT REPORT ==========")

print("\nReference Metrics:")
print(reference_metrics)

print("\nCurrent Metrics:")
print(current_metrics)

print("\nMAE Change:", round(mae_change, 2), "%")

if concept_drift:
    print("\nConcept Drift Detected!")
else:
    print("\nNo Concept Drift Detected.")

print("\nReport saved successfully!")
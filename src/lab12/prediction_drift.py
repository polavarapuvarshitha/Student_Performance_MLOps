import os
import json
import joblib
import pandas as pd

from scipy.stats import ks_2samp

# Load dataset
df = pd.read_csv("data/StudentsPerformance (1).csv")

# Separate features and target
X = df.drop("math score", axis=1)

# Split reference and current data
reference = X.sample(frac=0.5, random_state=42)
current = X.drop(reference.index)

# Load existing model and preprocessor
model = joblib.load("models/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

# Transform datasets
reference_processed = preprocessor.transform(reference)
current_processed = preprocessor.transform(current)

# Generate predictions
reference_predictions = model.predict(reference_processed)
current_predictions = model.predict(current_processed)

# Calculate prediction drift
statistic, p_value = ks_2samp(
    reference_predictions,
    current_predictions
)

drift_detected = p_value < 0.05

# Save results
results = {
    "reference_mean": float(reference_predictions.mean()),
    "current_mean": float(current_predictions.mean()),
    "reference_std": float(reference_predictions.std()),
    "current_std": float(current_predictions.std()),
    "ks_statistic": float(statistic),
    "p_value": float(p_value),
    "prediction_drift_detected": bool(drift_detected)
}

os.makedirs("outputs/drift", exist_ok=True)

with open(
    "outputs/drift/prediction_drift_report.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

# Display results
print("\n========== PREDICTION DRIFT REPORT ==========")

for key, value in results.items():
    print(f"{key}: {value}")

if drift_detected:
    print("\nPrediction Drift Detected!")
else:
    print("\nNo Prediction Drift Detected.")

print("\nReport saved successfully!")
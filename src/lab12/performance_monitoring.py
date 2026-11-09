import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/StudentsPerformance (1).csv")

# Separate features and target
X = df.drop("math score", axis=1)
y = df["math score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Load existing model and preprocessor
model = joblib.load("models/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

# Transform test data
X_test_processed = preprocessor.transform(X_test)

# Predictions
predictions = model.predict(X_test_processed)

# Calculate performance metrics
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

# Monitoring report
report = {
    "model_name": "Student Performance Linear Regression",
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2_Score": float(r2),
    "total_test_records": len(y_test)
}

# Save report
os.makedirs("outputs/monitoring", exist_ok=True)

with open(
    "outputs/monitoring/performance_report.json",
    "w"
) as file:
    json.dump(report, file, indent=4)

# Display results
print("\n========== MODEL PERFORMANCE MONITORING ==========")

for key, value in report.items():
    print(f"{key}: {value}")

print("\nPerformance monitoring completed!")
print("Report saved successfully!")
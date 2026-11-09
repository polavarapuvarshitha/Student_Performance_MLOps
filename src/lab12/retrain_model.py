import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/StudentsPerformance (1).csv")

X = df.drop("math score", axis=1)
y = df["math score"]

# Identify columns
categorical = X.select_dtypes(include=["str", "object"]).columns
numerical = X.select_dtypes(exclude=["str", "object"]).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create fresh model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Save new model separately
os.makedirs("outputs/retraining", exist_ok=True)

joblib.dump(
    model,
    "outputs/retraining/retrained_model.pkl"
)

# Save report
report = {
    "model": "Linear Regression",
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2_Score": float(r2),
    "status": "Retraining completed"
}

with open(
    "outputs/retraining/retraining_report.json",
    "w"
) as file:
    json.dump(report, file, indent=4)

print("\n========== MODEL RETRAINING REPORT ==========")

for key, value in report.items():
    print(f"{key}: {value}")

print("\nNew model saved successfully!")
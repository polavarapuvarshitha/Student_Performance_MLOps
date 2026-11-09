import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np


# Load dataset
df = pd.read_csv("data/StudentsPerformance (1).csv")

X = df.drop("math score", axis=1)
y = df["math score"]

# Features
categorical = X.select_dtypes(include="object").columns
numerical = X.select_dtypes(exclude="object").columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

# Train and compare
for name, algorithm in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2_Score": float(r2)
    })

# Create comparison report
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="R2_Score", ascending=False
)

os.makedirs("outputs/monitoring", exist_ok=True)

comparison.to_csv(
    "outputs/monitoring/model_comparison.csv",
    index=False
)

comparison.to_json(
    "outputs/monitoring/model_comparison.json",
    orient="records",
    indent=4
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nBest Model:", comparison.iloc[0]["Model"])
print("\nModel comparison completed!")
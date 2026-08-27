import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

print("Dataset loaded successfully!")

# Features and target
X = df[["reading score", "writing score"]]
y = df["math score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()

# MLflow experiment
mlflow.set_experiment("Student_Performance")

with mlflow.start_run():

    # Train
    model.fit(X_train, y_train)

    print("Model trained successfully!")

    # Prediction
    y_pred = model.predict(X_test)

    print("Predicted output:", y_pred[0])

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("R2 Score:", r2)

    # Log parameters
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("test_size", 0.2)

    # Log metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2_Score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "model")

# Save model locally
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")
print("MLflow tracking completed!")
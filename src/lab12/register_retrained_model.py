import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

# MLflow tracking configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Student_Performance_Prediction")

# Load retrained model
model = joblib.load(
    "outputs/retraining/retrained_model.pkl"
)

# Load dataset for model signature
df = pd.read_csv("data/StudentsPerformance (1).csv")

X = df.drop("math score", axis=1)
y = df["math score"]

input_example = X.head(5)
predictions = model.predict(input_example)

signature = infer_signature(
    input_example,
    predictions
)

# Register model
model_name = "StudentPerformanceModel"

with mlflow.start_run(run_name="Student_Performance_Retraining") as run:

    mlflow.log_param("algorithm", "LinearRegression")
    mlflow.log_param("dataset", "StudentsPerformance")

    mlflow.log_metric("MAE", 4.214763142474852)
    mlflow.log_metric("RMSE", 5.393993869732847)
    mlflow.log_metric("R2_Score", 0.8804332983749563)

    mlflow.set_tag("model_type", "Regression")
    mlflow.set_tag("training_status", "Retrained")

    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name=model_name,
    signature=signature,
    input_example=input_example,
    serialization_format="cloudpickle"
)

    print("\nMLflow Run ID:", run.info.run_id)

print("\nModel registration completed!")
print("Registered Model Name:", model_name)
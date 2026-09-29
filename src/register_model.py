import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Student_Performance_Prediction"
MODEL_NAME = "StudentPerformanceModel"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    print("Experiment not found!")
else:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs:
        run_id = runs[0].info.run_id

        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        print("Model registered successfully!")
        print("Model name:", MODEL_NAME)
        print("Model version:", result.version)
        print("Run ID:", run_id)
    else:
        print("No MLflow runs found!")
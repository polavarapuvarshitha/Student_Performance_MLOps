import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

MODEL_NAME = "StudentPerformanceModel"
MODEL_VERSION = 2

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    stage="Production"
)

print("Model moved to Production!")
print("Model name:", MODEL_NAME)
print("Model version:", MODEL_VERSION)
print("Stage: Production")
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from data_preprocessing import prepare_data


DATA_PATH = "data/StudentsPerformance.csv"
MODEL_PATH = "models/model.pkl"


def main():

    # Create models folder if it does not exist
    os.makedirs("models", exist_ok=True)

    # Prepare the data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(DATA_PATH)

    # Create ML model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Create complete ML pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Create MLflow experiment
    mlflow.set_experiment(
        "Student_Performance_Prediction"
    )

    # Start MLflow run
    with mlflow.start_run():

        # Train model
        pipeline.fit(X_train, y_train)

        # Log model parameters
        mlflow.log_param(
            "model",
            "RandomForestRegressor"
        )

        mlflow.log_param(
            "n_estimators",
            100
        )

        # Save model using Joblib
        joblib.dump(
            pipeline,
            MODEL_PATH
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(
            pipeline,
            "model"
        )

        print("Model trained successfully!")
        print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
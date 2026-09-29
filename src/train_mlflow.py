import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import preprocess_data


DATA_PATH = "data/StudentsPerformance (1).csv"

mlflow.set_experiment("Student_Performance_Prediction")


def train_with_mlflow():

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(DATA_PATH)

    model = LinearRegression()

    with mlflow.start_run():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print("MLflow experiment completed!")
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2:", r2)


if __name__ == "__main__":
    train_with_mlflow()
import json
import pickle
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import preprocess_data


DATA_PATH = "data/StudentsPerformance (1).csv"
MODEL_PATH = "models/model.pkl"


def evaluate_model():

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(DATA_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation completed!")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)


if __name__ == "__main__":
    evaluate_model()
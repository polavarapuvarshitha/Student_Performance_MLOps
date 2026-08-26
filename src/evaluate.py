import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_preprocessing import prepare_data


DATA_PATH = "data/StudentsPerformance.csv"
MODEL_PATH = "models/model.pkl"


def main():

    # Prepare test data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(DATA_PATH)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    print()
    print("========== MODEL EVALUATION ==========")
    print("MAE  :", mae)
    print("MSE  :", mse)
    print("RMSE :", rmse)
    print("R2   :", r2)
    print("=======================================")


if __name__ == "__main__":
    main()
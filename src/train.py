import os
import pickle

from sklearn.linear_model import LinearRegression
from preprocess import preprocess_data


DATA_PATH = "data/StudentsPerformance (1).csv"
MODEL_PATH = "models/model.pkl"


def train_model():
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(DATA_PATH)

    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open("artifacts/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("Model trained successfully!")
    print("Model saved to:", MODEL_PATH)
    print("Preprocessor saved to: artifacts/preprocessor.pkl")


if __name__ == "__main__":
    train_model()
import sys
import os

sys.path.append(os.path.abspath("src"))

from preprocess import preprocess_data

DATA_PATH = "data/StudentsPerformance (1).csv"


def run_pipeline():
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(DATA_PATH)

    print("Pipeline executed successfully!")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_pipeline()
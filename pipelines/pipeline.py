
import sys
import os

sys.path.append(os.path.abspath("src"))

from validate_data import validate_data
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model
from inference import predict_math_score

DATA_PATH = "data/StudentsPerformance (1).csv"


def run_pipeline():
    print("\n===== ML PIPELINE STARTED =====")

    # Step 1: Data Validation
    print("\nStep 1: Data Validation")
    validate_data()

    # Step 2: Data Preprocessing
    print("\nStep 2: Data Preprocessing")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(DATA_PATH)

    print("Preprocessing completed successfully!")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # Step 3: Model Training
    print("\nStep 3: Model Training")
    train_model()

    # Step 4: Model Evaluation
    # Step 4: Model Evaluation
    print("\nStep 4: Model Evaluation")
    evaluate_model()

    # Step 5: Model Inference
    print("\nStep 5: Model Inference")

    prediction = predict_math_score(
        "female",
        "group B",
        "bachelor's degree",
        "standard",
        "completed",
        75,
        80
    )

    print("Predicted Math Score:", round(prediction, 2))

    print("\n===== ML PIPELINE COMPLETED =====")


if __name__ == "__main__":
    run_pipeline()

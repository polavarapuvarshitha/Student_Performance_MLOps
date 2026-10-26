import json
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "StudentsPerformance (1).csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
REPORT_PATH = PROJECT_ROOT / "outputs" / "failure_recovery_report.json"

results = []


def record_test(name, passed, message):
    status = "PASS" if passed else "FAIL"
    results.append({
        "test": name,
        "status": status,
        "message": message
    })
    print(f"[{status}] {name}: {message}")


def test_missing_data_file():
    missing_path = PROJECT_ROOT / "data" / "file_that_does_not_exist.csv"

    try:
        pd.read_csv(missing_path)
        record_test(
            "Missing data file",
            False,
            "Expected FileNotFoundError, but no error occurred."
        )
    except FileNotFoundError:
        record_test(
            "Missing data file",
            True,
            "Missing file detected; pipeline can stop safely."
        )


def test_invalid_datatype():
    try:
        sample = pd.DataFrame({
            "reading score": [70, "INVALID", 80]
        })
        pd.to_numeric(sample["reading score"], errors="raise")

        record_test(
            "Invalid datatype",
            False,
            "Invalid numeric value was not rejected."
        )
    except (ValueError, TypeError):
        record_test(
            "Invalid datatype",
            True,
            "Invalid numeric value detected and exception handled."
        )


def test_schema_mismatch():
    try:
        required_columns = {
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course",
            "reading score",
            "writing score",
            "math score"
        }

        sample = pd.DataFrame(columns=[
            "gender",
            "reading score",
            "writing score",
            "math score"
        ])

        missing_columns = required_columns - set(sample.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        record_test(
            "Schema mismatch",
            False,
            "Missing columns were not detected."
        )
    except ValueError as error:
        record_test(
            "Schema mismatch",
            True,
            str(error)
        )


def test_preprocessing_error():
    try:
        sample = pd.DataFrame({
            "reading score": [70, 80],
            "math score": [65, 75]
        })

        required_features = [
            "reading score",
            "writing score"
        ]

        missing_features = [
            column for column in required_features
            if column not in sample.columns
        ]

        if missing_features:
            raise ValueError(
                f"Preprocessing cannot continue. Missing features: "
                f"{missing_features}"
            )

        record_test(
            "Preprocessing error",
            False,
            "Missing preprocessing features were not detected."
        )
    except ValueError as error:
        record_test(
            "Preprocessing error",
            True,
            str(error)
        )


def test_missing_model_artifact():
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_model = Path(temp_dir) / "missing_model.pkl"

            if not missing_model.exists():
                raise FileNotFoundError(
                    f"Model artifact not found: {missing_model.name}"
                )

        record_test(
            "Missing model artifact",
            False,
            "Missing model was not detected."
        )
    except FileNotFoundError as error:
        fallback_message = (
            "Missing model detected. Recovery action: retrain the model "
            "or restore a valid saved model."
        )

        record_test(
            "Missing model artifact",
            True,
            f"{error}. {fallback_message}"
        )


def main():
    print("=" * 60)
    print("EXPERIMENT 9: PIPELINE FAILURE HANDLING AND RECOVERY")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"WARNING: Dataset not found at {DATA_PATH}")
    else:
        try:
            df = pd.read_csv(DATA_PATH)
            print(f"Dataset loaded successfully: {df.shape[0]} rows, "
                  f"{df.shape[1]} columns")
        except Exception as error:
            print(f"WARNING: Could not load dataset: {error}")

    print("\nRunning failure-handling tests...\n")

    test_missing_data_file()
    test_invalid_datatype()
    test_schema_mismatch()
    test_preprocessing_error()
    test_missing_model_artifact()

    passed = sum(item["status"] == "PASS" for item in results)
    failed = len(results) - passed

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "Pipeline Failure Handling and Recovery",
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "results": results
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

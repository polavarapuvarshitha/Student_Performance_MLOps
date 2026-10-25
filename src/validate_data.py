import pandas as pd

DATA_PATH = "data/StudentsPerformance (1).csv"

REQUIRED_COLUMNS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "reading score",
    "writing score",
    "math score"
]


def validate_data():
    df = pd.read_csv(DATA_PATH)

    print("Data validation started!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    print("Missing values:", missing_values)
    print("Duplicate rows:", duplicate_rows)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    if missing_values > 0:
        raise ValueError(f"Dataset contains {missing_values} missing values.")

    print("Required columns validation: PASSED")
    print("Data validation completed successfully!")

    return True


if __name__ == "__main__":
    validate_data()

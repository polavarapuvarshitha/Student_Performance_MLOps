import pandas as pd

DATA_PATH = "data/StudentsPerformance (1).csv"


def validate_data():
    df = pd.read_csv(DATA_PATH)

    print("Data validation completed!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    required_columns = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "reading score",
        "writing score",
        "math score"
    ]

    if all(column in df.columns for column in required_columns):
        print("Required columns validation: PASSED")
    else:
        print("Required columns validation: FAILED")


if __name__ == "__main__":
    validate_data()
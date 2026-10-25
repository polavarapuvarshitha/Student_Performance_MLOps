
import pickle
import pandas as pd

MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"


def predict_math_score(
    gender,
    race_ethnicity,
    parental_education,
    lunch,
    test_preparation,
    reading_score,
    writing_score
):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    input_data = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race_ethnicity,
        "parental level of education": parental_education,
        "lunch": lunch,
        "test preparation course": test_preparation,
        "reading score": reading_score,
        "writing score": writing_score
    }])

    transformed_data = preprocessor.transform(input_data)
    prediction = model.predict(transformed_data)

    return float(prediction[0])


if __name__ == "__main__":
    result = predict_math_score(
        "female",
        "group B",
        "bachelor's degree",
        "standard",
        "completed",
        75,
        80
    )

    print("Predicted Math Score:", round(result, 2))

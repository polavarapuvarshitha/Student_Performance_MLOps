from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import traceback

app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student performance",
    version="1.0"
)

model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running!"
    }


@app.post("/predict")
def predict(data: dict):
    try:
        input_data = pd.DataFrame([data])

        prediction = model.predict(input_data)

        return {
            "predicted_performance_index": float(prediction[0])
        }

    except Exception as e:
        print("\n========== PREDICTION ERROR ==========")
        traceback.print_exc()
        print("======================================\n")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
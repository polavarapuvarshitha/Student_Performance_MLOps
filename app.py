from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained model
model = joblib.load("models/model.pkl")

# Create FastAPI app
app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student math score",
    version="1.0"
)


# Input data format
class StudentData(BaseModel):
    reading_score: float
    writing_score: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: StudentData):

    prediction = model.predict([
        [data.reading_score, data.writing_score]
    ])

    return {
        "predicted_math_score": round(float(prediction[0]), 2)
    }
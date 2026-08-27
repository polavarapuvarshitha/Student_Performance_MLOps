# Student Performance MLOps

## Project Description

This project predicts a student's math score using reading score and writing score.

The project follows an MLOps workflow including model training, model saving,
FastAPI deployment, Docker containerization, and MLflow experiment tracking.

## Dataset

Dataset: Students Performance Dataset

### Features
- Reading Score
- Writing Score

### Target
- Math Score

## Machine Learning Model

Model used:
- Linear Regression

## Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- R2 Score

## API

The trained model is deployed using FastAPI.

### Run API

```bash
python -m uvicorn app:app --reload
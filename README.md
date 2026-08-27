# Student Performance MLOps

A Machine Learning project that predicts student performance using Python, Scikit-learn, MLflow, FastAPI, GitHub, and Docker.

## Project Overview

This project follows an MLOps workflow:

- Data preprocessing
- Model training
- Model evaluation
- Experiment tracking with MLflow
- FastAPI prediction API
- Docker containerization
- Git and GitHub version control
- Docker Hub deployment

## Dataset

The project uses the Students Performance dataset.

The dataset contains student information such as:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course
- Reading Score
- Writing Score

## Model Evaluation

The trained model was evaluated using:

- MAE
- MSE
- RMSE
- R² Score

### Results

- MAE: 4.67
- MSE: 36.31
- RMSE: 6.03
- R² Score: 0.85

## API

The prediction API is built using FastAPI.

### Run locally

```bash
python -m uvicorn api.main:app --reload
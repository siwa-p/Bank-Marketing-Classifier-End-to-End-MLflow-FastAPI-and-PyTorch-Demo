# Bank Classifier MLflow Demo

## Project Overview
This project demonstrates a complete machine learning workflow for a bank marketing classifier using FastAPI, MLflow, PyTorch, and Docker. It covers data ingestion, model training, experiment tracking, and serving predictions via a REST API.

## Key Features
- **Data Ingestion:** Loads and preprocesses bank marketing data from CSV files.
- **DuckLake Integration:** Implements DuckLake for accelerated data querying and efficient data access during preprocessing and feature engineering.
- **Model Training:** Trains a neural network classifier using PyTorch, with hyperparameter optimization via Optuna.
- **Experiment Tracking:** Uses MLflow to log metrics, parameters, and model artifacts for reproducibility.
- **Model Serving:** Serves predictions through a FastAPI REST API, supporting both raw feature input and one-hot encoded features.
- **Dockerized Environment:** Includes Docker and docker-compose for easy local deployment and isolation.

## API Endpoints
- `GET /`: Welcome message.
- `GET /health`: Health check with UTC timestamp.
- `POST /get_features`: Accepts a list of raw features, returns a dictionary of processed (one-hot encoded) features for model input.
- `POST /predict`: Accepts a JSON dictionary of model input features, returns the prediction.

## Folder Structure
- `src/`: Source code for API, model, utilities, and prediction logic
- `data/`: Training and test CSV files
- `mlruns/`: MLflow experiment logs and model artifacts
- `Dockerfile`, `docker-compose.yml`: Containerization setup

## Summary
- Built a full ML pipeline for bank marketing classification
- Automated feature engineering and model input preparation
- Enabled experiment tracking and model management with MLflow
- Exposed a user-friendly API for inference and feature processing
- Containerized the workflow for reproducibility and easy deployment

---
For more details, see the source code and comments in each file.
# mlflow-demo


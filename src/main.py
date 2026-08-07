import os
import sys
from datetime import datetime
from typing import Literal

import mlflow
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from utilities.logging_config import logger
from utilities.utils import process_features

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utilities'))

from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI()

model = mlflow.pytorch.load_model("models:/bank_classifier@production", map_location=torch.device("cpu"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info("Model loaded from MLflow registry (production).")


class CustomerFeatures(BaseModel):
    age: int
    job: Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                 "retired", "self-employed", "services", "student", "technician",
                 "unemployed", "unknown"]
    marital: Literal["divorced", "married", "single"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["no", "yes"]
    balance: int
    housing: Literal["no", "yes"]
    loan: Literal["no", "yes"]
    contact: Literal["cellular", "telephone", "unknown"]
    day: int
    month: Literal["jan", "feb", "mar", "apr", "may", "jun",
                   "jul", "aug", "sep", "oct", "nov", "dec"]
    campaign: int
    previous: int
    poutcome: Literal["failure", "other", "success", "unknown"]


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Bank Classifier API!"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "OK", "timestamp": datetime.utcnow().isoformat()}


EXPECTED_COLS = [
    "age", "balance", "day", "campaign", "previous", "default", "housing", "loan",
    "job_admin.", "job_blue-collar", "job_entrepreneur", "job_housemaid", "job_management",
    "job_retired", "job_self-employed", "job_services", "job_student", "job_technician",
    "job_unemployed", "job_unknown", "marital_divorced", "marital_married", "marital_single",
    "education_primary", "education_secondary", "education_tertiary", "education_unknown",
    "contact_cellular", "contact_telephone", "contact_unknown", "month_apr", "month_aug",
    "month_dec", "month_feb", "month_jan", "month_jul", "month_jun", "month_mar",
    "month_may", "month_nov", "month_oct", "month_sep", "poutcome_failure", "poutcome_other",
    "poutcome_success", "poutcome_unknown"
]

@app.post("/predict", tags=["Prediction"])
async def predict(features: CustomerFeatures):
    try:
        df = pd.DataFrame([features.model_dump()])
        for col in ["default", "housing", "loan"]:
            df[col] = df[col] == "yes"
        processed = process_features(df).iloc[0]
        for col in EXPECTED_COLS:
            if col not in processed.index:
                processed[col] = 0
        input_tensor = torch.tensor(processed[EXPECTED_COLS].astype("float32").values, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            prediction = int(prob > 0.5)
        return {"prediction": prediction, "probability": round(prob, 4)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

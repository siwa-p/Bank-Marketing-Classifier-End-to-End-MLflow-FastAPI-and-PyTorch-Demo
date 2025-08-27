from datetime import datetime
from fastapi import FastAPI, HTTPException
from utilities.logging_config import logger
from pydantic import BaseModel
from single_prediction import predict_single
from utilities.utils import MLPModel, process_features
import sys
import os
import torch
import asyncio
from typing import List
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utilities'))

app = FastAPI()

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Bank Classifier API!"}

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "OK", "timestamp": datetime.utcnow().isoformat()}


from typing import Any

@app.post("/get_features", tags=["Features"])
async def get_features(q: List[Any]):
    # Define the feature names in the expected order
    feature_names = [
        "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"
    ]
    expected_cols = [
        "age", "balance", "day", "duration", "campaign", "pdays", "previous",
        "job_admin.", "job_blue-collar", "job_entrepreneur", "job_housemaid", "job_management",
        "job_retired", "job_self-employed", "job_services", "job_student", "job_technician",
        "job_unemployed", "job_unknown", "marital_divorced", "marital_married", "marital_single",
        "education_primary", "education_secondary", "education_tertiary", "education_unknown",
        "default_no", "default_yes", "housing_no", "housing_yes", "loan_no", "loan_yes",
        "contact_cellular", "contact_telephone", "contact_unknown", "month_apr", "month_aug",
        "month_dec", "month_feb", "month_jan", "month_jul", "month_jun", "month_mar",
        "month_may", "month_nov", "month_oct", "month_sep", "poutcome_failure", "poutcome_other",
        "poutcome_success", "poutcome_unknown"
    ]
    if len(q) != len(feature_names):
        raise HTTPException(status_code=400, detail=f"Expected {len(feature_names)} features, got {len(q)}")
    raw_dict = {name: q[i] for i, name in enumerate(feature_names)}
    processed = process_features(pd.DataFrame([raw_dict])).iloc[0]
    for col in expected_cols:
        if col not in processed.index:
            processed[col] = False
    # Convert all values to native Python types for JSON serialization
    def to_native(val):
        if isinstance(val, (np.generic, np.bool_)):
            return val.item()
        return bool(val) if isinstance(val, bool) else val
    return {col: to_native(processed[col]) for col in expected_cols}



@app.post("/predict", tags=["Prediction"])
async def predict(features: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLPModel(in_features=52, hidden_units=[256, 48, 32], dropout_prob=0.3)
    model.load_state_dict(torch.load("mlruns/best_model.pt", map_location=device))
    logger.info("Model loaded successfully.")
    try:
        # Only map fields that differ between API and model
        mapping = {
            "job_admin": "job_admin.",
            "job_blue_collar": "job_blue-collar",
            "job_self_employed": "job_self-employed"
        }
        model_input = {}
        for k, v in features.items():
            if k in mapping:
                model_input[mapping[k]] = v
            else:
                model_input[k] = v
        prediction = predict_single(model_input, model)
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

if __name__ == "__main__":
    input = [
    26, "technician", "married", "secondary", "no", 889, "yes", "no", "cellular", 3, "feb", 902, 1, -1, 0, "unknown"
    ]
    asyncio.run(predict(get_features(input)))
from datetime import datetime
from fastapi import FastAPI, HTTPException
from utilities.logging_config import logger
from pydantic import BaseModel
from single_prediction import predict_single
app = FastAPI()

class Features(BaseModel):
    age: int
    balance: float
    day: int
    duration: int
    campaign: int
    pdays: int
    previous: int
    job_admin: bool
    job_blue_collar: bool
    job_entrepreneur: bool
    job_housemaid: bool
    job_management: bool
    job_retired: bool
    job_self_employed: bool
    job_services: bool
    job_student: bool
    job_technician: bool
    job_unemployed: bool
    job_unknown: bool
    marital_divorced: bool
    marital_married: bool
    marital_single: bool
    education_primary: bool
    education_secondary: bool
    education_tertiary: bool
    education_unknown: bool
    default_no: bool
    default_yes: bool
    housing_no: bool
    housing_yes: bool
    loan_no: bool
    loan_yes: bool
    contact_cellular: bool
    contact_telephone: bool
    contact_unknown: bool
    month_apr: bool
    month_aug: bool
    month_dec: bool
    month_feb: bool
    month_jan: bool
    month_jul: bool
    month_jun: bool
    month_mar: bool
    month_may: bool
    month_nov: bool
    month_oct: bool
    month_sep: bool
    poutcome_failure: bool
    poutcome_other: bool
    poutcome_success: bool
    poutcome_unknown: bool


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Bank Classifier API!"}

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "OK", "timestamp": datetime.utcnow().isoformat()}

@app.get("/predict", tags=["Prediction"])
async def predict(features: Features):
    try:
        # Convert features to dictionary
        features_dict = features.dict()
        # Call the prediction function (to be implemented)
        prediction = predict_single(features_dict)
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


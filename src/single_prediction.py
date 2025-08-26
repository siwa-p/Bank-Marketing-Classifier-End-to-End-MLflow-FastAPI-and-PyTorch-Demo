import pandas as pd
import torch
from utilities.utils import MLPModel
import duckdb

# load the trained model
def predict_single(sample, model_path="models/best_model.pth"):

    columns = [
        'id', 'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
        'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
        'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician',
        'job_unemployed', 'job_unknown', 'marital_divorced', 'marital_married', 'marital_single',
        'education_primary', 'education_secondary', 'education_tertiary', 'education_unknown',
        'default_no', 'default_yes', 'housing_no', 'housing_yes', 'loan_no', 'loan_yes',
        'contact_cellular', 'contact_telephone', 'contact_unknown', 'month_apr', 'month_aug',
        'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun', 'month_mar',
        'month_may', 'month_nov', 'month_oct', 'month_sep', 'poutcome_failure', 'poutcome_other',
        'poutcome_success', 'poutcome_unknown'
    ]

    model = torch.load(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = pd.DataFrame([sample], columns=columns)
    input_tensor = torch.tensor(df.astype('float32').values, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)
        prediction = (prob > 0.5).long().item()
    return prediction

# Example usage:
sample = {
    'id': 123456,
    'age': 35,
    'balance': 2000,
    'day': 15,
    'duration': 300,
    'campaign': 2,
    'pdays': 999,
    'previous': 0,
    'job_admin.': False,
    'job_blue-collar': False,
    'job_entrepreneur': False,
    'job_housemaid': False,
    'job_management': True,
    'job_retired': False,
    'job_self-employed': False,
    'job_services': False,
    'job_student': False,
    'job_technician': False,
    'job_unemployed': False,
    'job_unknown': False,
    'marital_divorced': False,
    'marital_married': True,
    'marital_single': False,
    'education_primary': False,
    'education_secondary': True,
    'education_tertiary': False,
    'education_unknown': False,
    'default_no': True,
    'default_yes': False,
    'housing_no': True,
    'housing_yes': False,
    'loan_no': True,
    'loan_yes': False,
    'contact_cellular': True,
    'contact_telephone': False,
    'contact_unknown': False,
    'month_apr': False,
    'month_aug': False,
    'month_dec': False,
    'month_feb': False,
    'month_jan': False,
    'month_jul': False,
    'month_jun': False,
    'month_mar': False,
    'month_may': True,
    'month_nov': False,
    'month_oct': False,
    'month_sep': False,
    'poutcome_failure': False,
    'poutcome_other': False,
    'poutcome_success': False,
    'poutcome_unknown': True
}

print("Predicted class:", predict_single(sample))

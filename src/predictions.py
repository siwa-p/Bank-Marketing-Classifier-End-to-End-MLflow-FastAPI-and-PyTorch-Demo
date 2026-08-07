import duckdb
import os
import torch
import mlflow.pytorch
from torch.utils.data import DataLoader, Dataset
from utilities.utils import process_features
from utilities.logging_config import logger

import pandas as pd

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

class DuckLakeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        features_tot = process_features(self.data)
        self.X = torch.tensor(features_tot.astype('float32').values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.X[index]
    
con = duckdb.connect(database=':memory:')
folder_path = os.getenv("DUCKLAKE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ducklake_db_path = os.path.join(folder_path, "catalog.ducklake")
ducklake_data_path = os.path.join(folder_path, "catalog_data")
con.execute(f"ATTACH 'ducklake:{ducklake_db_path}' AS my_lake (DATA_PATH '{ducklake_data_path}')")
con.execute("USE my_lake")
test_data = con.execute("SELECT * FROM bank_schema.test").fetchdf()
print("Test data shape:", test_data.shape)
print("First few rows of test data:\n", test_data.head())
test_dataset = DuckLakeDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mlflow.pytorch.load_model("models:/bank_classifier@production", map_location=device)
logger.info("Model loaded from MLflow registry (production).")

def predict(model, dataloader):
    model.eval()
    model.to(device)
    all_predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predict_proba = torch.sigmoid(outputs).squeeze(1)
            preds = predict_proba.cpu().numpy()
            rounded_preds = [round(float(p), 2) for p in preds]
            all_predictions.extend(rounded_preds)
    return all_predictions

# save all predictions to a csv file
predictions = predict(model, test_loader)
output_df = pd.DataFrame({'id': test_data['id'], 'y': predictions})
output_df.to_csv("predictions.csv", index=False)
logger.info("Predictions saved to predictions.csv")
print("Predictions saved to predictions.csv")
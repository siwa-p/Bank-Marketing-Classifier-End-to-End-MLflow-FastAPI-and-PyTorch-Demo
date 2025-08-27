import mlflow
import duckdb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset   
from sklearn.model_selection import train_test_split  
import optuna
from utilities.utils import MLPModel, process_features  
import os
from utilities.logging_config import logger
import numpy as np
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("bank_classification")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class DuckLakeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        features_tot = process_features(self.data)
        self.X = torch.tensor(features_tot.astype('float32').values, dtype=torch.float32)
        self.y = torch.tensor(self.data['y'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.X[index], self.y[index]

con = duckdb.connect(database=':memory:')
folder_path = r"D:\Data Engineering Bootcamp\mlflow-demo"
ducklake_db_path = os.path.join(folder_path, "catalog.ducklake")
ducklake_data_path = os.path.join(folder_path, "catalog_data")
con.execute(f"ATTACH 'ducklake:{ducklake_db_path}' AS my_lake (DATA_PATH '{ducklake_data_path}')")
con.execute("USE my_lake")

train_data = con.execute("SELECT * FROM bank_schema.train").fetchdf()
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=1)
logger.info(f"Train data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
train_dataset = DuckLakeDataset(train_data)
test_dataset = DuckLakeDataset(val_data)
logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
in_features = train_dataset.X.shape[1]
logger.info(f"Number of input features: {in_features}")

def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    hidden1 = trial.suggest_int("hidden_units_1", 128, 256, step=32)
    hidden2 = trial.suggest_int("hidden_units_2", 16, 64, step=8)
    hidden3 = trial.suggest_int("hidden_units_3", 8, 32, step=4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with mlflow.start_run():
        mlflow.set_tag("trial_number", trial.number)
        mlflow.log_params({
            "learning_rate": lr,
            "hidden_units": [hidden1, hidden2, hidden3],
            "batch_size": batch_size
        })
        mlpmodel = MLPModel(in_features=in_features, hidden_units=[hidden1, hidden2, hidden3]).to(device)
        optimizer = optim.Adam(mlpmodel.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        
        for epoch in range(5):
            mlpmodel.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = mlpmodel(data)
                loss = loss_fn(output, target.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
            mlpmodel.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = mlpmodel(data)
                    loss = loss_fn(output, target.unsqueeze(1).float())
                    val_loss += loss.item() * data.size(0)
                    probs = torch.sigmoid(output)
                    predicted = (probs > 0.5).long().squeeze(1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            val_loss /= len(test_loader.dataset)
            val_accuracy = val_correct / val_total
            mlflow.log_metrics({
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
logger.info(f"Best trial: {study.best_trial.params}")
logger.info(f"Best validation loss: {study.best_value}")
best_params = study.best_params
with mlflow.start_run():
    mlflow.set_tag("final_model", "mnist_mlp")
    mlflow.log_params(best_params)
    final_model = MLPModel(
        in_features=in_features,
        hidden_units=[
            best_params['hidden_units_1'],
            best_params['hidden_units_2'],
            best_params["hidden_units_3"]
        ]
    ).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    for epoch in range(20):
        final_model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = final_model(data)
            loss = loss_fn(output, target.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
    final_model.eval()
    sample_input = np.random.uniform(size=[1, in_features]).astype(np.float32)
    with torch.no_grad():
        output = final_model(torch.tensor(sample_input).to(device))
        sample_output = output.cpu().numpy()
    signature = infer_signature(sample_input, sample_output)
    mlflow.pytorch.log_model(final_model, "model", signature=signature)
    logger.info("Final model logged to MLflow.")

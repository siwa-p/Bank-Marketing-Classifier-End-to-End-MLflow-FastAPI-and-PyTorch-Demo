import mlflow
import duckdb
import torch
import torch.nn as nn
import torch.optim as optim
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

# Subsample for faster hyperparameter search
search_train = train_data.sample(frac=0.1, random_state=1)
search_val = val_data.sample(frac=0.1, random_state=1)
search_train_dataset = DuckLakeDataset(search_train)
search_val_dataset = DuckLakeDataset(search_val)
logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
in_features = train_dataset.X.shape[1]
logger.info(f"Number of input features: {in_features}")

neg = (train_data['y'] == 0).sum()
pos = (train_data['y'] == 1).sum()
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
logger.info(f"Class balance — neg: {neg}, pos: {pos}, pos_weight: {pos_weight.item():.2f}")

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden1 = trial.suggest_int("hidden_units_1", 64, 128, step=32)
    hidden2 = trial.suggest_int("hidden_units_2", 16, 48, step=16)
    hidden3 = trial.suggest_int("hidden_units_3", 8, 16, step=8)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    train_loader = DataLoader(search_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(search_val_dataset, batch_size=batch_size, shuffle=False)

    with mlflow.start_run():
        mlflow.set_tag("trial_number", trial.number)
        mlflow.log_params({
            "learning_rate": lr,
            "hidden_units": [hidden1, hidden2, hidden3],
            "batch_size": batch_size
        })
        mlpmodel = MLPModel(in_features=in_features, hidden_units=[hidden1, hidden2, hidden3]).to(device)
        optimizer = optim.Adam(mlpmodel.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(5):
            mlpmodel.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = mlpmodel(data)
                loss = loss_fn(output, target.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                probs = torch.sigmoid(output)
                predicted = (probs > 0.5).long().squeeze(1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            train_loss /= len(train_loader.dataset)
            train_accuracy = train_correct / train_total
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
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)
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
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0
    for epoch in range(10):
        final_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = final_model(data)
            loss = loss_fn(output, target.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            probs = torch.sigmoid(output)
            predicted = (probs > 0.5).long().squeeze(1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        final_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = final_model(data)
                loss = loss_fn(output, target.unsqueeze(1).float())
                val_loss += loss.item() * data.size(0)
                probs = torch.sigmoid(output)
                predicted = (probs > 0.5).long().squeeze(1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }, step=epoch)
    final_model.eval()
    sample_input = np.random.uniform(size=[1, in_features]).astype(np.float32)
    with torch.no_grad():
        output = final_model(torch.tensor(sample_input).to(device))
        sample_output = output.cpu().numpy()
    signature = infer_signature(sample_input, sample_output)
    mlflow.pytorch.log_model(final_model, "model", signature=signature, registered_model_name="bank_classifier")
    client = mlflow.MlflowClient()
    versions = client.search_model_versions("name='bank_classifier'")
    latest_version = max(versions, key=lambda v: int(v.version)).version
    client.set_registered_model_alias("bank_classifier", "production", latest_version)
    logger.info(f"Model version {latest_version} registered and set as 'production'.")

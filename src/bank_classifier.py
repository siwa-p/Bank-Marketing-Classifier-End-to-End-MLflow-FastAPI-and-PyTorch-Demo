import mlflow
import duckdb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset   
from sklearn.model_selection import train_test_split  
import optuna
from utilities.utils import MLPModel  
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("bank_classification")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuckLakeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        features = self.data.drop('y', axis=1)
        numerical_features = features.select_dtypes(include=['number'])
        categorical_features = features.select_dtypes(exclude = ['number'])
        if not categorical_features.empty:
            encoded_features = pd.get_dummies(categorical_features)
            features_tot = pd.concat([numerical_features, encoded_features], axis=1)
        else:
            features_tot = numerical_features
        features_tot = features_tot.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.X = torch.tensor(features_tot.astype('float32').values, dtype=torch.float32)
        self.y = torch.tensor(self.data['y'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.X[index], self.y[index]



con = duckdb.connect(database=':memory:')
con.execute("ATTACH 'ducklake:/home/prahald/Documents/Data Engineering Bootcamp/mlflow-demo/catalog.ducklake' AS my_lake (DATA_PATH '/home/prahald/Documents/Data Engineering Bootcamp/mlflow-demo/catalog_data')")
con.execute("USE my_lake")

train_data = con.execute("SELECT * FROM bank_schema.train").fetchdf()
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=1)

train_dataset = DuckLakeDataset(train_data)
test_dataset = DuckLakeDataset(val_data)
in_features = train_dataset.X.shape[1]
model = MLPModel(in_features=in_features).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    hidden1 = trial.suggest_int("hidden_units_1", 80, 144, step=16)
    hidden2 = trial.suggest_int("hidden_units_2", 16, 64, step=8)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    with mlflow.start_run():
        mlflow.set_tag("trial_number", trial.number)
        mlflow.set_tag("study_name", "mnist_hyperopt")
        mlflow.log_params({
            "learning_rate": lr,
            "hidden_units": [hidden1, hidden2],
            "batch_size": batch_size
        })
        in_features = train_dataset.X.shape[1]
        mlpmodel = MLPModel(in_features=in_features, hidden_units=[hidden1, hidden2]).to(device)
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
            print(f"Epoch {epoch+1} training complete.")

                    
            mlpmodel.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = mlpmodel(data)
                    loss = loss_fn(output, target.unsqueeze(1).float())
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_loss /= len(test_loader)
            val_acc = 100. * val_correct / val_total
            mlflow.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)     
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print("Best hyperparameters:", study.best_params)
print("Best validation loss:", study.best_value)

# train with best hyperparameters and log final model
best_params = study.best_params
with mlflow.start_run():
    mlflow.set_tag("final_model", "mnist_mlp")
    mlflow.log_params(best_params)
    in_features = train_dataset.X.shape[1]
    
    final_model = MLPModel(in_features=in_features, hidden_units=[best_params['hidden_units_1'], best_params['hidden_units_2']]).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    
    for epoch in range(5):
        final_model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = final_model(data)
            loss = loss_fn(output, target.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
    
    mlflow.pytorch.log_model(final_model, "model", registered_model_name="MNIST_MLP_Model")
    
if __name__ == "__main__":
    print("Training complete. Best hyperparameters:", best_params)
    
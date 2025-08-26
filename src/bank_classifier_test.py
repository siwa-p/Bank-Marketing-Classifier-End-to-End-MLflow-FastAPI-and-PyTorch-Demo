import duckdb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset   
from sklearn.model_selection import train_test_split 
from torchvision import transforms, datasets
import optuna
from utilities.utils import MLPModel  
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

for epoch in range(1):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} training complete.")

# Simple validation
model.eval()
val_loss = 0
val_correct = 0
val_total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target.unsqueeze(1).float())
        val_loss += loss.item()
        _, predicted = output.max(1)
        val_total += target.size(0)
        val_correct += predicted.eq(target).sum().item()
val_loss /= len(test_loader)
val_acc = 100. * val_correct / val_total
print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2f}%")


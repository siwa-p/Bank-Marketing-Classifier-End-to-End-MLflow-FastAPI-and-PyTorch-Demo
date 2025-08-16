import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchinfo import summary
import optuna
from utilities.utils import MLPModel  
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mnist_classification")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    hidden1 = trial.suggest_int("hidden_units_1", 64, 256, step=32)
    hidden2 = trial.suggest_int("hidden_units_2", 32, 128, step=16)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
        )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
        )
    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": lr,
            "hidden_units": [hidden1, hidden2],
            "batch_size": batch_size
        })
        mlpmodel = MLPModel(hidden_units=[hidden1, hidden2]).to(device)
        optimizer = optim.Adam(mlpmodel.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(5):  # Reduced epochs for faster trials
            mlpmodel.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = mlpmodel(data)
                loss = loss_fn(output, target)
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
                    loss = loss_fn(output, target)
                    
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
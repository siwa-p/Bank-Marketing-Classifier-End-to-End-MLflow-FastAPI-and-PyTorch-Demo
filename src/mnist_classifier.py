import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchinfo import summary
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mnist_classification")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
params = {
    "epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 64,
    "optimizer": "adam",
    "model_type": "MLP",
    "hidden_units": [128, 64],
}

with mlflow.start_run():
    mlflow.log_params(params)
    
    model = MLPModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(model, input_size=(params['batch_size'], 1, 28, 28))))
    mlflow.log_artifact("model_summary.txt")
    
    for epoch in range(params["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                batch_loss = total_loss / (batch_idx + 1)
                batch_acc = 100. * correct / total
                mlflow.log_metrics(
                    {"batch_loss": batch_loss, "batch_accuracy": batch_acc}, step=epoch * len(train_loader) + batch_idx
                )
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss /= len(test_loader)
        val_acc = 100. * val_correct / val_total
        mlflow.log_metrics(
            {
                "train_loss": epoch_loss, 
                "train_accuracy": epoch_acc, 
                "val_loss": val_loss, 
                "val_accuracy": val_acc
            },
            step=epoch
        )
        print(f'Epoch {epoch+1}/{params["epochs"]}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
    model_info = mlflow.pytorch.log_model(model, name = "mnist_model")
    
    # final evaluation
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total
    mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
    print(f"Final Test Accuracy: {test_acc:.2f}%")
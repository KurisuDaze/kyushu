import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
       
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        
        # Generate noise
        noise = torch.normal(self.mean, self.std, size=tensor.size())
        
        # Add noise
        noisy_tensor = tensor + noise
        
        noisy_tensor = torch.clamp(noisy_tensor, 0., 1.)
        
        return noisy_tensor
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  
    AddGaussianNoise(mean=0., std=0.3)  
])

# Load dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)

# Loss function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                # forward
                outputs = model(images)
                loss = criterion(outputs, labels)

                # backward & optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                tepoch.set_postfix(loss=total_loss/len(train_loader), epoch=epoch+1)
        
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss/len(train_loader):.4f}")

# Test
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save model
def save_model(model, path="mnist_cnn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load model
def load_model(model, path="mnist_cnn.pth"):
    if not os.path.exists(path):
        return
    model.load_state_dict(torch.load(path))
    model.to(device)
    print(f"Model loaded from {path}")

# Run py
load_model(model)
#train_model(model, train_loader, criterion, optimizer, epochs=5)
test_model(model, test_loader)
#save_model(model)

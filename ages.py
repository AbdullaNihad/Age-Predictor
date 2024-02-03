import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
# Assuming you have already defined your dataset and DataLoader (train_loader)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

class XRayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        age = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, age

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets and dataloaders
train_dataset = XRayDataset(csv_file='trainingdata.csv', root_dir='./JPGs/', transform=transform)
test_dataset = XRayDataset(csv_file='testdata.csv', root_dir='./JPGs/', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Build the ResNet Model
class AgePredictionResNet(nn.Module):
    def __init__(self):
        super(AgePredictionResNet, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Linear(resnet.fc.in_features, 1)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)

# Initialize the model, loss function, and optimizer
model = AgePredictionResNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 60

for epoch in range(num_epochs):
    model.train()
    running_mae = 0.0
    running_mse = 0.0

    for images, ages in train_loader:
        images, ages = images.to(device), ages.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages.float().view(-1, 1))
        loss.backward()
        optimizer.step()

        # Update running MAE and MSE
        running_mae += mean_absolute_error(ages.cpu().numpy(), outputs.detach().cpu().numpy())
        running_mse += mean_squared_error(ages.cpu().numpy(), outputs.detach().cpu().numpy())

    # Calculate average MAE and MSE for the epoch
    average_mae = running_mae / len(train_loader)
    average_mse = running_mse / len(train_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, MAE: {average_mae:.2f}, MSE: {average_mse:.2f}')

torch.save(model.state_dict(), 'age_prediction_model.pth')
# Evaluation on the test set
model.eval()
all_predictions = []
all_actual_ages = []

with torch.no_grad():
    for images, ages in test_loader:
        images, ages = images.to(device), ages.to(device)
        outputs = model(images)
        all_predictions.extend(outputs.cpu().numpy())
        all_actual_ages.extend(ages.cpu().numpy())

# Calculate final MAE and MSE on the test set
final_mae = mean_absolute_error(all_actual_ages, all_predictions)
final_mse = mean_squared_error(all_actual_ages, all_predictions)


print(f'Final MAE on test set: {final_mae:.2f}')
print(f'Final MSE on test set: {final_mse:.2f}')


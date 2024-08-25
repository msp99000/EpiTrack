import streamlit as st
import scipy.io
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

#THESE FIRST THREE ARE THE SAME FOR ALL MODELS
# Function to load EEG data
def load_eeg_data(base_dir):
    categories = ['preictal', 'interictal', 'ictal']
    labels = {'preictal': 0, 'interictal': 1, 'ictal': 2}
    X = []  # Raw Data Matrix 
    y = []  # Label vector
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        for file in os.listdir(cat_dir):
            file_path = os.path.join(cat_dir, file)
            if file.endswith('.mat'):
                mat_data = scipy.io.loadmat(file_path)
                data = mat_data[category]
                X.append(data.flatten())  # Flatten the EEG segment
                y.append(labels[category])
    return np.array(X), np.array(y)

# Load the EEG data
base_dir = 'EEG_Epilepsy_Datasets'
X, y = load_eeg_data(base_dir)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=100)




#1D-CNN

import scipy.io
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

# Convert data to PyTorch tensors with an added channel dimension
# Convert data to PyTorch tensors and ensure they are correctly shaped with channel as the second dimension
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Shape [num_samples, 1, 1024]
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)    # Shape [num_samples, 1, 1024]
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Improved CNN model
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=5, padding=2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 256, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = x.view(-1, 256 * 256)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Hyperparameters and optimization
model = ImprovedCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()




for inputs, labels in train_loader:
    print(inputs.shape)  # Should be [32, 1, 1024]
    print(labels.shape)  # Should be [32]
    break  # Exit after the first batch


# Training loop
# Training loop
def train_save_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # Ensure inputs are correctly shaped
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}: Training Complete")
    return model, train_losses  # Ensure this return statement is included

def train_and_save():
    model = ImprovedCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    trained_model, train_losses = train_save_model(model, train_loader, criterion, optimizer, num_epochs=20)
    torch.save(trained_model.state_dict(), 'cnn_model.pth')

# Uncomment below to train and save the model, then comment it out after use.
# Training the model
model, train_losses = train_save_model(model, train_loader, criterion, optimizer, num_epochs=50)
    
# Example of training and saving the model
cnn_model = ImprovedCNN()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train_save_model(cnn_model, train_loader, criterion, optimizer)


# Before calling the train_save_model function, ensure the DataLoader is correctly set up:
print(f'Training data loader: {len(train_loader.dataset)} samples')
print(f'Testing data loader: {len(test_loader.dataset)} samples')

# Then call the train_save_model function
model, train_losses = train_save_model(model, train_loader, criterion, optimizer)


# Train the model
model, train_losses = train_save_model(model, train_loader, criterion, optimizer)



import pandas as pd
import torch

def evaluate_model_and_save_results(model, test_loader, criterion, file_path='cnn_pred.pkl'):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.tolist())
            all_probabilities.extend(probabilities.tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total

    # Save results to a DataFrame and then to a pickle file
    pred_df = pd.DataFrame({
        'Predictions': all_predictions,
        'Probabilities': all_probabilities
    })
    pred_df.to_pickle(file_path)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy*100:.2f}%")
    return test_loss, accuracy

# Assuming `model` is the trained ImprovedCNN model, evaluate and save results
test_loss, test_accuracy = evaluate_model_and_save_results(model, test_loader, criterion, file_path='improved_cnn_pred.pkl')

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)














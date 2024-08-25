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
import pickle

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




# Define extract_knn_features function
from scipy.signal import welch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def extract_knn_features(segment, fs=200):

    # Calculate the desired statistical features
    min_val = np.min(segment)
    max_val = np.max(segment)
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    variance_val = np.var(segment)
    skewness_val = scipy.stats.skew(segment)
    kurtosis_val = scipy.stats.kurtosis(segment)
# calculate Zero Crossing Rate
    centered_segment = segment - np.mean(segment)
    zcr_val = ((centered_segment[:-1] * centered_segment[1:]) < 0).sum()

 #calculate signal magnitued Area
    sma_val = np.sum(np.abs(segment)) / len(segment)
#calculate   Spectral Centroid
    freqs, psd = welch(segment, fs=fs)
    centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) != 0 else 0
# Hjorth Parameters
    first_derivative = np.diff(segment, n=1)
    activity = variance_val
    mobility = np.sqrt(np.var(first_derivative) / activity) if activity != 0 else 0
    complexity = np.sqrt(np.var(np.diff(first_derivative, n=1)) / np.var(first_derivative)) / mobility if mobility != 0 else 0
   # Return them as a flat list (no other dimensions)
    return [min_val, max_val, mean_val, std_val, variance_val, skewness_val, kurtosis_val, zcr_val,  sma_val, centroid, activity, mobility, complexity]

# Apply feature extraction to the dataset (X_train) and store them in a new matrix (X_train_features)
X_train_features = np.array([extract_knn_features(segment) for segment in X_train])
X_test_features = np.array([extract_knn_features(segment) for segment in X_test])

knn_model = KNeighborsClassifier(n_neighbors=18)

# Fit the model on the training data
knn_model.fit(X_train_features, y_train)

# Fit the model on the training data
knn_model.fit(X_train_features, y_train)

# Predict the labels and probabilities for the test data
knn_predictions = knn_model.predict(X_test_features)
knn_probabilities = knn_model.predict_proba(X_test_features)

knn_accuracy = accuracy_score(y_test, knn_predictions)
print(knn_accuracy)

# Create a DataFrame similar to the CNN output
knn_results_df = pd.DataFrame({
    'Predictions': knn_predictions,
    'Probabilities': [list(probs) for probs in knn_probabilities]
})

# Save the DataFrame to a pickle file for later use
knn_results_df.to_pickle('knn_pred.pkl')

# Optionally, print to check the format
print(knn_results_df.head())

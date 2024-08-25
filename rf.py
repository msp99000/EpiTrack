from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
import pywt
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


# Save y_test to a pickle file
with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)


#nest-13, db4, l5 for 86%

def extract_wavelet_features(segment):
    coeffs = pywt.wavedec(segment, 'db4', level=7)
    features = []
    for coeff in coeffs:
        features.extend([np.min(coeff), np.max(coeff), np.mean(coeff), np.std(coeff)])
    return features

#X_train = X_train_smote
#y_train = y_train_smote
X_train_features = np.array([extract_wavelet_features(segment) for segment in X_train])
X_test_features = np.array([extract_wavelet_features(segment) for segment in X_test])


rf = RandomForestClassifier(n_estimators=15, random_state=42)     #instantiate model

rf.fit(X_train_features, y_train)         #fit/train the model on our data i.e. using our features and their corresponding labels. (recall dog example)

rf_pred = rf.predict(X_test_features)     #use test features to make label prediction (preictal, interictal, ictal)

rf_probabilities = rf.predict_proba(X_test_features)

rf_accuracy = accuracy_score(y_test, rf_pred)
print(rf_accuracy)

#Key variables:
#13

#X_train_features   - the features extracted from the training set (X_train)
#y_train            - the labels/correct answers for the training set features
#X_test_features    - the features extracted from the testing set (X_test)
#rf_pred            - Think of this as y_prediction if thats easier. Your models label predictions based on extracted testing features (X_test_features).
#y_test             - the actual correct answers of the testing set. We compare the models predictions to this to get accuracy


# Create a DataFrame similar to the CNN output
rf_results_df = pd.DataFrame({
    'Predictions': rf_pred,
    'Probabilities': [list(probs) for probs in rf_probabilities]
})

# Save the DataFrame to a pickle file for later use
rf_results_df.to_pickle('rf_pred.pkl')

# Optionally, print to check the format
print(rf_results_df.head())

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



def extract_wavelet_features(segment):
    coeffs = pywt.wavedec(segment, 'db4', level=6)
    features = []
    for coeff in coeffs:
        features.extend([np.min(coeff), np.max(coeff), np.mean(coeff), np.std(coeff)])
    return features

#X_train = X_train_smote
#y_train = y_train_smote
X_train_features = np.array([extract_wavelet_features(segment) for segment in X_train])
X_test_features = np.array([extract_wavelet_features(segment) for segment in X_test])

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#op 0 md=3, n_e=50,

# Initialize base estimator
base_est = DecisionTreeClassifier(max_depth=3)

# Initialize ada model
ada_clf = AdaBoostClassifier(base_est, n_estimators=50, random_state=42)

ada_clf.fit(X_train_features, y_train)

ada_pred = ada_clf.predict(X_test_features)
ada_probabilities = ada_clf.predict_proba(X_test_features)

# Create a DataFrame similar to the CNN output
ada_results_df = pd.DataFrame({
    'Predictions': ada_pred,
    'Probabilities': [list(probs) for probs in ada_probabilities]
})

# Save the DataFrame to a pickle file for later use
ada_results_df.to_pickle('ada_pred.pkl')

# Optionally, print to check the format
print(ada_results_df.head())


# Evaluate the model
accuracy_ada = accuracy_score(y_test, ada_pred)
print("AdaBoost Accuracy:", accuracy_score(y_test, ada_pred))
#print(ada_pred)

#print("Adaboost Probabilities:\n", ada_probabilities)

#EEG data is highly variable and depends on many factors including the sampling rate, sampling hardware, patients,
#
#gpt 1.76 trillion, a1 36k per

from scipy.special import expit
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
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=100)

#train_test_split is a function from sklearn that does the splitting for us

#Now, X_train has 80% of the raw/normalized data, X_test has 20% of the raw/normalized data
#y_train has 80% of the corresponding labels, y_test has the other 20%
#print(y_test)

def extract_features(segment):
    # Calculate the desired statistical features
    min_val = np.min(segment)
    max_val = np.max(segment)
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    variance_val = np.var(segment)
    skewness_val = scipy.stats.skew(segment)
    kurtosis_val = scipy.stats.kurtosis(segment)
    rms_val = np.sqrt(np.mean(np.square(segment)))

    # Return them as a flat list (no other dimensions)
    return [min_val, max_val, mean_val, std_val, variance_val, skewness_val, kurtosis_val, rms_val]

# Apply feature extraction to the dataset (X_train) and store them in a new matrix (X_train_features)
X_train_features = np.array([extract_features(segment) for segment in X_train])
X_test_features = np.array([extract_features(segment) for segment in X_test])


import pandas as pd

# Create a DataFrame from the feature matrix
#df_train_features = pd.DataFrame(X_train_features, columns=['min', 'max', 'mean', 'std', 'variance', 'skewness', 'kurtosis', 'energy'])
#df_train_features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

pca = PCA(n_components=3)  # You can experiment with the number of components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=3)
X_train_selected = selector.fit_transform(X_train_pca, y_train)
X_test_selected = selector.transform(X_test_pca)

model = SVC(C=100, kernel = 'rbf', decision_function_shape='ovo', probability = True, random_state = 42)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
svm_probabilities = model.predict_proba(X_test_pca)
model.score(X_test_pca, y_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef
print(classification_report(y_test, y_pred))
print("")
print(confusion_matrix(y_test, y_pred))
print("")
print(accuracy_score(y_test, y_pred))
print("")

svm_results_df = pd.DataFrame({
    'Predictions': y_pred,
    'Probabilities': [list(probs) for probs in svm_probabilities]
})

# Save the DataFrame to a pickle file for later use
svm_results_df.to_pickle('svm_pred.pkl')

# Optionally, print to check the format
#print(svm_results_df.head())

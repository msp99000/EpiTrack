import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

st.subheader("Bayesian Inference and Probabilistic Models")

# Sample EEG data (replace with actual data)
X_sample = np.random.randn(30, 100)  # Replace with your actual features
y_sample = np.random.randint(0, 2, size=30)  # Replace with your actual labels

# Function to apply Bayesian Ridge Regression
def apply_bayesian_inference(X, y):
    model = BayesianRidge()
    model.fit(X, y)
    return model

if st.button('Apply Bayesian Inference'):
    bayesian_model = apply_bayesian_inference(X_sample, y_sample)
    
    # Get predictions and the standard deviation of the predictions
    mean_prediction, std_prediction = bayesian_model.predict(X_sample, return_std=True)
    
    # Display prediction for the selected sample index
    sample_index = 13  # Adjust or use a slider to select index dynamically
    st.write(f"**Bayesian Prediction for Sample {sample_index}: {int(mean_prediction[sample_index])}**")

    # Display confidence intervals for the predictions
    st.write(f"**Prediction Mean:** {mean_prediction[sample_index]:.2f}")
    st.write(f"**Prediction Uncertainty (±1 Std):** {std_prediction[sample_index]:.2f}")

    # Plot the posterior distribution of the prediction
    st.write("### Posterior Distribution of the Prediction")
    fig, ax = plt.subplots(facecolor='#1f1f2e')
    ax.set_facecolor('#1f1f2e')
    x_vals = np.linspace(mean_prediction[sample_index] - 3 * std_prediction[sample_index],
                         mean_prediction[sample_index] + 3 * std_prediction[sample_index], 100)
    y_vals = (1 / (std_prediction[sample_index] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean_prediction[sample_index]) / std_prediction[sample_index])**2)
    
    ax.plot(x_vals, y_vals, color='cyan')
    ax.fill_between(x_vals, y_vals, alpha=0.2, color='cyan')
    ax.set_title("Posterior Distribution", color='white')
    ax.set_xlabel("Predicted Value", color='white')
    ax.set_ylabel("Density", color='white')
    ax.tick_params(colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    st.pyplot(fig)

    # Additional plot: Confidence intervals across all samples
    st.write("### Confidence Intervals for All Predictions")
    fig, ax = plt.subplots(facecolor='#1f1f2e')
    ax.set_facecolor('#1f1f2e')
    ax.errorbar(range(len(mean_prediction)), mean_prediction, yerr=std_prediction, fmt='o', color='skyblue', ecolor='lightgray', elinewidth=2, capsize=4)
    ax.set_title("Bayesian Prediction Confidence Intervals", color='white')
    ax.set_xlabel("Sample Index", color='white')
    ax.set_ylabel("Predicted Value", color='white')
    ax.tick_params(colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    st.pyplot(fig)

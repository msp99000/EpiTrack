import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import welch
import pandas as pd
import numpy as np

# Custom CSS for better styling
st.markdown("""
    <style>
        .metric-box {
            background-color: #1f1f2e;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .metric-box h4 {
            color: #ffb400;
            font-family: 'Lato', sans-serif;
        }
        .stPlot {
            margin-top: -20px;
        }
    </style>
""", unsafe_allow_html=True)

# Display Stationarity Tests
st.subheader("Time-Series Analysis and Stationarity Tests")
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='metric-box'><h4>ADF Test</h4>", unsafe_allow_html=True)
    adf_result = adfuller(np.random.randn(100))  # Replace with your data
    st.markdown(f"ADF Statistic: **{adf_result[0]:.2f}**")
    st.markdown(f"p-value: **{adf_result[1]:.2e}**")
    st.markdown(f"Critical Values: {adf_result[4]}</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-box'><h4>KPSS Test</h4>", unsafe_allow_html=True)
    kpss_result = kpss(np.random.randn(100), regression='c')  # Replace with your data
    st.markdown(f"KPSS Statistic: **{kpss_result[0]:.2f}**")
    st.markdown(f"p-value: **{kpss_result[1]:.2e}**")
    st.markdown(f"Critical Values: {kpss_result[3]}</div>", unsafe_allow_html=True)

# Frequency Domain Analysis with Styled Plot
st.subheader("Frequency Domain Analysis")
freqs, psd = welch(np.random.randn(100), fs=256)  # Replace with your data

fig, ax = plt.subplots(facecolor='#1f1f2e')
ax.set_facecolor('#1f1f2e')
sns.lineplot(x=freqs, y=psd, color='skyblue', ax=ax)
ax.set_title("Power Spectral Density", color='white')
ax.set_xlabel("Frequency (Hz)", color='white')
ax.set_ylabel("Power Spectral Density", color='white')
ax.tick_params(colors='white')
st.pyplot(fig)

# Statistical Feature Extraction
st.subheader("Statistical Feature Extraction")
mean_val = np.mean(np.random.randn(100))  # Replace with your data
variance_val = np.var(np.random.randn(100))  # Replace with your data
st.markdown(f"**Mean:** {mean_val:.2f}")
st.markdown(f"**Variance:** {variance_val:.2f}")

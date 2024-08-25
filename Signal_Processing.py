import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch
import seaborn as sns

st.subheader("Signal Processing and Noise Reduction")

# Function to apply band-pass filtering
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=256, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Function to calculate SNR (Signal-to-Noise Ratio)
def calculate_snr(signal):
    mean_signal = np.mean(signal)
    std_noise = np.std(signal)
    snr = mean_signal / std_noise
    return snr

# Sample EEG data (replace with actual data)
data = np.random.randn(1024)
sample_index = st.slider('Select Test Sample Index for Signal Processing', 0, 29, 13)

# Display original signal metrics
st.write("### Original Signal Metrics:")
st.write(f"Mean: {np.mean(data):.2f}")
st.write(f"Standard Deviation: {np.std(data):.2f}")
st.write(f"Signal-to-Noise Ratio (SNR): {calculate_snr(data):.2f}")

if st.button('Apply Band-Pass Filter'):
    filtered_data = bandpass_filter(data)

    # Display filtered signal metrics
    st.write("### Filtered Signal Metrics:")
    st.write(f"Mean: {np.mean(filtered_data):.2f}")
    st.write(f"Standard Deviation: {np.std(filtered_data):.2f}")
    st.write(f"Signal-to-Noise Ratio (SNR): {calculate_snr(filtered_data):.2f}")

    # Plot original vs. filtered signal
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#1f1f2e')
    ax1.set_facecolor('#1f1f2e')
    ax2.set_facecolor('#1f1f2e')

    ax1.plot(data, color='yellow')
    ax1.set_title("Original EEG Signal", color='white')
    ax1.set_xlabel("Datapoint (0-1024)", color='white')
    ax1.set_ylabel("Voltage", color='white')
    ax1.tick_params(colors='white')
    ax1.spines['top'].set_color('white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')

    ax2.plot(filtered_data, color='cyan')
    ax2.set_title(f"Filtered EEG Recording: {sample_index}", color='white')
    ax2.set_xlabel("Datapoint (0-1024)", color='white')
    ax2.set_ylabel("Voltage", color='white')
    ax2.tick_params(colors='white')
    ax2.spines['top'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['right'].set_color('white')

    st.pyplot(fig)

    # Frequency domain analysis before and after filtering
    st.subheader("Frequency Domain Analysis")
    freqs, psd_original = welch(data, fs=256)
    _, psd_filtered = welch(filtered_data, fs=256)

    fig, ax = plt.subplots(facecolor='#1f1f2e')
    ax.set_facecolor('#1f1f2e')
    sns.lineplot(x=freqs, y=psd_original, color='yellow', label='Original Signal', ax=ax)
    sns.lineplot(x=freqs, y=psd_filtered, color='cyan', label='Filtered Signal', ax=ax)
    ax.set_title("Power Spectral Density (Original vs. Filtered)", color='white')
    ax.set_xlabel("Frequency (Hz)", color='white')
    ax.set_ylabel("Power Spectral Density", color='white')
    ax.legend(loc='upper right')
    ax.tick_params(colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    st.pyplot(fig)

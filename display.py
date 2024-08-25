import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score

from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import butter, filtfilt, welch
from scipy.stats import f_oneway
from lifelines import KaplanMeierFitter

st.set_page_config(layout="centered", page_icon="🌐", page_title="EpiTrack")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.selectbox(
    "Choose a section",
    [
        "EEG Fusion Prediction",
        "EDA",
        "Signal Processing",
        "Bayesian Inference",
        "Survival Analysis",
        "Advanced Data Analysis and Model Insights",
        "Statistical Hypothesis Testing",
    ],
)

# Assume X, y are already loaded and preprocessed
# Placeholder example data for illustration purposes:
X = np.random.rand(30, 1000)  # Replace this with your EEG data
y = np.random.randint(0, 3, 30)  # Replace this with your labels

# Load y_test (actual labels for evaluation)
with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)


# Load results from individual models (for fusion)
def load_results(file_path):
    return pd.read_pickle(file_path)


models = ["rf", "ada"]  # Add your other models if needed
results_dfs = {model: load_results(f"{model}_pred.pkl") for model in models}


# Function for EEG fusion prediction
def calculate_and_display_fusion_prediction(index, results_dfs):
    class_labels = {0: "Preictal", 1: "Interictal", 2: "Ictal"}
    cumulative_probabilities = [0] * len(class_labels)

    for df in results_dfs.values():
        probabilities = df.iloc[index]["Probabilities"]
        cumulative_probabilities = [
            sum(x) for x in zip(cumulative_probabilities, probabilities)
        ]

    averaged_probabilities = [
        prob / len(results_dfs) for prob in cumulative_probabilities
    ]
    final_predicted_class_index = averaged_probabilities.index(
        max(averaged_probabilities)
    )
    return final_predicted_class_index, averaged_probabilities


def display_fusion_prediction(index, results_dfs):
    predicted_class_index, averaged_probabilities = (
        calculate_and_display_fusion_prediction(index, results_dfs)
    )

    class_labels = {0: "Preictal", 1: "Interictal", 2: "Ictal"}
    correct_label = class_labels[y_test[index]]
    predicted_class = class_labels[predicted_class_index]
    averaged_probabilities_percent = [prob * 100 for prob in averaged_probabilities]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div style='font-size: 20px; font-family: serif; color: white;'>Correct Label: <strong>{correct_label}</strong></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='font-size: 20px; font-family: serif; color: white;'>Fusion Prediction: <strong>{predicted_class}</strong></div>",
            unsafe_allow_html=True,
        )
    with col3:
        fusion_predictions = [
            calculate_and_display_fusion_prediction(i, results_dfs)[0]
            for i in range(30)
        ]
        accuracy = accuracy_score(y_test[:30], fusion_predictions)
        st.markdown(
            f"<div style='font-size: 20px; font-family: serif; color: white;'>Fusion Accuracy: {accuracy * 100:.2f}%</div>",
            unsafe_allow_html=True,
        )

    st.title("")  # Add a blank title for vertical space

    prob_df = pd.DataFrame(
        {
            "Class": list(class_labels.values()),
            "Probability (%)": averaged_probabilities_percent,
        }
    )

    fig, ax = plt.subplots(facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    sns.barplot(x="Probability (%)", y="Class", data=prob_df, palette="coolwarm", ax=ax)
    ax.set_ylabel("")
    ax.set_title("Fusion Prediction Probabilities", color="white", fontname="serif")
    ax.set_xlabel("Probability (%)", color="white", fontname="serif")
    ax.set_yticklabels(prob_df["Class"], color="white", fontname="serif")
    ax.tick_params(colors="white")
    ax.spines["top"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    st.pyplot(fig)


# ---- EEG Fusion Prediction (Original Section) ----
if section == "EEG Fusion Prediction":
    st.header("EEG Fusion Prediction and Visualization")

    st.markdown(
        """
    **Overview:**  
    This section visualizes a raw EEG recording and predicts its classification using a fusion model of five different machine learning algorithms (CNN, KNN, RF, SVM, and AdaBoost).  
    The fusion model aggregates the predictions from each of these models using probability averaging to make the final prediction.  
    This approach helps improve robustness by combining multiple models’ strengths while mitigating their individual weaknesses.
    """
    )

    if "X" in globals() and "y" in globals():
        sample_index = st.slider("Select Test Sample Index", 0, 29, 0)

        if st.button("Show EEG Sample & Fusion Prediction"):
            fig, ax = plt.subplots(facecolor="#0D1117")
            ax.set_facecolor("#0D1117")
            ax.plot(X[sample_index], color="yellow")
            ax.set_title(
                f"EEG Recording: {sample_index}",
                fontdict={"fontname": "serif", "fontsize": 14, "color": "white"},
            )
            ax.set_xlabel(
                "Datapoint (0-1024)",
                fontdict={"fontname": "serif", "fontsize": 14, "color": "white"},
            )
            ax.set_ylabel(
                "Voltage",
                fontdict={"fontname": "serif", "fontsize": 14, "color": "white"},
            )
            ax.tick_params(colors="white")
            st.pyplot(fig)

            st.title("")  # Add a blank title for vertical space

            display_fusion_prediction(sample_index, results_dfs)

            # Conclusive remark
            st.markdown(
                """
            **Conclusion:**  
            The fusion model provides a robust classification by leveraging the strengths of multiple machine learning algorithms.  
            The prediction accuracy and probability distribution offer a comprehensive understanding of the model's performance for each EEG sample.
            """
            )
    else:
        st.write("Please provide the correct path to the EEG dataset.")

# ---- Exploratory Data Analysis (EDA) ----
elif section == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.markdown(
        """
    **Overview:**  
    This section helps analyze the EEG data through exploratory techniques.  
    Key analyses include:
    - **Stationarity Tests:** To check whether the EEG signal has a constant mean and variance over time using the ADF and KPSS tests.
    - **Frequency Domain Analysis:** To examine the distribution of power across various frequency bands using the Power Spectral Density (PSD) estimate.
    - **Statistical Feature Extraction:** To extract key summary statistics (e.g., mean, variance) that can be used for further modeling.
    """
    )

    def stationarity_tests(data):
        adf_result = adfuller(data)
        st.write("ADF Statistic:", adf_result[0])
        st.write("p-value:", adf_result[1])
        # st.write("Critical Values:", adf_result[4])

        kpss_result = kpss(data, regression="c")
        st.write("\nKPSS Statistic:", kpss_result[0])
        st.write("p-value:", kpss_result[1])
        st.write("Critical Values:", kpss_result[3])

    def frequency_domain_analysis(data, fs=256):
        freqs, psd = welch(data, fs=fs)
        plt.figure(figsize=(10, 6))
        plt.semilogy(freqs, psd, color="blue")
        plt.title("Power Spectral Density")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (V^2/Hz)")
        st.pyplot(fig)

    def extract_statistical_features(data):
        mean_val = np.mean(data)
        variance_val = np.var(data)
        st.write(f"Mean: {mean_val}")
        st.write(f"Variance: {variance_val}")

    sample_index = st.slider("Select Test Sample Index for EDA", 0, 29, 0)
    if st.button("Perform EDA"):
        st.subheader("Time-Series Analysis and Stationarity Tests")
        stationarity_tests(X[sample_index])

        st.subheader("Frequency Domain Analysis")
        frequency_domain_analysis(X[sample_index])

        st.subheader("Statistical Feature Extraction")
        extract_statistical_features(X[sample_index])

        # Conclusive remark
        st.markdown(
            """
        **Conclusion:**  
        The EDA provides insights into the underlying characteristics of the EEG data, such as stationarity and frequency content.  
        These insights are crucial for understanding the nature of the EEG signals and for selecting appropriate modeling techniques.
        """
        )

# ---- Signal Processing and Noise Reduction ----
elif section == "Signal Processing":
    st.header("Signal Processing and Noise Reduction")

    st.markdown(
        """
    **Overview:**  
    This section focuses on signal processing techniques, especially filtering to reduce noise.  
    Here, a band-pass filter is applied to retain frequencies in the range of interest (e.g., 0.5–50 Hz) while filtering out noise.  
    The filtered signal is then visualized to show the effect of noise reduction.
    """
    )

    def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=256, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    sample_index = st.slider("Select Test Sample Index for Signal Processing", 0, 29, 0)
    if st.button("Apply Band-Pass Filter"):
        filtered_data = bandpass_filter(X[sample_index])
        st.write("Filtered Data (First 10 Points):", filtered_data[:10])

        fig, ax = plt.subplots(facecolor="#0D1117")
        ax.set_facecolor("#0D1117")
        ax.plot(filtered_data, color="cyan")
        ax.set_title(
            f"Filtered EEG Recording: {sample_index}",
            fontdict={"fontname": "serif", "fontsize": 14, "color": "white"},
        )
        ax.set_xlabel(
            "Datapoint (0-1024)",
            fontdict={"fontname": "serif", "fontsize": 14, "color": "white"},
        )
        ax.set_ylabel(
            "Voltage", fontdict={"fontname": "serif", "fontsize": 14, "color": "white"}
        )
        ax.tick_params(colors="white")
        st.pyplot(fig)

        # Conclusive remark
        st.markdown(
            """
        **Conclusion:**  
        Signal processing, particularly noise reduction, enhances the quality of the EEG signals, making them more suitable for analysis and modeling.  
        The band-pass filter effectively isolates the frequencies relevant for seizure detection while minimizing artifacts.
        """
        )

# ---- Bayesian Inference and Probabilistic Models ----
elif section == "Bayesian Inference":
    st.header("Bayesian Inference and Probabilistic Models")

    st.markdown(
        """
    **Overview:**  
    Bayesian inference allows us to update the probability of a hypothesis as more evidence or data becomes available.  
    In this section, we apply a Bayesian Ridge model to the EEG data and predict the class of a selected sample.  
    We also visualize the posterior distribution to show the uncertainty in the prediction.
    """
    )

    def apply_bayesian_inference(X, y):
        model = BayesianRidge()
        model.fit(X, y)
        return model

    sample_index = st.slider("Select Sample Index for Bayesian Inference", 0, 29, 0)
    if st.button("Apply Bayesian Inference"):
        bayesian_model = apply_bayesian_inference(X, y)
        prediction = bayesian_model.predict([X[sample_index]])
        st.write(f"Bayesian Prediction for Sample {sample_index}: {prediction[0]}")

        # Display posterior distribution
        st.write(f"**Prediction Mean:** {prediction[0]:.2f}")
        st.write(f"**Prediction Uncertainty (±1 Std):** {bayesian_model.alpha_:.2f}")

        # Visualize posterior distribution
        fig, ax = plt.subplots(facecolor="#1f1f2e")
        ax.set_facecolor("#1f1f2e")
        x_vals = np.linspace(
            prediction[0] - 3 * bayesian_model.alpha_,
            prediction[0] + 3 * bayesian_model.alpha_,
            100,
        )
        y_vals = (1 / (bayesian_model.alpha_ * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_vals - prediction[0]) / bayesian_model.alpha_) ** 2
        )
        ax.plot(x_vals, y_vals, color="cyan")
        ax.fill_between(x_vals, y_vals, alpha=0.2, color="cyan")
        ax.set_title("Posterior Distribution", color="white")
        ax.set_xlabel("Predicted Value", color="white")
        ax.set_ylabel("Density", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

        # Conclusive remark
        st.markdown(
            """
        **Conclusion:**  
        Bayesian inference provides a probabilistic prediction that incorporates uncertainty.  
        This is particularly valuable when interpreting model outputs for medical applications, where understanding prediction confidence is crucial.
        """
        )

# ---- Survival Analysis ----
elif section == "Survival Analysis":
    st.header("Survival Analysis")

    st.markdown(
        """
    **Overview:**  
    Survival analysis is relevant when studying the time until a particular event occurs, such as the onset of a seizure.  
    In the context of EEG data analysis, we can model the time between seizure events to gain insights into seizure patterns.  
    This analysis is particularly valuable in understanding how long a patient remains seizure-free and predicting the likelihood of future seizures based on historical data.

    In this section, we perform Kaplan-Meier analysis to estimate the probability of remaining seizure-free over time.  
    The survival curve visualizes this probability, allowing us to better understand the temporal patterns associated with seizure occurrences.
    """
    )

    seizure_times = np.random.exponential(scale=100, size=len(y))
    event_occurred = (y == 2).astype(int)

    def perform_kaplan_meier_analysis(seizure_times, event_occurred):
        kmf = KaplanMeierFitter()
        kmf.fit(seizure_times, event_occurred)
        fig, ax = plt.subplots(facecolor="#1f1f2e")
        ax.set_facecolor("#1f1f2e")
        kmf.plot_survival_function(ax=ax, color="skyblue", ci_show=True, alpha=0.8)

        # Median survival time
        median_survival_time = kmf.median_survival_time_
        ax.axhline(
            y=0.5,
            color="red",
            linestyle="--",
            label=f"Median Survival Time: {median_survival_time:.2f}",
        )

        ax.set_title("Kaplan-Meier Estimate", color="white")
        ax.set_xlabel("Timeline", color="white")
        ax.set_ylabel("Survival Probability", color="white")
        ax.tick_params(colors="white")
        ax.legend(loc="best", fontsize="medium")
        st.pyplot(fig)

    if st.button("Perform Kaplan-Meier Analysis"):
        perform_kaplan_meier_analysis(seizure_times, event_occurred)

        # Conclusive remark
        st.markdown(
            """
        **Conclusion:**  
        Kaplan-Meier analysis provides a visual representation of the probability of remaining seizure-free over time.  
        This information is crucial for both clinicians and patients in managing epilepsy and planning treatment strategies.
        """
        )

# ---- Advanced Data Analysis and Model Insights ----
elif section == "Advanced Data Analysis and Model Insights":
    st.header("Advanced Data Analysis and Model Insights")

    st.markdown(
        """
    **Overview:**  
    This section provides comprehensive insights into the data and model performance through various visualizations, including correlation analysis, time-series exploration, and hyperparameter tuning evaluations. Currently modifying and testing this section as it often causes app crashes due to TLEs 
    """
    )

    def plot_correlation_heatmap(X):
        # Select the first 10 features
        df = pd.DataFrame(X).iloc[:, :10]
        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)

    def plot_time_series_analysis(X, y, sample_index):
        plt.figure(figsize=(12, 6))
        plt.plot(X[sample_index])
        plt.title(f"Time-Series Analysis for Sample Index {sample_index}")
        plt.xlabel("Time")
        plt.ylabel("EEG Signal")
        plt.grid(True, linestyle="--", alpha=0.7)
        st.pyplot()

    def plot_hyperparameter_heatmap(hyperparam_combinations, validation_scores):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            validation_scores,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=hyperparam_combinations["param_1"],
            yticklabels=hyperparam_combinations["param_2"],
            ax=ax,
        )
        ax.set_title("Hyperparameter Tuning Heatmap")
        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 2")
        st.pyplot(fig)

    def plot_validation_score_lineplot(hyperparams, validation_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(hyperparams, validation_scores, marker="o")
        plt.title("Effect of Hyperparameters on Validation Scores")
        plt.xlabel("Hyperparameter Value")
        plt.ylabel("Validation Score")
        plt.grid(True, linestyle="--", alpha=0.7)
        st.pyplot()

    # Sample index for time-series analysis
    sample_index = st.slider(
        "Select Sample Index for Time-Series Analysis", 0, len(X) - 1, 0
    )

    if st.button("Analyze Data and Model Insights"):
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        plot_correlation_heatmap(X)

        # Time-Series Analysis
        st.subheader("Time-Series Analysis")
        plot_time_series_analysis(X, y, sample_index)

        # Hyperparameter Tuning Heatmap
        st.subheader("Hyperparameter Tuning Heatmap")
        # Example data (replace with your actual hyperparameter combinations and scores)
        hyperparam_combinations = {"param_1": [0.1, 0.2, 0.3], "param_2": [1, 2, 3]}
        validation_scores = np.array(
            [[0.7, 0.75, 0.78], [0.8, 0.82, 0.85], [0.76, 0.79, 0.81]]
        )
        plot_hyperparameter_heatmap(hyperparam_combinations, validation_scores)

        # Validation Score Line Plot
        st.subheader("Validation Score Line Plot")
        # Example data (replace with your actual hyperparameter values and validation scores)
        hyperparams = [0.1, 0.2, 0.3, 0.4, 0.5]
        validation_scores = [0.7, 0.75, 0.8, 0.78, 0.76]
        plot_validation_score_lineplot(hyperparams, validation_scores)

        # Conclusive remark
        st.markdown(
            """
        **Conclusion:**  
        The visualizations provided here offer insights into data relationships, trends in time-series signals, and the impact of hyperparameter tuning on model performance.  
        Understanding these factors is essential for improving model accuracy and ensuring the stability of predictions across different configurations.
        """
        )

# ---- Statistical Hypothesis Testing ----
elif section == "Statistical Hypothesis Testing":
    st.header("Statistical Hypothesis Testing")

    st.markdown(
        """
    **Overview:**  
    Hypothesis testing helps us determine whether there is enough evidence to reject a null hypothesis.  
    In this section, we perform an ANOVA (Analysis of Variance) test to compare the means of the three classes (preictal, interictal, ictal) and determine if they are significantly different.  
    We also visualize the distribution of values in each class, along with the F-statistic and p-value.
    """
    )

    def perform_anova_test(X, y):
        preictal = X[y == 0].flatten()  # Flatten the array to ensure 1D
        interictal = X[y == 1].flatten()  # Flatten the array to ensure 1D
        ictal = X[y == 2].flatten()  # Flatten the array to ensure 1D

        # ANOVA Test
        f_stat, p_value = f_oneway(preictal, interictal, ictal)

        # Create a DataFrame for visualization
        data = {
            "Class": ["Preictal"] * len(preictal)
            + ["Interictal"] * len(interictal)
            + ["Ictal"] * len(ictal),
            "Values": np.concatenate([preictal, interictal, ictal]),
        }
        df = pd.DataFrame(data)

        # Plot the distribution of values in each class
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Class", y="Values", data=df, palette="coolwarm")
        plt.title(
            f"ANOVA Test Result\nF-Statistic: {f_stat:.2f}, p-value: {p_value:.4f}"
        )
        plt.xlabel("EEG Class")
        plt.ylabel("Values")
        plt.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

        # Display interpretation
        st.write(f"**F-Statistic:** {f_stat:.2f}")
        st.write(f"**p-value:** {p_value:.4f}")
        if p_value < 0.05:
            st.write(
                "The p-value is below 0.05, indicating that the means of the three classes are significantly different."
            )
        else:
            st.write(
                "The p-value is above 0.05, indicating that the means of the three classes are not significantly different."
            )

    sample_index = st.slider("Select Sample Index for Hypothesis Testing", 0, 29, 0)
    if st.button("Perform ANOVA"):
        perform_anova_test(X, y)

        # Conclusive remark
        st.markdown(
            """
        **Conclusion:**  
        The ANOVA test compares the means of the preictal, interictal, and ictal classes to assess if there are statistically significant differences.  
        This helps validate whether the classes are distinct, which is important for ensuring the effectiveness of classification models.
        The visualization above gives a clearer picture of the class differences and how they relate to the ANOVA results.
        """
        )

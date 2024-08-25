"""
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

def perform_anova_test(X, y):
    preictal = X[y == 0]
    interictal = X[y == 1]
    ictal = X[y == 2]

    # ANOVA Test
    f_stat, p_value = f_oneway(preictal, interictal, ictal)

    # Create a DataFrame for visualization
    data = {
        'Class': ['Preictal'] * len(preictal) + ['Interictal'] * len(interictal) + ['Ictal'] * len(ictal),
        'Values': np.concatenate([preictal, interictal, ictal])
    }
    df = pd.DataFrame(data)

    # Plot the distribution of values in each class
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Values', data=df, palette='coolwarm')
    plt.title(f"ANOVA Test Result\nF-Statistic: {f_stat:.2f}, p-value: {p_value:.4f}")
    plt.xlabel("EEG Class")
    plt.ylabel("Values")
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

    # Display interpretation
    st.write(f"**F-Statistic:** {f_stat:.2f}")
    st.write(f"**p-value:** {p_value:.4f}")
    if p_value < 0.05:
        st.write("The p-value is below 0.05, indicating that the means of the three classes are significantly different.")
    else:
        st.write("The p-value is above 0.05, indicating that the means of the three classes are not significantly different.")

"""

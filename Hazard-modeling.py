import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import plotly.express as px

st.subheader("Survival Analysis")

# Simulated survival data (replace with actual data)
np.random.seed(42)
seizure_times = np.random.exponential(scale=100, size=100)  # Replace with your timeline data
event_occurred = np.random.binomial(1, 0.5, size=100)  # Replace with actual event data (e.g., seizure events)

# Kaplan-Meier Fitting
kmf = KaplanMeierFitter()
kmf.fit(seizure_times, event_observed=event_occurred)

if st.button('Perform Kaplan-Meier Analysis'):
    # Median survival time
    median_survival_time = kmf.median_survival_time_
    
    # Customized Kaplan-Meier plot
    st.write("### Kaplan-Meier Survival Curve with Median and Confidence Intervals")
    
    fig, ax = plt.subplots(facecolor='#1f1f2e')
    ax.set_facecolor('#1f1f2e')
    kmf.plot_survival_function(ax=ax, color='skyblue', ci_show=True, alpha=0.8)
    
    ax.axhline(y=0.5, color='red', linestyle='--', label=f"Median Survival Time: {median_survival_time:.2f}")
    ax.set_title("Kaplan-Meier Estimate", color='white')
    ax.set_xlabel("Timeline", color='white')
    ax.set_ylabel("Survival Probability", color='white')
    ax.tick_params(colors='white')
    ax.legend(loc='best', fontsize='medium')
    
    # Highlighting the median survival time
    ax.annotate(f'Median: {median_survival_time:.2f}', xy=(median_survival_time, 0.5), xycoords='data',
                xytext=(median_survival_time + 50, 0.6), textcoords='data',
                arrowprops=dict(facecolor='white', shrink=0.05), color='white', fontsize=10)
    
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    st.pyplot(fig)

    # Adding at-risk counts (optional)
    st.write("### At-Risk Counts")
    add_at_risk_counts(kmf, ax=ax)

    st.pyplot(fig)

# Optional: Interactive Kaplan-Meier Plot using Plotly
if st.checkbox("View Interactive Kaplan-Meier Plot"):
    survival_df = kmf.survival_function_.reset_index()
    fig = px.line(survival_df, x='timeline', y='KM_estimate', title="Interactive Kaplan-Meier Plot")
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Survival Probability",
        template="plotly_dark"
    )
    st.plotly_chart(fig)

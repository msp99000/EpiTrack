import shap
import streamlit as st
import matplotlib.pyplot as plt

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

def visualize_shap_values(model, X, sample_index):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Force Plot
    st.write("### SHAP Force Plot for the Selected Sample")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][sample_index], X[sample_index]))

    # Global Feature Importance
    st.write("### Global Feature Importance")
    fig, ax = plt.subplots(facecolor='#1f1f2e')
    ax.set_facecolor('#1f1f2e')
    shap.summary_plot(shap_values[1], X, plot_type="bar")
    st.pyplot(fig)

    # Detailed SHAP Waterfall Plot
    st.write("### SHAP Waterfall Plot for Detailed Explanation")
    fig, ax = plt.subplots(facecolor='#1f1f2e')
    ax.set_facecolor('#1f1f2e')
    shap.waterfall_plot(shap.Explanation(values=shap_values[1][sample_index], base_values=explainer.expected_value[1], data=X[sample_index], feature_names=[f'Feature {i}' for i in range(X.shape[1])]))
    st.pyplot(fig)

import shap
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def visualize_shap_values(model, X, sample_index):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Force Plot
    st.write("### SHAP Force Plot for the Selected Sample")
    st_shap(
        shap.force_plot(
            explainer.expected_value, shap_values[sample_index], X.iloc[sample_index]
        )
    )

    # Global Feature Importance
    st.write("### Global Feature Importance")
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot()  # Display the plot directly

    # Detailed SHAP Waterfall Plot
    st.write("### SHAP Waterfall Plot for Detailed Explanation")
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_index],
            base_values=explainer.expected_value,
            data=X.iloc[sample_index],
            feature_names=[f"Feature {i}" for i in range(X.shape[1])],
        )
    )
    st.pyplot()  # Display the plot directly

import numpy as np
import pandas as pd
import streamlit as st

from utils.state import get_dataset, save_plot
from utils.ui import footer, init_page, kpi_cards, page_header, section
from visualization.plots import (
    calibration_plot_figure,
    calibration_plot_interactive,
    confusion_matrix_figure,
    confusion_matrix_interactive,
    roc_curve_figure,
    roc_curve_interactive,
)


init_page("Visualization")
page_header(
    "Model Visualization",
    "Review interactive model evaluation plots from the latest classification model.",
)
df = get_dataset()
if df is None:
    st.info("Upload a dataset first.")
    footer()
    st.stop()

holdout = st.session_state.get("last_ml_holdout")
task = st.session_state.get("last_ml_task")
model = st.session_state.get("last_ml_model")

if holdout is None or task != "classification" or model is None:
    st.warning("Train a classification model first to enable ROC, confusion matrix, and calibration plots.")
    footer()
    st.stop()

x_test, y_test, predictions = holdout
y_test = pd.Series(y_test)
predictions = pd.Series(predictions)
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(x_test)[:, 1]
else:
    y_prob = predictions

kpi_cards(
    [
        ("Holdout Rows", f"{len(y_test):,}", "Evaluation sample"),
        ("Classes", f"{y_test.nunique():,}", "Observed target classes"),
        ("Predictions", f"{len(predictions):,}", "Generated labels"),
        ("Probability Output", "Yes" if hasattr(model, "predict_proba") else "No", "Needed for ROC/calibration"),
    ]
)

if y_test.nunique() != 2:
    st.warning("ROC and calibration plots require a binary classification target.")
else:
    y_encoded = y_test if np.issubdtype(y_test.dtype, np.number) else y_test.astype("category").cat.codes
    section("ROC Curve")
    st.plotly_chart(roc_curve_interactive(y_encoded, y_prob), use_container_width=True)
    section("Calibration")
    st.plotly_chart(calibration_plot_interactive(y_encoded, y_prob), use_container_width=True)
    roc_fig = roc_curve_figure(y_encoded, y_prob)
    cal_fig = calibration_plot_figure(y_encoded, y_prob)
    save_plot("ROC Curve", roc_fig)
    save_plot("Calibration Plot", cal_fig)

section("Confusion Matrix")
st.plotly_chart(confusion_matrix_interactive(y_test, predictions), use_container_width=True)
cm_fig = confusion_matrix_figure(y_test, predictions)
save_plot("Confusion Matrix", cm_fig)

footer()

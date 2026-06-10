import pandas as pd
import streamlit as st

from analysis.regression import linear_regression, logistic_regression
from utils.data_summary import binary_columns, continuous_columns
from utils.state import get_dataset, save_result
from utils.ui import data_table, footer, init_page, kpi_cards, page_header, section
from visualization.plots import confusion_matrix_figure, roc_curve_figure


init_page("Regression")
page_header(
    "Regression Analysis",
    "Fit interpretable linear and logistic models with confidence intervals, odds ratios, p-values, and diagnostics.",
)
df = get_dataset()
if df is None:
    st.info("Upload a dataset first.")
    footer()
    st.stop()

model_type = st.radio("Model", ["Linear regression", "Logistic regression"], horizontal=True)
columns = df.columns.tolist()
continuous = continuous_columns(df)
binary = binary_columns(df)
kpi_cards(
    [
        ("Observations", f"{df.shape[0]:,}", "Before model-wise exclusions"),
        ("Variables", f"{df.shape[1]:,}", "Candidate predictors"),
        ("Continuous Outcomes", f"{len(continuous):,}", "Linear regression candidates"),
        ("Missing Cells", f"{int(df.isna().sum().sum()):,}", "Rows dropped model-wise"),
    ]
)

section("Model Configuration")
outcome_options = continuous if model_type == "Linear regression" else binary
if not outcome_options:
    st.warning(
        "Linear regression requires at least one continuous outcome variable."
        if model_type == "Linear regression"
        else "Logistic regression requires at least one binary outcome variable."
    )
    footer()
    st.stop()
outcome = st.selectbox("Outcome", outcome_options)
predictors = st.multiselect("Predictors", [col for col in columns if col != outcome])

if st.button("Fit model", type="primary", disabled=not predictors):
    try:
        with st.spinner(f"Fitting {model_type}..."):
            if model_type == "Linear regression":
                model, summary, diagnostics = linear_regression(df, outcome, predictors)
            else:
                model, summary, diagnostics, predicted, y = logistic_regression(df, outcome, predictors)
        st.success(f"{model_type} fitted successfully.")

        if model_type == "Linear regression":
            section("Coefficients")
            data_table(summary, height=360)
            section("Model Diagnostics")
            data_table(pd.DataFrame([diagnostics]), height=160)
            save_result("Linear Regression Coefficients", summary)
            save_result("Linear Regression Diagnostics", pd.DataFrame([diagnostics]))
        else:
            section("Coefficients and Odds Ratios")
            data_table(summary, height=420)
            section("Model Diagnostics")
            display_diagnostics = {k: v for k, v in diagnostics.items() if k != "confusion_matrix"}
            data_table(pd.DataFrame([display_diagnostics]), height=160)
            section("Model Evaluation")
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.pyplot(roc_curve_figure(y, predicted))
            with chart_cols[1]:
                st.pyplot(confusion_matrix_figure(y, (predicted >= 0.5).astype(int)))
            save_result("Logistic Regression Coefficients", summary)
            save_result("Logistic Regression Diagnostics", pd.DataFrame([display_diagnostics]))
    except Exception as exc:
        st.error(str(exc))

footer()

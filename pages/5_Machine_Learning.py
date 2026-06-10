import streamlit as st

from analysis.ml_models import available_models, train_model
from utils.data_summary import continuous_columns
from utils.state import get_dataset, save_result
from utils.ui import data_table, footer, init_page, kpi_cards, page_header, section


init_page("Machine Learning")
page_header(
    "Machine Learning",
    "Train classification and regression models with shared preprocessing, cross-validation, and feature importance.",
)
df = get_dataset()
if df is None:
    st.info("Upload a dataset first.")
    footer()
    st.stop()

columns = df.columns.tolist()
task = st.radio("Task", ["classification", "regression"], horizontal=True)
target_options = columns if task == "classification" else continuous_columns(df)

if not target_options:
    st.warning("Regression tasks require at least one continuous target.")
    footer()
    st.stop()

kpi_cards(
    [
        ("Rows", f"{df.shape[0]:,}", "Available examples"),
        ("Features", f"{df.shape[1] - 1:,}", "Potential predictors"),
        ("Continuous Fields", f"{len(continuous_columns(df)):,}", "Regression-ready variables"),
        ("Task", task.title(), "Current modeling mode"),
    ]
)

section("Training Configuration")
target = st.selectbox("Target", target_options)
predictors = st.multiselect("Features", [col for col in columns if col != target])
model_name = st.selectbox("Model", available_models(task))

with st.expander("Available model families"):
    st.write(
        "Classification includes logistic regression, tree ensembles, boosting, SVM, KNN, Naive Bayes, and XGBoost. "
        "Regression includes linear/regularized regression, tree ensembles, boosting, SVM, KNN, and XGBoost."
    )

if st.button("Train model", type="primary", disabled=not predictors):
    try:
        with st.spinner(f"Training {model_name}..."):
            pipeline, metrics, importance, holdout = train_model(df, target, predictors, model_name, task)
        st.session_state["last_ml_model"] = pipeline
        st.session_state["last_ml_holdout"] = holdout
        st.session_state["last_ml_task"] = task
        st.session_state["last_ml_model_name"] = model_name

        st.success(f"{model_name} training completed.")
        section("Cross-validation and Holdout Metrics")
        data_table(metrics, height=180)
        save_result("Machine Learning Metrics", metrics)

        section("Feature Importance")
        if importance.empty:
            st.warning("Feature importance is unavailable for this model type.")
        else:
            data_table(importance, height=420)
            value_column = [col for col in importance.columns if col != "feature"][0]
            st.bar_chart(importance.set_index("feature")[value_column].head(25))
            save_result("Feature Importance", importance)
    except Exception as exc:
        st.error(str(exc))

if not continuous_columns(df):
    st.caption("Regression tasks require a continuous target; classification targets may be numeric or categorical.")

footer()

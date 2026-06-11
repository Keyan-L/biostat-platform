from __future__ import annotations

from typing import Optional, Union

import pandas as pd
import streamlit as st


DATASET_KEY = "dataset"
DATASET_NAME_KEY = "dataset_name"
DATASET_SIGNATURE_KEY = "dataset_signature"
RESULTS_KEY = "analysis_results"
PLOTS_KEY = "exportable_plots"
ML_STATE_KEYS = ["last_ml_model", "last_ml_holdout", "last_ml_task", "last_ml_model_name"]


def set_dataset(df: pd.DataFrame, name: str) -> None:
    previous_df = st.session_state.get(DATASET_KEY)
    previous_name = st.session_state.get(DATASET_NAME_KEY)
    signature = dataset_signature(df)
    previous_signature = st.session_state.get(DATASET_SIGNATURE_KEY)
    dataset_changed = (
        previous_name != name
        or previous_df is None
        or previous_df.shape != df.shape
        or previous_signature != signature
    )
    st.session_state[DATASET_KEY] = df
    st.session_state[DATASET_NAME_KEY] = name
    st.session_state[DATASET_SIGNATURE_KEY] = signature
    if dataset_changed:
        clear_analysis_state()


def clear_analysis_state() -> None:
    st.session_state.pop(RESULTS_KEY, None)
    st.session_state.pop(PLOTS_KEY, None)
    for key in ML_STATE_KEYS:
        st.session_state.pop(key, None)


def dataset_signature(df: pd.DataFrame) -> int:
    return int(pd.util.hash_pandas_object(df, index=True).sum())


def get_dataset() -> Optional[pd.DataFrame]:
    return st.session_state.get(DATASET_KEY)


def get_dataset_name() -> str:
    return st.session_state.get(DATASET_NAME_KEY, "dataset")


def save_result(name: str, result: Union[pd.DataFrame, dict, str]) -> None:
    st.session_state.setdefault(RESULTS_KEY, {})
    st.session_state[RESULTS_KEY][name] = result


def get_results() -> dict:
    return st.session_state.get(RESULTS_KEY, {})


def save_plot(name: str, figure) -> None:
    st.session_state.setdefault(PLOTS_KEY, {})
    st.session_state[PLOTS_KEY][name] = figure


def get_plots() -> dict:
    return st.session_state.get(PLOTS_KEY, {})

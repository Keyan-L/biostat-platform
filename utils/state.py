from __future__ import annotations

import pandas as pd
import streamlit as st


DATASET_KEY = "dataset"
DATASET_NAME_KEY = "dataset_name"
RESULTS_KEY = "analysis_results"
PLOTS_KEY = "exportable_plots"


def set_dataset(df: pd.DataFrame, name: str) -> None:
    st.session_state[DATASET_KEY] = df
    st.session_state[DATASET_NAME_KEY] = name


def get_dataset() -> pd.DataFrame | None:
    return st.session_state.get(DATASET_KEY)


def get_dataset_name() -> str:
    return st.session_state.get(DATASET_NAME_KEY, "dataset")


def save_result(name: str, result: pd.DataFrame | dict | str) -> None:
    st.session_state.setdefault(RESULTS_KEY, {})
    st.session_state[RESULTS_KEY][name] = result


def get_results() -> dict:
    return st.session_state.get(RESULTS_KEY, {})


def save_plot(name: str, figure) -> None:
    st.session_state.setdefault(PLOTS_KEY, {})
    st.session_state[PLOTS_KEY][name] = figure


def get_plots() -> dict:
    return st.session_state.get(PLOTS_KEY, {})

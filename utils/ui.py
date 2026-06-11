from __future__ import annotations

from typing import Optional, Union

import pandas as pd
import streamlit as st

from utils.state import get_dataset, get_dataset_name, get_results


PAGE_LINKS = [
    ("Overview", "app.py", ":material/dashboard:"),
    ("Data Upload", "pages/1_Data_Upload.py", ":material/upload_file:"),
    ("Exploratory Analysis", "pages/2_Exploratory_Data_Analysis.py", ":material/monitoring:"),
    ("Statistical Tests", "pages/3_Statistical_Tests.py", ":material/science:"),
    ("Regression", "pages/4_Regression_Analysis.py", ":material/functions:"),
    ("Machine Learning", "pages/5_Machine_Learning.py", ":material/model_training:"),
    ("Visualization", "pages/6_Visualization.py", ":material/analytics:"),
    ("Reporting", "pages/7_Reporting.py", ":material/picture_as_pdf:"),
]


def init_page(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} | Biostat Analytics",
        page_icon=":material/biotech:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    render_sidebar()


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f7fb;
            --panel: #ffffff;
            --ink: #16212f;
            --muted: #667085;
            --border: #d9e2ec;
            --primary: #126a8a;
            --primary-dark: #0d4f66;
            --accent: #d97706;
            --success: #15803d;
            --danger: #b42318;
        }
        html, body, .stApp {
            background: var(--bg);
            color: var(--ink);
            font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1480px;
        }
        [data-testid="stSidebar"] {
            background: #0d2635;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #dbeafe;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stSidebar"] a {
            border-radius: 8px;
            margin: 0.1rem 0;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        h1 {
            font-size: clamp(1.8rem, 2.2vw, 2.45rem);
            line-height: 1.15;
            margin-bottom: 0.25rem;
        }
        h2 {
            font-size: 1.25rem;
            margin-top: 1.2rem;
        }
        .app-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            padding: 1.1rem 1.25rem;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 12px 30px rgba(15, 35, 52, 0.06);
            margin-bottom: 1rem;
        }
        .eyebrow {
            color: var(--primary);
            text-transform: uppercase;
            font-size: 0.74rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            max-width: 940px;
            margin: 0.25rem 0 0;
        }
        .status-pill {
            white-space: nowrap;
            border-radius: 999px;
            border: 1px solid #b6d7e2;
            color: var(--primary-dark);
            background: #e8f6fa;
            padding: 0.42rem 0.72rem;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .kpi-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            min-height: 104px;
            box-shadow: 0 10px 24px rgba(15, 35, 52, 0.055);
        }
        .kpi-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            color: var(--ink);
            font-size: 1.65rem;
            font-weight: 780;
            line-height: 1.1;
        }
        .kpi-help {
            color: var(--muted);
            margin-top: 0.35rem;
            font-size: 0.84rem;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(15, 35, 52, 0.05);
            margin-bottom: 1rem;
        }
        .section-heading {
            font-size: 1.1rem;
            font-weight: 760;
            margin: 1.1rem 0 0.65rem;
            color: var(--ink);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid #b6d7e2;
            font-weight: 700;
        }
        div.stButton > button[kind="primary"] {
            background: var(--primary);
            border-color: var(--primary);
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        .footer {
            color: var(--muted);
            border-top: 1px solid var(--border);
            padding-top: 0.85rem;
            margin-top: 2rem;
            font-size: 0.84rem;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }
        @media (max-width: 720px) {
            .app-header {
                display: block;
            }
            .status-pill {
                display: inline-block;
                margin-top: 0.75rem;
            }
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    df = get_dataset()
    with st.sidebar:
        st.markdown("### Biostat Analytics")
        st.caption("Clinical analytics workspace")
        st.divider()
        for label, page, icon in PAGE_LINKS:
            st.page_link(page, label=label, icon=icon)
        st.divider()
        if df is None:
            st.info("No dataset loaded")
        else:
            st.success(get_dataset_name())
            st.caption(f"{df.shape[0]:,} rows · {df.shape[1]:,} columns")
        st.caption(f"{len(get_results())} saved result set(s)")


def page_header(title: str, subtitle: str, status: Optional[str] = None) -> None:
    status = status or dataset_status()
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <div class="eyebrow">Biostatistics Analysis Platform</div>
            <h1>{title}</h1>
            <p class="subtitle">{subtitle}</p>
          </div>
          <div class="status-pill">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataset_status() -> str:
    df = get_dataset()
    if df is None:
        return "No dataset"
    return f"{get_dataset_name()} · {df.shape[0]:,} rows"


def kpi_cards(items: list[tuple[str, Union[str, int, float], str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-help">{help_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def section(title: str) -> None:
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)


def data_table(df: pd.DataFrame, height: int = 360) -> None:
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)


def footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <span>Biostat Analytics · Streamlit SaaS dashboard prototype</span>
            <span>Exploratory outputs require study-level validation before formal use</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

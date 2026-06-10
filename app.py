import streamlit as st

from utils.state import get_dataset
from utils.ui import data_table, footer, init_page, kpi_cards, page_header, section


init_page("Overview")
page_header(
    "Analytics Command Center",
    "A streamlined workspace for clinical data review, statistical analysis, predictive modeling, and reproducible reporting.",
)

df = get_dataset()
kpi_cards(
    [
        ("Dataset", "Loaded" if df is not None else "Not loaded", "Current session state"),
        ("Rows", 0 if df is None else f"{df.shape[0]:,}", "Records available"),
        ("Columns", 0 if df is None else f"{df.shape[1]:,}", "Variables detected"),
        ("Missing Cells", 0 if df is None else f"{int(df.isna().sum().sum()):,}", "Data quality signal"),
    ]
)

section("Workflow")
steps = st.columns(4)
steps[0].page_link("pages/1_Data_Upload.py", label="Upload data", icon=":material/upload_file:")
steps[1].page_link("pages/2_Exploratory_Data_Analysis.py", label="Explore variables", icon=":material/monitoring:")
steps[2].page_link("pages/5_Machine_Learning.py", label="Train models", icon=":material/model_training:")
steps[3].page_link("pages/7_Reporting.py", label="Export report", icon=":material/picture_as_pdf:")

if df is None:
    st.info("Upload a CSV or Excel file, or load the included sample dataset on the Data Upload page.")
else:
    st.success("Dataset is available across all pages for the current Streamlit session.")
    section("Dataset Preview")
    data_table(df.head(20), height=420)

footer()

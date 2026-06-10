import streamlit as st

from utils.data_loader import load_tabular_file
from utils.data_summary import missing_value_summary, variable_types
from utils.sample_data import load_sample_dataset
from utils.state import get_dataset, get_dataset_name, set_dataset
from utils.ui import data_table, footer, init_page, kpi_cards, page_header, section


init_page("Data Upload")
page_header(
    "Data Upload",
    "Bring in CSV or Excel data, profile variables, and inspect missingness before analysis.",
)

upload_col, sample_col = st.columns([2, 1])

with upload_col:
    uploaded_file = st.file_uploader("Dataset file", type=["csv", "xlsx", "xls"])

with sample_col:
    st.write("")
    st.write("")
    if st.button("Load sample dataset", type="secondary", use_container_width=True):
        with st.spinner("Loading sample dataset..."):
            sample_df = load_sample_dataset()
            set_dataset(sample_df, "sample_biostat_dataset.csv")
        st.success("Loaded sample_biostat_dataset.csv")

if uploaded_file is not None:
    try:
        with st.spinner("Reading uploaded file..."):
            df = load_tabular_file(uploaded_file)
            set_dataset(df, uploaded_file.name)
        st.success(f"Loaded {uploaded_file.name}")
    except Exception as exc:
        st.error(str(exc))

df = get_dataset()
if df is None:
    st.info("No dataset loaded yet.")
    footer()
    st.stop()

kpi_cards(
    [
        ("File", get_dataset_name(), "Active dataset"),
        ("Rows", f"{df.shape[0]:,}", "Observations"),
        ("Columns", f"{df.shape[1]:,}", "Variables"),
        ("Missing Cells", f"{int(df.isna().sum().sum()):,}", "Total missingness"),
    ]
)

section("Dataset Preview")
data_table(df.head(100), height=420)

section("Variable Types")
data_table(variable_types(df), height=360)

section("Missing Value Summary")
data_table(missing_value_summary(df), height=360)

footer()

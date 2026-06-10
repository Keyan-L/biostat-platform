import pandas as pd
import streamlit as st

from analysis.statistical_tests import (
    chi_square_test,
    fisher_exact_test,
    independent_t_test,
    mann_whitney_u_test,
    one_way_anova,
)
from utils.data_summary import continuous_columns, grouping_columns
from utils.state import get_dataset, save_result
from utils.ui import data_table, footer, init_page, kpi_cards, page_header, section


init_page("Statistical Tests")
page_header(
    "Statistical Tests",
    "Run common inferential tests for group comparisons and categorical associations.",
)
df = get_dataset()
if df is None:
    st.info("Upload a dataset first.")
    footer()
    st.stop()

continuous = continuous_columns(df)
grouping = grouping_columns(df)
kpi_cards(
    [
        ("Rows", f"{df.shape[0]:,}", "Available observations"),
        ("Continuous Variables", f"{len(continuous):,}", "Eligible continuous outcomes"),
        ("Group Variables", f"{len(grouping):,}", "Categorical or low-cardinality fields"),
        ("Missing Cells", f"{int(df.isna().sum().sum()):,}", "Excluded pairwise by tests"),
    ]
)

section("Test Configuration")
test = st.selectbox(
    "Test",
    [
        "Independent t-test",
        "Mann-Whitney U test",
        "Chi-square test",
        "Fisher exact test",
        "One-way ANOVA",
    ],
)

try:
    if test in ["Independent t-test", "Mann-Whitney U test", "One-way ANOVA"]:
        if not continuous:
            st.warning("This test requires at least one continuous variable.")
            footer()
            st.stop()
        if not grouping:
            st.warning("This test requires at least one grouping variable.")
            footer()
            st.stop()
        outcome = st.selectbox("Continuous outcome", continuous)
        group = st.selectbox("Grouping variable", grouping)
        if st.button("Run test", type="primary"):
            with st.spinner(f"Running {test}..."):
                if test == "Independent t-test":
                    result = independent_t_test(df, outcome, group)
                elif test == "Mann-Whitney U test":
                    result = mann_whitney_u_test(df, outcome, group)
                else:
                    result = one_way_anova(df, outcome, group)
            result_df = pd.DataFrame([result])
            st.success(f"{test} completed.")
            section("Results")
            data_table(result_df, height=180)
            save_result(test, result_df)
    else:
        if len(grouping) < 2:
            st.warning("At least two grouping variables are required.")
            footer()
            st.stop()
        row_col = st.selectbox("Rows", grouping)
        col_col = st.selectbox("Columns", [col for col in grouping if col != row_col])
        if st.button("Run test", type="primary"):
            with st.spinner(f"Running {test}..."):
                if test == "Chi-square test":
                    result, table = chi_square_test(df, row_col, col_col)
                else:
                    result, table = fisher_exact_test(df, row_col, col_col)
            st.success(f"{test} completed.")
            section("Contingency Table")
            data_table(table.reset_index(), height=260)
            result_df = pd.DataFrame([result])
            section("Results")
            data_table(result_df, height=180)
            save_result(test, result_df)
except Exception as exc:
    st.error(str(exc))

footer()

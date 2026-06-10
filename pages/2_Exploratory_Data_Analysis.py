import streamlit as st

from utils.data_summary import continuous_columns, grouping_columns, summary_statistics
from utils.state import get_dataset, save_plot, save_result
from utils.ui import data_table, footer, init_page, kpi_cards, page_header, section
from visualization.plots import (
    boxplot,
    correlation_heatmap,
    correlation_heatmap_interactive,
    histogram,
    missing_data_heatmap,
    missing_data_heatmap_interactive,
    pairwise_scatter,
)


init_page("Exploratory Analysis")
page_header(
    "Exploratory Data Analysis",
    "Profile distributions, associations, outliers, and missing-data patterns with interactive charts.",
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
        ("Continuous", f"{len(continuous):,}", "Numeric analysis variables"),
        ("Group Fields", f"{len(grouping):,}", "Categorical or low-cardinality variables"),
        ("Complete Rows", f"{df.dropna().shape[0]:,}", "Rows without missing cells"),
        ("Missing Rate", f"{df.isna().mean().mean() * 100:.1f}%", "Overall cell missingness"),
    ]
)

section("Summary Statistics")
with st.spinner("Calculating summary statistics..."):
    summary = summary_statistics(df)
    if summary.empty:
        st.warning("No continuous variables available for summary statistics.")
    else:
        data_table(summary.reset_index(names="variable"), height=420)
        save_result("Summary Statistics", summary.reset_index(names="variable"))

section("Distributions")
if continuous:
    col = st.selectbox("Histogram variable", continuous)
    with st.spinner("Rendering histogram..."):
        fig = histogram(df, col)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No continuous variables available for histograms.")

section("Boxplots")
if continuous:
    box_numeric = st.selectbox("Boxplot numeric variable", continuous, key="box_numeric")
    box_group = st.selectbox("Optional grouping variable", [None] + grouping, key="box_group")
    with st.spinner("Rendering boxplot..."):
        fig = boxplot(df, box_numeric, box_group)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No continuous variables available for boxplots.")

section("Correlation Matrix")
if len(continuous) >= 2:
    with st.spinner("Rendering correlation matrix..."):
        continuous_df = df[continuous]
        st.plotly_chart(correlation_heatmap_interactive(continuous_df), use_container_width=True)
        corr_fig = correlation_heatmap(continuous_df)
        save_plot("Correlation Matrix", corr_fig)
else:
    st.warning("At least two continuous variables are required.")

section("Pairwise Scatter Plots")
if len(continuous) >= 2:
    selected = st.multiselect("Variables", continuous, default=continuous[: min(4, len(continuous))])
    hue = st.selectbox("Color by", [None] + grouping)
    if selected:
        with st.spinner("Rendering pairwise scatter plot..."):
            st.plotly_chart(pairwise_scatter(df, selected, hue), use_container_width=True)
else:
    st.warning("At least two continuous variables are required.")

section("Missing Data Visualization")
with st.spinner("Rendering missing-data pattern..."):
    st.plotly_chart(missing_data_heatmap_interactive(df), use_container_width=True)
    missing_fig = missing_data_heatmap(df)
    save_plot("Missing Data Pattern", missing_fig)

footer()

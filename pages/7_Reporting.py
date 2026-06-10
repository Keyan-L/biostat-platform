import streamlit as st

from utils.reporting import figure_to_png_bytes, generate_pdf_report, results_to_csv_bytes
from utils.state import get_dataset, get_dataset_name, get_plots, get_results
from utils.ui import data_table, footer, init_page, kpi_cards, page_header, section


init_page("Reporting")
page_header(
    "Reporting",
    "Package saved analysis results and export plots for review, documentation, and downstream reporting.",
)
df = get_dataset()
results = get_results()
plots = get_plots()

if df is None:
    st.info("Upload a dataset first.")
    footer()
    st.stop()

kpi_cards(
    [
        ("Saved Results", f"{len(results):,}", "Tables available"),
        ("Saved Plots", f"{len(plots):,}", "Figures available"),
        ("Rows", f"{df.shape[0]:,}", "Dataset records"),
        ("Columns", f"{df.shape[1]:,}", "Dataset variables"),
    ]
)

section("Saved Results")
if results:
    st.write(f"{len(results)} result set(s) available for export.")
    for name, result in results.items():
        with st.expander(name):
            data_table(result if hasattr(result, "shape") else [result], height=260)
else:
    st.warning("Run analyses on earlier pages to populate report results.")

st.download_button(
    "Download results CSV",
    data=results_to_csv_bytes(results),
    file_name="biostat_results.csv",
    mime="text/csv",
    disabled=not results,
)

with st.spinner("Preparing PDF report..."):
    pdf_bytes = generate_pdf_report(get_dataset_name(), df, results, plots)
st.download_button(
    "Generate PDF report",
    data=pdf_bytes,
    file_name="biostat_report.pdf",
    mime="application/pdf",
)

section("Export Plots")
if plots:
    for name, figure in plots.items():
        safe_name = name.lower().replace(" ", "_")
        st.download_button(
            f"Download {name}",
            data=figure_to_png_bytes(figure),
            file_name=f"{safe_name}.png",
            mime="image/png",
        )
else:
    st.info("No exportable matplotlib plots have been saved yet.")

footer()

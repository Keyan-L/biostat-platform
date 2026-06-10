from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def result_to_dataframe(result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        return pd.DataFrame([result])
    return pd.DataFrame({"result": [str(result)]})


def results_to_csv_bytes(results: dict) -> bytes:
    parts = []
    for name, result in results.items():
        parts.append(f"### {name}\n")
        parts.append(result_to_dataframe(result).to_csv(index=False))
        parts.append("\n")
    return "".join(parts).encode("utf-8")


def figure_to_png_bytes(figure) -> bytes:
    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight", dpi=160)
    return output.getvalue()


def generate_pdf_report(dataset_name: str, df: pd.DataFrame | None, results: dict, plots: dict) -> bytes:
    output = BytesIO()
    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.1, 0.92, "Biostatistics Analysis Report", fontsize=20, weight="bold")
        fig.text(0.1, 0.87, f"Dataset: {dataset_name}", fontsize=11)
        fig.text(0.1, 0.84, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=11)
        if df is not None:
            fig.text(0.1, 0.79, f"Rows: {df.shape[0]:,}", fontsize=11)
            fig.text(0.1, 0.76, f"Columns: {df.shape[1]:,}", fontsize=11)
            fig.text(0.1, 0.73, f"Missing cells: {int(df.isna().sum().sum()):,}", fontsize=11)
        fig.text(0.1, 0.67, "Included analysis sections:", fontsize=13, weight="bold")
        y = 0.63
        for name in results.keys() or ["No analysis results saved yet"]:
            fig.text(0.12, y, f"- {name}", fontsize=10)
            y -= 0.025
            if y < 0.12:
                break
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for name, result in results.items():
            table_df = result_to_dataframe(result).head(25)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            ax.set_title(name, loc="left", fontsize=14, weight="bold")
            table = ax.table(
                cellText=table_df.round(4).astype(str).values,
                colLabels=table_df.columns,
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for name, figure in plots.items():
            if hasattr(figure, "savefig"):
                fig = figure
                try:
                    fig.tight_layout()
                except Exception:
                    pass
                pdf.savefig(fig, bbox_inches="tight")
    return output.getvalue()

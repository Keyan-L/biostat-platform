from __future__ import annotations

from io import BytesIO

import pandas as pd


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Upload a CSV, XLSX, or XLS file.")
    validate_uploaded_dataframe(df)
    return df


def validate_uploaded_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("The uploaded file does not contain any rows or columns.")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {', '.join(map(str, duplicates))}")
    if df.dropna(how="all").empty:
        raise ValueError("The uploaded dataset only contains empty rows.")
    if df.dropna(axis=1, how="all").empty:
        raise ValueError("The uploaded dataset only contains empty columns.")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return output.getvalue()

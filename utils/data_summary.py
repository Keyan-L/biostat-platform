from __future__ import annotations

import numpy as np
import pandas as pd


def variable_types(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "variable": column,
                "dtype": str(series.dtype),
                "inferred_type": infer_variable_type(series),
                "unique_values": int(series.nunique(dropna=True)),
                "missing": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def infer_variable_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "Binary"
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) <= 2:
            return "Binary numeric"
        return "Continuous"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"
    if series.nunique(dropna=True) <= 10:
        return "Categorical"
    return "Text / high-cardinality categorical"


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "variable": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": np.round(df.isna().mean().values * 100, 2),
        }
    )
    return summary.sort_values("missing_percent", ascending=False).reset_index(drop=True)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=np.number).columns.tolist()


def continuous_columns(df: pd.DataFrame, min_unique: int = 6) -> list[str]:
    columns = []
    for column in numeric_columns(df):
        if df[column].nunique(dropna=True) >= min_unique:
            columns.append(column)
    return columns


def grouping_columns(df: pd.DataFrame, max_unique: int = 10) -> list[str]:
    columns = []
    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        if unique_count <= 1:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            if unique_count <= 2:
                columns.append(column)
        elif unique_count <= max_unique:
            columns.append(column)
    return columns


def binary_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if df[column].nunique(dropna=True) == 2]


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.assign(
        missing=numeric.isna().sum(),
        skewness=numeric.skew(numeric_only=True),
        kurtosis=numeric.kurtosis(numeric_only=True),
    )

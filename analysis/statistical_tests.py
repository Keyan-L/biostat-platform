from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def independent_t_test(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict:
    groups = _two_groups(df, numeric_col, group_col)
    _require_min_group_size(groups, 2, "Independent t-test")
    statistic, p_value = stats.ttest_ind(groups[0], groups[1], nan_policy="omit", equal_var=False)
    return _two_group_result("Independent t-test", groups, statistic, p_value)


def mann_whitney_u_test(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict:
    groups = _two_groups(df, numeric_col, group_col)
    _require_min_group_size(groups, 2, "Mann-Whitney U test")
    statistic, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
    return _two_group_result("Mann-Whitney U test", groups, statistic, p_value)


def chi_square_test(df: pd.DataFrame, row_col: str, col_col: str) -> tuple[dict, pd.DataFrame]:
    table = pd.crosstab(df[row_col], df[col_col])
    _validate_contingency_table(table, "Chi-square test")
    statistic, p_value, dof, expected = stats.chi2_contingency(table)
    result = {
        "test": "Chi-square test",
        "statistic": statistic,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "minimum_expected_count": np.min(expected),
        "expected_count_warning": bool(np.min(expected) < 5),
    }
    return result, table


def fisher_exact_test(df: pd.DataFrame, row_col: str, col_col: str) -> tuple[dict, pd.DataFrame]:
    table = pd.crosstab(df[row_col], df[col_col])
    _validate_contingency_table(table, "Fisher exact test")
    if table.shape != (2, 2):
        raise ValueError("Fisher exact test requires a 2 x 2 contingency table.")
    odds_ratio, p_value = stats.fisher_exact(table)
    result = {
        "test": "Fisher exact test",
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }
    return result, table


def one_way_anova(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict:
    grouped = [
        group[numeric_col].dropna().values
        for _, group in df[[numeric_col, group_col]].dropna().groupby(group_col)
    ]
    if len(grouped) < 2:
        raise ValueError("ANOVA requires at least two groups.")
    _require_min_group_size(grouped, 2, "ANOVA")
    statistic, p_value = stats.f_oneway(*grouped)
    if np.isnan(statistic) or np.isnan(p_value):
        raise ValueError("ANOVA could not be computed. Check that each group has variation and enough observations.")
    return {
        "test": "One-way ANOVA",
        "outcome": numeric_col,
        "group": group_col,
        "groups": len(grouped),
        "statistic": statistic,
        "p_value": p_value,
    }


def _two_groups(df: pd.DataFrame, numeric_col: str, group_col: str) -> list[pd.Series]:
    clean = df[[numeric_col, group_col]].dropna()
    if clean.empty:
        raise ValueError("No complete observations are available for the selected variables.")
    if clean[numeric_col].nunique(dropna=True) < 2:
        raise ValueError("The selected continuous variable has fewer than two distinct values after removing missing data.")
    levels = clean[group_col].unique()
    if len(levels) != 2:
        raise ValueError("This test requires exactly two groups.")
    return [clean.loc[clean[group_col] == level, numeric_col] for level in levels]


def _require_min_group_size(groups: list[pd.Series | np.ndarray], minimum: int, test_name: str) -> None:
    sizes = [len(group) for group in groups]
    if any(size < minimum for size in sizes):
        raise ValueError(f"{test_name} requires at least {minimum} non-missing observations in each group. Current group sizes: {sizes}.")


def _validate_contingency_table(table: pd.DataFrame, test_name: str) -> None:
    if table.empty:
        raise ValueError(f"{test_name} requires at least one complete observation for both selected variables.")
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(f"{test_name} requires at least two row levels and two column levels.")
    if table.to_numpy().sum() == 0:
        raise ValueError(f"{test_name} cannot be computed because the contingency table is empty.")


def _two_group_result(test: str, groups: list[pd.Series], statistic: float, p_value: float) -> dict:
    return {
        "test": test,
        "group_1_n": int(groups[0].shape[0]),
        "group_1_mean": float(groups[0].mean()),
        "group_2_n": int(groups[1].shape[0]),
        "group_2_mean": float(groups[1].mean()),
        "statistic": statistic,
        "p_value": p_value,
    }

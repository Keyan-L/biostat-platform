from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from sklearn.metrics import confusion_matrix, roc_auc_score


def _design_matrix(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    x = pd.get_dummies(df[predictors], drop_first=True, dtype=float)
    return sm.add_constant(x, has_constant="add")


def linear_regression(df: pd.DataFrame, outcome: str, predictors: list[str]):
    data = df[[outcome] + predictors].dropna()
    y = data[outcome].astype(float)
    x = _design_matrix(data, predictors)
    _validate_regression_inputs(data, y, x, "Linear regression")
    model = sm.OLS(y, x).fit()
    summary = model_summary_table(model, odds_ratios=False)
    diagnostics = {
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "observations": int(model.nobs),
        "residual_mean": float(np.mean(model.resid)),
    }
    return model, summary, diagnostics


def logistic_regression(df: pd.DataFrame, outcome: str, predictors: list[str]):
    data = df[[outcome] + predictors].dropna()
    y = data[outcome]
    if y.nunique() != 2:
        raise ValueError("Logistic regression outcome must have exactly two classes.")
    y = pd.Series(pd.Categorical(y).codes, index=data.index, name=outcome)
    x = _design_matrix(data, predictors)
    _validate_regression_inputs(data, y, x, "Logistic regression")
    if y.value_counts().min() < 2:
        raise ValueError(f"Logistic regression requires at least 2 observations in each outcome class. Current counts: {y.value_counts().to_dict()}.")
    try:
        model = sm.Logit(y.astype(float), x).fit(disp=False)
    except PerfectSeparationError as exc:
        raise ValueError("Logistic regression failed because predictors perfectly separate the outcome. Remove overly predictive or sparse predictors.") from exc
    except np.linalg.LinAlgError as exc:
        raise ValueError("Logistic regression failed due to a singular design matrix. Remove collinear or sparse predictors.") from exc
    summary = model_summary_table(model, odds_ratios=True)
    predicted = model.predict(x)
    classes = (predicted >= 0.5).astype(int)
    diagnostics = {
        "pseudo_r_squared": model.prsquared,
        "aic": model.aic,
        "bic": model.bic,
        "observations": int(model.nobs),
        "roc_auc": roc_auc_score(y, predicted),
        "confusion_matrix": confusion_matrix(y, classes).tolist(),
    }
    return model, summary, diagnostics, predicted, y


def _validate_regression_inputs(data: pd.DataFrame, y: pd.Series, x: pd.DataFrame, model_name: str) -> None:
    if data.empty:
        raise ValueError(f"{model_name} has no complete rows after removing missing values.")
    if len(data) < 5:
        raise ValueError(f"{model_name} requires at least 5 complete observations.")
    if y.nunique(dropna=True) < 2:
        raise ValueError(f"{model_name} outcome must have at least two distinct values.")
    if x.shape[1] >= len(data):
        raise ValueError(
            f"{model_name} has too many parameters ({x.shape[1]}) for {len(data)} complete observations. "
            "Use fewer predictors or combine sparse categories."
        )
    rank = np.linalg.matrix_rank(x.to_numpy(dtype=float))
    if rank < x.shape[1]:
        raise ValueError(f"{model_name} design matrix is rank deficient. Remove collinear predictors or sparse categories.")
    if np.isinf(x.to_numpy(dtype=float)).any() or np.isinf(pd.to_numeric(y, errors="coerce").to_numpy()).any():
        raise ValueError(f"{model_name} inputs contain infinite values. Replace or remove them before fitting.")


def model_summary_table(model, odds_ratios: bool) -> pd.DataFrame:
    conf = model.conf_int()
    output = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.values,
            "std_error": model.bse.values,
            "p_value": model.pvalues.values,
            "ci_lower": conf[0].values,
            "ci_upper": conf[1].values,
        }
    )
    if odds_ratios:
        output["odds_ratio"] = np.exp(output["estimate"])
        output["or_ci_lower"] = np.exp(output["ci_lower"])
        output["or_ci_upper"] = np.exp(output["ci_upper"])
    return output

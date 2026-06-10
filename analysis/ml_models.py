from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - optional dependency is declared in requirements.
    XGBClassifier = None
    XGBRegressor = None


CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "AdaBoost",
    "Decision Tree",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "Naive Bayes",
    "XGBoost",
]

REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "AdaBoost",
    "Decision Tree",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "XGBoost",
]


def available_models(task: str) -> list[str]:
    return CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS


def train_model(df: pd.DataFrame, target: str, predictors: list[str], model_name: str, task: str):
    data = df[[target] + predictors].dropna(subset=[target])
    x = data[predictors]
    y = data[target]
    validate_model_inputs(x, y, predictors, task)
    if task == "classification":
        y = pd.Series(pd.Categorical(y).codes, index=data.index, name=target)
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
    elif not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression requires a numeric target.")

    preprocessor = build_preprocessor(x)
    estimator = build_estimator(model_name, task)

    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    stratify = y if task == "classification" and y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=stratify
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    if task == "classification":
        score = accuracy_score(y_test, predictions)
        metric_name = "accuracy"
        if hasattr(pipeline, "predict_proba") and pd.Series(y).nunique() == 2:
            proba = pipeline.predict_proba(x_test)[:, 1]
            roc_auc = roc_auc_score(y_test, proba)
        else:
            roc_auc = None
        extra_metrics = {}
    else:
        score = float(((y_test - predictions) ** 2).mean() ** 0.5)
        metric_name = "rmse"
        roc_auc = None
        extra_metrics = {
            "mean_absolute_error": mean_absolute_error(y_test, predictions),
            "r_squared": r2_score(y_test, predictions),
        }

    scoring = "accuracy" if task == "classification" else "neg_root_mean_squared_error"
    cv = cross_validation_strategy(y, task)
    cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring=scoring)
    display_cv_scores = cv_scores if task == "classification" else -cv_scores
    importance = feature_importance(pipeline, x)

    metrics = {
        "model": model_name,
        "task": task,
        metric_name: score,
        "cross_validation_mean": display_cv_scores.mean(),
        "cross_validation_sd": display_cv_scores.std(),
        "roc_auc": roc_auc,
        "training_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
    }
    metrics.update(extra_metrics)
    return pipeline, pd.DataFrame([metrics]), importance, (x_test, y_test, predictions)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
        ]
    )


def validate_model_inputs(x: pd.DataFrame, y: pd.Series, predictors: list[str], task: str) -> None:
    if not predictors:
        raise ValueError("Select at least one predictor before training a model.")
    if len(y) < 10:
        raise ValueError("At least 10 rows with a non-missing target are recommended for model training.")
    if y.isna().all():
        raise ValueError("The selected target contains only missing values.")
    if x.empty:
        raise ValueError("No predictor columns are available for model training.")
    all_missing = [column for column in x.columns if x[column].isna().all()]
    if all_missing:
        raise ValueError(f"Remove predictor columns that are entirely missing: {', '.join(all_missing)}.")
    constant = [column for column in x.columns if x[column].nunique(dropna=True) <= 1]
    if constant:
        raise ValueError(f"Remove predictor columns with no variation: {', '.join(constant)}.")
    high_cardinality = [
        column
        for column in x.select_dtypes(exclude=np.number).columns
        if x[column].nunique(dropna=True) > 50 and x[column].nunique(dropna=True) / max(len(x), 1) > 0.5
    ]
    if high_cardinality:
        raise ValueError(f"High-cardinality categorical predictors look like IDs or free text: {', '.join(high_cardinality)}.")
    numeric = x.select_dtypes(include=np.number)
    if not numeric.empty and np.isinf(numeric.to_numpy()).any():
        raise ValueError("Numeric predictors contain infinite values. Replace or remove them before training.")
    if task == "classification":
        class_counts = y.value_counts(dropna=True)
        if len(class_counts) < 2:
            raise ValueError("Classification requires at least two target classes.")
        if class_counts.min() < 2:
            raise ValueError(f"Each class needs at least 2 observations for stratified training. Current counts: {class_counts.to_dict()}.")
        test_rows = math.ceil(len(y) * 0.25)
        if len(class_counts) > test_rows:
            raise ValueError(
                f"The target has {len(class_counts)} classes, but the 25% holdout would contain only {test_rows} rows. "
                "Use more data or reduce the number of classes."
            )
    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression requires a numeric target.")
        if y.nunique(dropna=True) < 2:
            raise ValueError("Regression target must have at least two distinct values.")
        if np.isinf(pd.to_numeric(y, errors="coerce").to_numpy()).any():
            raise ValueError("Regression target contains infinite values. Replace or remove them before training.")


def build_estimator(model_name: str, task: str):
    if model_name == "XGBoost":
        if XGBClassifier is None or XGBRegressor is None:
            raise ImportError("Install xgboost to train XGBoost models.")
        if task == "classification":
            return XGBClassifier(eval_metric="logloss", random_state=42)
        return XGBRegressor(random_state=42)

    if task == "classification":
        estimators = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
            "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            "Support Vector Machine": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
            "Naive Bayes": GaussianNB(),
        }
    else:
        estimators = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.05, max_iter=5000),
            "Elastic Net": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
            "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
            "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Support Vector Machine": SVR(kernel="rbf"),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7),
        }
    if model_name not in estimators:
        raise ValueError(f"Unsupported model for {task}: {model_name}")
    return estimators[model_name]


def cross_validation_strategy(y: pd.Series, task: str):
    if task == "classification":
        min_class_count = int(pd.Series(y).value_counts().min())
        folds = max(2, min(5, min_class_count))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    folds = max(2, min(5, len(y)))
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def feature_importance(pipeline: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        column = "importance"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = abs(coef[0] if getattr(coef, "ndim", 1) > 1 else coef)
        column = "absolute_coefficient"
    else:
        return pd.DataFrame()
    return pd.DataFrame({"feature": names, column: values}).sort_values(column, ascending=False)

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve


def histogram(df: pd.DataFrame, column: str):
    return px.histogram(df, x=column, marginal="box", template="plotly_white")


def boxplot(df: pd.DataFrame, numeric_col: str, group_col: str | None = None):
    return px.box(df, x=group_col, y=numeric_col, points="outliers", template="plotly_white")


def correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig


def correlation_heatmap_interactive(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template="plotly_white",
    )
    fig.update_layout(title="Correlation Matrix", height=560, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def pairwise_scatter(df: pd.DataFrame, columns: list[str], hue: str | None = None):
    return px.scatter_matrix(df, dimensions=columns, color=hue, template="plotly_white")


def missing_data_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="mako", ax=ax)
    ax.set_title("Missing Data Pattern")
    ax.set_xlabel("Variables")
    fig.tight_layout()
    return fig


def missing_data_heatmap_interactive(df: pd.DataFrame):
    missing = df.isna().astype(int)
    fig = go.Figure(
        data=go.Heatmap(
            z=missing.values,
            x=missing.columns,
            y=list(range(1, len(missing) + 1)),
            colorscale=[[0, "#eef2f7"], [1, "#d97706"]],
            showscale=True,
            colorbar=dict(title="Missing"),
        )
    )
    fig.update_layout(
        title="Missing Data Pattern",
        xaxis_title="Variables",
        yaxis_title="Rows",
        height=520,
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def roc_curve_figure(y_true, y_score):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7b8794")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    return fig


def roc_curve_interactive(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color="#126a8a", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Reference", line=dict(color="#98a2b3", dash="dash")))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def confusion_matrix_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    size = max(5, min(8, 1.1 * cm.shape[0] + 2.6))
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def confusion_matrix_interactive(y_true, y_pred):
    labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        x=[str(label) for label in labels],
        y=[str(label) for label in labels],
        text_auto=True,
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def calibration_plot_figure(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7b8794", label="Ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title("Calibration Plot")
    ax.legend()
    fig.tight_layout()
    return fig


def calibration_plot_interactive(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model", line=dict(color="#126a8a", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(color="#98a2b3", dash="dash")))
    fig.update_layout(
        title="Calibration Plot",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed probability",
        template="plotly_white",
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig

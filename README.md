# Biostatistics Analysis Platform

A professional Streamlit application for biomedical and clinical dataset analysis. The platform supports data upload, exploratory analysis, statistical tests, regression modeling, machine learning, evaluation visualization, and report export.

## Features

- Multi-page Streamlit workflow
- Professional SaaS-style dashboard UI with icon sidebar navigation, KPI cards, responsive spacing, and a consistent visual theme
- CSV and Excel upload
- Included sample biostatistics dataset
- Dataset dimensions, variable typing, and missing-value summaries
- Summary statistics, histograms, boxplots, correlation matrices, pairwise scatter plots, and missing data maps
- Independent t-test, Mann-Whitney U test, Chi-square test, Fisher exact test, and one-way ANOVA
- Linear and logistic regression with p-values, confidence intervals, odds ratios, and diagnostics
- Logistic regression, regularized regression, tree ensembles, boosting, SVM, KNN, Naive Bayes, Random Forest, and XGBoost models with cross-validation and feature importance where available
- ROC curves, confusion matrices, and calibration plots
- CSV result export and PDF report generation

## Project Structure

```text
biostat-platform/
├── app.py
├── data/
│   └── sample_biostat_dataset.csv
├── pages/
│   ├── 1_Data_Upload.py
│   ├── 2_Exploratory_Data_Analysis.py
│   ├── 3_Statistical_Tests.py
│   ├── 4_Regression_Analysis.py
│   ├── 5_Machine_Learning.py
│   ├── 6_Visualization.py
│   └── 7_Reporting.py
├── analysis/
│   ├── ml_models.py
│   ├── regression.py
│   └── statistical_tests.py
├── visualization/
│   └── plots.py
├── utils/
│   ├── data_loader.py
│   ├── data_summary.py
│   ├── reporting.py
│   └── state.py
├── requirements.txt
└── .gitignore
```

## Quick Start

Python 3.11 or 3.12 is recommended for the broadest binary-wheel support across the scientific Python stack.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the **Data Upload** page and click **Load sample dataset** to try the full workflow without bringing your own data.

## Notes

The app stores datasets and analysis outputs in Streamlit session state. For regulated clinical workflows, validate every model and statistical assumption against your study protocol before using outputs in formal reporting.

# Python 3.9.6 Compatibility Report

## Summary

The repository was reviewed and updated for Python 3.9.6 compatibility without changing application logic or UI behavior.

## Code Changes

- Replaced Python 3.10-style union type syntax (`A | B`) with Python 3.9-compatible `typing.Optional` and `typing.Union`.
- Updated affected type annotations in:
  - `analysis/statistical_tests.py`
  - `utils/reporting.py`
  - `utils/state.py`
  - `utils/ui.py`
  - `visualization/plots.py`
- Kept existing `list[...]`, `dict[...]`, and `tuple[...]` annotations because they are valid in Python 3.9.

## Dependency Changes

`requirements.txt` now includes upper bounds so future installs remain on package lines that support Python 3.9.6:

- `streamlit>=1.35,<1.51`
- `pandas>=2.2,<3.0`
- `numpy>=1.26,<2.1`
- `scipy>=1.12,<1.14`
- `statsmodels>=0.14,<0.15`
- `scikit-learn>=1.4,<1.7`
- `matplotlib>=3.8,<3.10`
- `seaborn>=0.13,<0.14`
- `plotly>=5.22,<7.0`
- `openpyxl>=3.1,<3.2`
- `xgboost>=2.0,<3.0`

## Verification

- Verified the active virtual environment uses Python 3.9.6.
- Confirmed Python files compile under Python 3.9.6.
- Started the app with:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

The Streamlit startup result is recorded after the final run.

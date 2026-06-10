from __future__ import annotations

from pathlib import Path

import pandas as pd


SAMPLE_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_biostat_dataset.csv"


def load_sample_dataset() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH)

import os

import pandas as pd


def test_raw_data_exists() -> None:
    data_path = os.getenv("DATA_PATH", "data/raw/weatherAUS.csv")
    assert os.path.exists(data_path), f"Data not found: {data_path}"


def test_raw_data_schema_basic() -> None:
    data_path = os.getenv("DATA_PATH", "data/raw/weatherAUS.csv")
    df = pd.read_csv(data_path)

    required_cols = {
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Humidity3pm",
        "Pressure3pm",
        "RainTomorrow",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"

    assert df.shape[0] >= 1000, "Dataset is too small for this lab"
    assert df["RainTomorrow"].notna().any(), "RainTomorrow has no non-null values"

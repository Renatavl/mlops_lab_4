from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from common import load_hydra_config, resolve_path, set_global_seed


def clean_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    data = df.copy()
    data = data.dropna(subset=[target_column])

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data["Year"] = data["Date"].dt.year
        data["Month"] = data["Date"].dt.month
        data["Day"] = data["Date"].dt.day
        data = data.drop(columns=["Date"])

    target = data[target_column].astype(str).str.strip().map({"No": 0, "Yes": 1})
    valid_mask = target.notna()
    data = data.loc[valid_mask].copy()
    data[target_column] = target.loc[valid_mask].astype(int)
    return data


def split_and_impute(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )

    feature_columns = [column for column in train_df.columns if column != target_column]
    categorical_cols = (
        train_df[feature_columns]
        .select_dtypes(include=["object", "category", "bool"])
        .columns.tolist()
    )
    numeric_cols = (
        train_df[feature_columns].select_dtypes(include=["number"]).columns.tolist()
    )

    numeric_fill_values: dict[str, float] = {}
    categorical_fill_values: dict[str, str] = {}

    for column in numeric_cols:
        median_value = train_df[column].median()
        numeric_fill_values[column] = (
            0.0 if pd.isna(median_value) else float(median_value)
        )

    for column in categorical_cols:
        mode = train_df[column].mode(dropna=True)
        categorical_fill_values[column] = "Unknown" if mode.empty else str(mode.iloc[0])

    for frame in (train_df, test_df):
        for column, value in numeric_fill_values.items():
            frame[column] = frame[column].fillna(value)
        for column, value in categorical_fill_values.items():
            frame[column] = frame[column].fillna(value)

    metadata = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "split": {
            "test_size": test_size,
            "random_state": random_state,
        },
        "rows": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
            "total": int(len(df)),
        },
    }
    return train_df, test_df, metadata


def save_processed_dataset(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str, output_path: Path
) -> None:
    payload = {
        "X_train": train_df.drop(columns=[target_column]),
        "X_test": test_df.drop(columns=[target_column]),
        "y_train": train_df[target_column],
        "y_test": test_df[target_column],
    }
    joblib.dump(payload, output_path)


def main() -> None:
    cfg = load_hydra_config()
    set_global_seed(int(cfg.seed))

    raw_path = resolve_path(cfg.data.raw_path)
    processed_path = resolve_path(cfg.data.processed_path)
    metadata_path = resolve_path(cfg.data.metadata_path)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = clean_dataframe(df, target_column=str(cfg.data.target_column))

    train_df, test_df, metadata = split_and_impute(
        df=df,
        target_column=str(cfg.data.target_column),
        test_size=float(cfg.split.test_size),
        random_state=int(cfg.split.random_state),
    )

    save_processed_dataset(
        train_df, test_df, str(cfg.data.target_column), processed_path
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Prepared data saved successfully.")
    print(f"Processed dataset: {processed_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

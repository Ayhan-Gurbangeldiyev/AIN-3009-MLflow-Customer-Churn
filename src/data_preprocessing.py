"""Data loading and preprocessing utilities for the churn project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]
RANDOM_STATE = 42


@dataclass(frozen=True)
class PreparedData:
    """Container for train/test splits and preprocessing metadata."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    numeric_features: list[str]
    categorical_features: list[str]


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load the Telco churn dataset and normalize known schema quirks."""

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and binary target."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    x = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS], errors="ignore").copy()
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Target column contains values other than 'Yes' and 'No'.")

    return x, y.astype(int)


def build_preprocessor(x: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Build a reusable sklearn preprocessing transformer."""

    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def prepare_data(
    data_path: str | Path,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    scale_numeric: bool = True,
) -> PreparedData:
    """Load, clean, split, and describe data for model training."""

    df = load_dataset(data_path)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    preprocessor = build_preprocessor(x_train, scale_numeric=scale_numeric)

    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        numeric_features=x_train.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        categorical_features=x_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
    )


def dataset_summary(data_path: str | Path) -> dict[str, object]:
    """Return key dataset facts for README/report and sanity checks."""

    df = load_dataset(data_path)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "total_charges_missing_after_numeric": int(df["TotalCharges"].isna().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "columns_list": df.columns.tolist(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect and validate the Telco churn dataset.")
    parser.add_argument("--data", default="data/telco_churn.csv", help="Path to the churn CSV file.")
    args = parser.parse_args()

    summary = dataset_summary(args.data)
    for key, value in summary.items():
        print(f"{key}: {value}")

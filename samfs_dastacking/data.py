import os
from typing import Iterable, Optional

import pandas as pd

from .config import TARGET_COL


def read_csv_auto(path: str) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def load_split(split_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = os.path.join(split_dir, "train.csv")
    test_csv = os.path.join(split_dir, "test.csv")
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(f"train.csv/test.csv not found in: {split_dir}")
    return read_csv_auto(train_csv), read_csv_auto(test_csv)


def validate_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Iterable[str],
    target_col: str = TARGET_COL,
) -> list[str]:
    features = list(features)
    required = features + [target_col]
    missing_train = [c for c in required if c not in train_df.columns]
    missing_test = [c for c in required if c not in test_df.columns]
    if missing_train or missing_test:
        raise KeyError(
            f"Missing columns. train={missing_train or 'none'}, "
            f"test={missing_test or 'none'}"
        )
    return features


def subset_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Iterable[str],
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = validate_columns(train_df, test_df, features, target_col)
    cols = features + [target_col]
    return train_df[cols].copy(), test_df[cols].copy()


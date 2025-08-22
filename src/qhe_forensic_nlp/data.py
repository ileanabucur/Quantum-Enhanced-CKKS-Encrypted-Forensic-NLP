"""
Data loading and splitting utilities.

- `load_csv`: read a CSV and validate required columns.
- `train_test_split_df`: stratified train/test split to preserve label ratios.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ("text", "label")


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset and ensure it has the required columns.

    Args:
        path: Path to a CSV file with columns 'text' and 'label'.

    Returns:
        The loaded DataFrame.

    Raises:
        ValueError: If the required columns are missing.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV must contain columns {REQUIRED_COLUMNS}. Missing: {missing}"
        )

    return df


def train_test_split_df(
    df: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split to preserve the class distribution.

    Args:
        df: Input DataFrame (must contain a 'label' column).
        test_size: Fraction of the dataset to use for the test split.
        seed: Random seed for reproducibility.

    Returns:
        A tuple (train_df, test_df).
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

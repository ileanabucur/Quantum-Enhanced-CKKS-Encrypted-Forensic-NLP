"""
Small utility helpers for reproducibility and filesystem I/O:

- set_seed: set Python and NumPy RNG seeds
- ensure_dir: create a directory if it does not already exist
- save_json: write a Python object to disk as pretty-printed JSON
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for Python's `random` and NumPy to improve reproducibility.

    Args:
        seed: Integer seed value to use for RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    """
    Create a directory if it doesn't exist (no error if it already exists).

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def save_json(path: str | os.PathLike[str], obj: Any) -> None:
    """
    Save a Python object to disk as JSON with pretty indentation.

    Args:
        path: Output file path.
        obj:  JSON-serializable Python object (dict, list, etc.).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

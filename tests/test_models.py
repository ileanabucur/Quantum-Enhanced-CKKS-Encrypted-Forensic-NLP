"""
Smoke test for SVM (LinearSVC) and RandomForest training pipelines.

This test:
  1) Generates a small synthetic dataset.
  2) Trains LinearSVC on TF-IDF features.
  3) Trains RandomForest on Word2Vec features.
  4) Verifies that the expected metrics JSON files are written and valid.

Notes:
- Uses subprocess to execute the CLI scripts exactly as a user would.
- Keeps the dataset small to keep CI fast.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _sh(cmd: Iterable[str]) -> None:
    """Run a shell command, echoing it first; raise on non-zero exit."""
    print("\n>>>", " ".join(cmd))
    subprocess.run(list(cmd), check=True)


def test_svm_and_rf(tmp_path: Path) -> None:
    """
    End-to-end smoke test for two classic models:
      - LinearSVC (TF-IDF)
      - RandomForest (Word2Vec)
    """
    # Use project-relative paths (pytest runs at repo root by default).
    data_path = Path("data/samples/small2.csv")
    results_dir = Path("results")

    # 1) Prepare a tiny dataset
    _sh(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "--out",
            str(data_path),
            "--n",
            "80",
        ]
    )

    # 2) Train LinearSVC on TF-IDF
    _sh(
        [
            sys.executable,
            "scripts/train.py",
            "classic",
            "--data",
            str(data_path),
            "--emb",
            "tfidf",
            "--model",
            "linearsvc",
        ]
    )

    # 3) Train RandomForest on Word2Vec
    _sh(
        [
            sys.executable,
            "scripts/train.py",
            "classic",
            "--data",
            str(data_path),
            "--emb",
            "w2v",
            "--model",
            "randomforest",
        ]
    )

    # 4) Check output metrics files exist
    tfidf_json = results_dir / "classic_linearsvc_tfidf_metrics.json"
    w2v_json = results_dir / "classic_randomforest_w2v_metrics.json"

    assert tfidf_json.is_file(), f"Missing file: {tfidf_json}"
    assert w2v_json.is_file(), f"Missing file: {w2v_json}"

    # 5) Sanity-check JSON structure (common keys expected across runs)
    required_keys = {"accuracy", "precision", "recall", "f1", "confusion_matrix"}

    with tfidf_json.open("r", encoding="utf-8") as f:
        m1 = json.load(f)
    with w2v_json.open("r", encoding="utf-8") as f:
        m2 = json.load(f)

    assert required_keys.issubset(m1.keys()), f"Missing keys in {tfidf_json}: {required_keys - set(m1.keys())}"
    assert required_keys.issubset(m2.keys()), f"Missing keys in {w2v_json}: {required_keys - set(m2.keys())}"

"""
Smoke test for the baseline pipeline (TF-IDF + Logistic Regression).

This test:
  1) Generates a small synthetic dataset.
  2) Trains a Logistic Regression model on TF-IDF features.
  3) Verifies that the expected metrics JSON is produced and minimally valid.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _sh(cmd: list[str]) -> None:
    """Run a shell command, echoing it first; raise on non-zero exit."""
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def test_smoke_pipeline() -> None:
    data_path = Path("data/samples/small.csv")
    results_json = Path("results/classic_logreg_tfidf_metrics.json")

    # 1) Generate a tiny dataset
    _sh(
        [
            sys.executable,
            "scripts/ggenerate_dataset.py",
            "--out",
            str(data_path),
            "--n",
            "60",
        ]
    )

    # 2) Train baseline (LogReg on TF-IDF)
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
            "logreg",
            "--max-iter",
            "100",
        ]
    )

    # 3) Check output exists
    assert results_json.is_file(), f"Missing file: {results_json}"

    # Minimal JSON sanity check
    required = {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
    with results_json.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    missing = required - set(metrics)
    assert not missing, f"Missing keys in {results_json}: {missing}"

"""
Smoke test for the baseline pipeline (TF-IDF + Logistic Regression).
"""

import json
import subprocess
import sys
from pathlib import Path


def _sh(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def test_smoke_pipeline() -> None:
    data_path = Path("data/samples/small.csv")
    results_json = Path("results/classic_logreg_tfidf_metrics.json")

    # 1) Generate a tiny dataset
    _sh(
        [
            sys.executable,
            "scripts/generate_dataset.py",
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

    # 3) Check output exists and is valid JSON
    assert results_json.is_file(), f"Missing file: {results_json}"
    with results_json.open("r", encoding="utf-8") as f:
        json.load(f)

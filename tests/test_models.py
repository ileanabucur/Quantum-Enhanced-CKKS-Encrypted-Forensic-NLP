import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# In CI saltiamo il test "pesante" (resta eseguibile in locale)
SKIP_CI = os.environ.get("CI", "").lower() == "true"


def _sh(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


@pytest.mark.skipif(SKIP_CI, reason="Skip heavier SVM/RF sweep in CI for stability/speed")
def test_svm_and_rf() -> None:
    data_path = Path("data/samples/small2.csv")

    # 1) Dataset
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

    # 2) LinearSVC (TF-IDF)
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

    # 3) RandomForest (Word2Vec)
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

    # 4) File prodotti
    assert Path("results/classic_linearsvc_tfidf_metrics.json").is_file()
    assert Path("results/classic_randomforest_w2v_metrics.json").is_file()

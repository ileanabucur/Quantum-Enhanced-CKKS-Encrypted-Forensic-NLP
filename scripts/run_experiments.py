import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def sh(cmd: List[str]) -> None:
    """Run a shell command, echoing it first (raise on error)."""
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dataset(data_path: Path, n_rows: int = 600) -> None:
    """
    Ensure the synthetic dataset exists. If missing, generate it using the helper script.
    """
    if data_path.exists():
        return
    sh(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "--out",
            str(data_path),
            "--n",
            str(n_rows),
        ]
    )


def sweep_runs(data_path: Path) -> None:
    """
    Run the full sweep:
      - Classic models over TF-IDF and Word2Vec
      - HE encrypted inference (TF-IDF)
      - Quantum VQC (TF-IDF)
    """
    for emb in ("tfidf", "w2v"):
        for model in ("logreg", "linearsvc", "randomforest"):
            sh(
                [
                    sys.executable,
                    "scripts/train.py",
                    "classic",
                    "--data",
                    str(data_path),
                    "--emb",
                    emb,
                    "--model",
                    model,
                ]
            )

    # HE inference (default to TF-IDF for speed/stability)
    sh(
        [
            sys.executable,
            "scripts/train.py",
            "he",
            "--data",
            str(data_path),
            "--emb",
            "tfidf",
        ]
    )

    # Quantum VQC (small, laptop-friendly defaults)
    sh(
        [
            sys.executable,
            "scripts/train.py",
            "quantum",
            "--data",
            str(data_path),
            "--emb",
            "tfidf",
            "--qubits",
            "4",
            "--layers",
            "3",
            "--epochs",
            "40",
        ]
    )


def aggregate_json(results_dir: Path, out_csv: Path) -> None:
    """
    Aggregate per-run JSON metrics into a single CSV for plotting/reporting.

    Expected JSON fields include (some may be absent for some runs):
      - accuracy, precision, recall, f1, confusion_matrix, encryption_ms
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for jf in results_dir.glob("*metrics.json"):
        with open(jf, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        run_name = jf.stem.replace("_metrics", "")
        rows.append({"run": run_name, **metrics})

    # Keep the fixed header used elsewhere in the repo; fill missing keys with None
    fieldnames = [
        "run",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "encryption_ms",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"Saved {out_csv}")


def main() -> None:
    """
    Orchestrate the whole experiment sweep:
      1) Ensure dataset exists (generate if missing)
      2) Run classic/HE/quantum experiments
      3) Aggregate JSON metrics into a CSV
    """
    parser = argparse.ArgumentParser(description="Run experiment sweep and aggregate results.")
    parser.add_argument(
        "--data",
        default="data/samples/synthetic.csv",
        help="Path to the dataset CSV (generated if missing).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=600,
        help="Number of rows to generate if the dataset is missing.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where per-run metrics JSON files are stored.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/aggregate_metrics.csv",
        help="Output CSV path for the aggregated metrics.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    results_dir = Path(args.results_dir)
    out_csv = Path(args.out_csv)

    ensure_dataset(data_path, n_rows=args.n)
    sweep_runs(data_path)
    aggregate_json(results_dir, out_csv)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    """
    Read an aggregated results CSV and produce a simple bar chart
    of F1 scores per run.

    Expected CSV columns:
      - 'run' : string identifier for each experiment/run
      - 'f1'  : numeric F1 score for that run
    """
    parser = argparse.ArgumentParser(
        description="Plot F1 score by run from an aggregated results CSV."
    )
    parser.add_argument(
        "--results-csv",
        required=True,
        help="Path to the aggregated metrics CSV (e.g., results/aggregate_metrics.csv).",
    )
    parser.add_argument(
        "--out",
        default="results/report.png",
        help="Output path for the PNG plot.",
    )
    args = parser.parse_args()

    # ---- Load and validate data -------------------------------------------------
    df = pd.read_csv(args.results_csv)

    missing = {"run", "f1"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"CSV missing required columns: {', '.join(sorted(missing))}"
        )

    # Ensure F1 is numeric (coerce errors to NaN, then drop)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df = df.dropna(subset=["f1"])

    # Sort by F1 descending for easier reading in the chart
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)

    # ---- Prepare output directory ----------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Plot -------------------------------------------------------------------
    # One chart, default matplotlib style/colors (keeps dependencies light).
    plt.figure()
    plt.bar(df["run"], df["f1"])
    plt.xticks(rotation=60, ha="right")
    plt.xlabel("Run")
    plt.ylabel("F1 score")
    plt.title("F1 score by run")
    plt.tight_layout()

    # Save and close the figure
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

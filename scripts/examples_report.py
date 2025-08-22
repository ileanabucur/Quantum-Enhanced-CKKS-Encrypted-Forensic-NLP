import argparse
import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """
    Generate example artifacts:
      1) Confusion matrix image for the baseline run.
      2) Small JSON with HE encryption timing.

    Inputs come from metrics JSON files produced by the training scripts.
    """

    parser = argparse.ArgumentParser(
        description="Create confusion-matrix and HE timing artifacts."
    )
    parser.add_argument(
        "--baseline-json",
        default="results/classic_logreg_tfidf_metrics.json",
        help="Path to the baseline metrics JSON containing 'confusion_matrix'.",
    )
    parser.add_argument(
        "--he-json",
        default="results/he_tfidf_metrics.json",
        help="Path to the HE metrics JSON containing 'encryption_ms'.",
    )
    parser.add_argument(
        "--cm-out",
        default="results/confusion_matrix.png",
        help="Output path for the confusion matrix PNG.",
    )
    parser.add_argument(
        "--he-out",
        default="results/he_timing.json",
        help="Output path for the HE timing JSON summary.",
    )
    args = parser.parse_args()

    # ---------- 1) Confusion matrix plot ----------
    if os.path.exists(args.baseline_json):
        # Load metrics from the baseline JSON
        with open(args.baseline_json, "r", encoding="utf-8") as f:
            metrics: Dict[str, Any] = json.load(f)

        # The training scripts save a nested list under 'confusion_matrix'
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"], dtype=int)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(args.cm_out) or ".", exist_ok=True)

            # Plot the confusion matrix
            plt.figure()
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix — Baseline (TF-IDF + LogReg)")
            plt.xlabel("Predicted")
            plt.ylabel("True")

            # Annotate each cell with its value for readability
            for (i, j), val in np.ndenumerate(cm):
                plt.text(j, i, int(val), ha="center", va="center")

            plt.tight_layout()
            plt.savefig(args.cm_out, dpi=200)
            plt.close()
            print(f"Saved: {args.cm_out}")
        else:
            print(
                f"Warning: 'confusion_matrix' key not found in {args.baseline_json}"
            )
    else:
        print(f"Warning: baseline metrics not found at {args.baseline_json}")

    # ---------- 2) HE encryption timing summary ----------
    if os.path.exists(args.he_json):
        with open(args.he_json, "r", encoding="utf-8") as f:
            he_metrics: Dict[str, Any] = json.load(f)

        # Grab encryption time if present; default to 0.0
        encryption_ms = float(he_metrics.get("encryption_ms", 0.0))
        summary = {"encryption_ms": round(encryption_ms, 2)}

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(args.he_out) or ".", exist_ok=True)

        # Save a minimal summary JSON
        with open(args.he_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved: {args.he_out}")
    else:
        print(f"Warning: HE metrics not found at {args.he_json}")


if __name__ == "__main__":
    main()

import argparse
import csv
import os
import random
from typing import Dict, List


# Reproducibility for the synthetic sampling below.
random.seed(42)

# Short, transparent toy sentences for a binary sensitive / non-sensitive task.
SENSITIVE: List[str] = [
    "Subject confessed to the crime in the report.",
    "Contains medical information about the victim's condition and treatment.",
    "Phone number and address of the witness are listed.",
    "Document includes bank account numbers and transaction details.",
    "Mention of undercover agent identities is present.",
    "Surveillance notes with exact timestamps and GPS coordinates.",
    "Minor's full name and school information are included.",
    "Password and login credentials were found in the file.",
    "Confidential informant code names and payouts are described.",
    "Explicit details of the alleged assault are recorded.",
]

NON_SENSITIVE: List[str] = [
    "General summary of the incident without personal identifiers.",
    "Weather conditions at the time of the scene are described.",
    "Publicly available law reference cited in the report.",
    "Officer arrived and secured the area; routine procedure.",
    "Status update: case forwarded to administrative review.",
    "High-level timeline with non-identifiable locations.",
    "Equipment checklist and maintenance log entry.",
    "Training material reference without case specifics.",
    "Meeting notes with anonymized participant roles only.",
    "Public safety announcement included in the document.",
]


def make_dataset(n: int = 600, p_sensitive: float = 0.5) -> List[Dict[str, int]]:
    """
    Create a tiny, transparent synthetic dataset.

    Each row is a dict: {"text": <sentence>, "label": 0|1}
    where label=1 denotes SENSITIVE and label=0 denotes NON_SENSITIVE.

    Args:
        n: Total number of rows to generate.
        p_sensitive: Probability of sampling from SENSITIVE vs NON_SENSITIVE.

    Returns:
        A list of rows suitable for writing with csv.DictWriter.
    """
    rows: List[Dict[str, int]] = []
    for _ in range(n):
        if random.random() < p_sensitive:
            rows.append({"text": random.choice(SENSITIVE), "label": 1})
        else:
            rows.append({"text": random.choice(NON_SENSITIVE), "label": 0})
    return rows


def main() -> None:
    """CLI entry point: generate and write a CSV with 'text' and 'label' columns."""
    parser = argparse.ArgumentParser(description="Generate a small synthetic dataset.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/samples/synthetic.csv",
        help="Output CSV path (will create parent directories if needed).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=600,
        help="Number of rows to generate.",
    )
    parser.add_argument(
        "--p-sensitive",
        type=float,
        default=0.5,
        help="Probability of sampling a sensitive example (0.0–1.0).",
    )
    args = parser.parse_args()

    # Ensure parent directory exists (handle cases where dirname is empty).
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Generate rows and write to CSV.
    rows = make_dataset(n=args.n, p_sensitive=args.p_sensitive)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

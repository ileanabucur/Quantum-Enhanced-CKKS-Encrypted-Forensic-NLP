"""
Evaluation helpers for classification tasks.

- `compute_metrics`: return accuracy, precision, recall, F1, and confusion matrix
  as a plain Python dict (JSON-serializable).
"""

from typing import Any, Dict, Iterable

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    average: str = "binary",
    zero_division: int = 0,
) -> Dict[str, Any]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: Averaging mode for precision/recall/F1.
                 Common options: "binary", "macro", "micro", "weighted".
                 Default "binary" expects two classes.
        zero_division: Sets the value to return when there is a zero division
                       (e.g., no positive predictions). Default 0.

    Returns:
        Dict with:
            - accuracy: float
            - precision: float
            - recall: float
            - f1: float
            - confusion_matrix: 2D list (e.g., [[tn, fp], [fn, tp]] for binary)
    """
    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=zero_division,
    )

    # Convert numpy array to a plain list for easy JSON serialization
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }

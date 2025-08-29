"""
Lightweight wrappers around scikit-learn's LinearSVC:
- train_linearsvc: fit a linear SVM with explicit, readable defaults
- predict_linearsvc: return hard-label predictions

Notes:
- LinearSVC optimizes a linear SVM using a liblinear-style solver.
- It does NOT expose `predict_proba`; use `decision_function` if you need scores.
- Works with sparse (e.g., TF-IDF) or dense feature matrices.
"""

from typing import Any

import numpy as np
from sklearn.svm import LinearSVC


def train_linearsvc(
    X: Any,
    y: Any,
    C: float = 1.0,
    max_iter: int = 2000,
) -> LinearSVC:
    """
    Train a linear Support Vector Classifier.

    Args:
        X: Feature matrix (scipy sparse or dense array-like).
        y: Target labels (array-like, shape [n_samples]).
        C: Inverse of regularization strength (higher = less regularization).
        max_iter: Maximum number of optimization iterations.

    Returns:
        A fitted `LinearSVC` model.
    """
    model = LinearSVC(C=C, max_iter=max_iter)
    model.fit(X, y)
    return model


def predict_linearsvc(model: LinearSVC, X: Any) -> np.ndarray:
    """
    Predict hard labels using a fitted LinearSVC model.

    Args:
        model: A fitted `LinearSVC` instance.
        X: Feature matrix to predict on.

    Returns:
        NumPy array of predicted labels.
    """
    return model.predict(X)

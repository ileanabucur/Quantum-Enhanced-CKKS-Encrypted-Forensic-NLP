"""
Lightweight wrappers around scikit-learn's LogisticRegression:
- train_logreg: fit a model with simple, explicit defaults
- predict_logreg: return hard-label predictions
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_logreg(
    X: Any,
    y: Any,
    C: float = 1.0,
    max_iter: int = 200,
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Args:
        X: Feature matrix (NumPy array or scipy sparse matrix).
        y: Target labels (array-like, shape [n_samples]).
        C: Inverse of regularization strength (higher = less regularization).
        max_iter: Maximum number of optimization iterations.

    Returns:
        A fitted `LogisticRegression` model.
    """
    model = LogisticRegression(C=C, max_iter=max_iter, solver="liblinear")
    model.fit(X, y)
    return model


def predict_logreg(model: LogisticRegression, X: Any) -> np.ndarray:
    """
    Predict hard labels using a fitted Logistic Regression model.

    Args:
        model: A fitted `LogisticRegression` instance.
        X: Feature matrix to predict on.

    Returns:
        NumPy array of predicted labels.
    """
    return model.predict(X)

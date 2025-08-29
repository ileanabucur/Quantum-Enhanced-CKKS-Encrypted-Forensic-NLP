"""
Lightweight wrappers around scikit-learn's RandomForestClassifier:
- train_rf: fit a model with explicit, readable defaults
- predict_rf: return hard-label predictions
"""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_rf(
    X: Any,
    y: Any,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X: Feature matrix (dense array-like). If you used TF-IDF (sparse),
           convert before calling (e.g., `X.toarray()`).
        y: Target labels (array-like, shape [n_samples]).
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree (None = expand until pure).
        random_state: Seed for reproducibility.

    Returns:
        A fitted `RandomForestClassifier` model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def predict_rf(model: RandomForestClassifier, X: Any) -> np.ndarray:
    """
    Predict hard labels using a fitted Random Forest model.

    Args:
        model: A fitted `RandomForestClassifier` instance.
        X: Feature matrix to predict on (dense array-like).

    Returns:
        NumPy array of predicted labels.
    """
    return model.predict(X)

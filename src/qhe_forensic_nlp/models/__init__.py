# Make subpackage exports explicit for convenient imports
from .baseline import train_logreg, predict_logreg
from .svm import train_linearsvc, predict_linearsvc
from .random_forest import train_rf, predict_rf
from .he_linear import HEEncryptedLinear
from .quantum_vqc import QuantumVQCClassifier

__all__ = [
    "train_logreg", "predict_logreg",
    "train_linearsvc", "predict_linearsvc",
    "train_rf", "predict_rf",
    "HEEncryptedLinear", "QuantumVQCClassifier",
]

"""
Tiny Variational Quantum Classifier (VQC) built with PennyLane.

Design goals:
- Keep the circuit small and laptop-friendly (default.qubit backend).
- Angle-encode (a subset of) features onto RX rotations.
- Simple variational layers = RX + RZ on each qubit, plus configurable entanglement.
- Optimize with full-batch gradient descent on binary cross-entropy.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class QuantumVQCClassifier:
    """
    Minimal VQC wrapper.

    Args:
        n_qubits: Number of qubits in the circuit.
        layers: Number of variational layers (RX/RZ per qubit + entanglement).
        epochs: Number of optimization steps (full-batch).
        lr: Learning rate for gradient descent.
        seed: RNG seed for weight initialization.
        entangle: Entanglement pattern: "chain" or "full".
    """
    n_qubits: int = 4
    layers: int = 3
    epochs: int = 40
    lr: float = 0.1
    seed: int = 42
    entangle: Literal["chain", "full"] = "chain"

    # --------------------------------- internals ---------------------------------

    def _build(self):
        """
        Lazily import PennyLane and create a CPU simulator device.
        Returns (device, qml_module).
        """
        import pennylane as qml  # local import to keep import-time light

        dev = qml.device("default.qubit", wires=self.n_qubits)
        return dev, qml

    # ----------------------------------- fit -------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the VQC with full-batch gradient descent.

        - Features are scaled and angle-encoded onto the first d = min(n_features, n_qubits) wires.
        - The model outputs an expectation value in [-1, 1], mapped to [0, 1] as a pseudo-probability.

        Args:
            X: (n_samples, n_features) dense array.
            y: (n_samples,) binary labels in {0,1}; cast to float internally.

        Returns:
            self
        """
        import pennylane as qml  # ensure available at runtime

        # Basic validation / normalization
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if set(np.unique(y)) - {0.0, 1.0}:
            raise ValueError("y must contain binary labels {0,1}.")

        np.random.seed(self.seed)
        dev, qml = self._build()

        n_features = X.shape[1]
        d = min(n_features, self.n_qubits)  # how many features we can embed

        # Variational parameters: for each layer and qubit, we keep two angles (RX, RZ)
        W = 0.01 * np.random.randn(self.layers, self.n_qubits, 2)

        def angle_embed(x: np.ndarray) -> np.ndarray:
            """
            Scale first d features into [-pi, pi] to keep rotations reasonable.
            """
            xx = x[:d]
            max_abs = np.max(np.abs(xx)) if xx.size else 0.0
            if max_abs > 0:
                xx = np.pi * xx / (max_abs + 1e-8)
            return xx

        @qml.qnode(dev)
        def circuit(x, W):
            # Encode data
            angles = angle_embed(x)
            for i in range(d):
                qml.RX(angles[i], wires=i)

            # Variational layers: local rotations + entanglement
            for l in range(self.layers):
                for q in range(self.n_qubits):
                    qml.RX(W[l, q, 0], wires=q)
                    qml.RZ(W[l, q, 1], wires=q)

                if self.entangle == "full":
                    # All-to-all CNOTs (upper-triangular)
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            qml.CNOT(wires=[i, j])
                else:
                    # Chain entanglement: 0->1->2->...->(n-1)
                    for q in range(self.n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])

            # Return expectation on one qubit; value in [-1, 1]
            return qml.expval(qml.PauliZ(0))

        def predict_proba(W, Xb: np.ndarray) -> np.ndarray:
            """
            Forward pass: map expval in [-1,1] to [0,1] via (ev+1)/2.
            """
            preds = []
            for i in range(Xb.shape[0]):
                ev = circuit(Xb[i], W)
                preds.append(0.5 * (ev + 1.0))
            return np.asarray(preds, dtype=float)

        def loss(W, Xb, yb) -> float:
            """
            Binary cross-entropy on pseudo-probabilities.
            """
            p = predict_proba(W, Xb)
            eps = 1e-7
            return -np.mean(yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps))

        # Optimize with simple gradient descent
        opt = qml.GradientDescentOptimizer(stepsize=self.lr)
        for _ in range(self.epochs):
            W, _ = opt.step_and_cost(lambda WW: loss(WW, X, y), W)

        # Save learned state
        self.W_ = W
        self._circuit = circuit
        self._d = d
        return self

    # --------------------------------- predict -----------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict hard labels (0/1) by thresholding the pseudo-probability at 0.5.

        Args:
            X: (n_samples, n_features) dense array.

        Returns:
            (n_samples,) int array of 0/1 labels.
        """
        if not hasattr(self, "_circuit") or not hasattr(self, "W_"):
            raise RuntimeError("Model is not fitted. Call `fit` first.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        preds = []
        for i in range(X.shape[0]):
            ev = self._circuit(X[i], self.W_)
            p = 0.5 * (ev + 1.0)  # map [-1,1] -> [0,1]
            preds.append(1 if p >= 0.5 else 0)
        return np.asarray(preds, dtype=int)

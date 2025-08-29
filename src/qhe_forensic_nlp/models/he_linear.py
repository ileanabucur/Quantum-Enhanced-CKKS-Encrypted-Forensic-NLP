"""
CKKS-based encrypted inference for a linear classifier (demo-friendly).

Workflow:
  1) Train a plaintext linear model to get weights (w, b).
  2) Encrypt feature vectors with TenSEAL CKKS.
  3) Compute encrypted dot-products and (demo) decrypt the scalar score.
  4) Apply a 5th-order polynomial sigmoid approximation and threshold.

Note:
  - In production, decryption should occur on the **client** side.
  - TenSEAL is imported lazily to keep this module importable without it.
"""

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np


@dataclass
class HEEncryptedLinear:
    """
    Minimal helper for CKKS-encrypted linear inference.

    Args:
        scale: CKKS global scale (precision). Higher -> more precision, larger ciphertexts.
        poly_mod_degree: Polynomial modulus degree (must be a power of two).
    """
    scale: float = 2**40
    poly_mod_degree: int = 8192

    def _ctx(self) -> Any:
        """
        Create and return a TenSEAL CKKS context with reasonable demo defaults.
        Galois keys are generated to enable vector rotations (handy for packed ops).

        Returns:
            A configured TenSEAL context.
        """
        try:
            import tenseal as ts
        except ImportError as e:
            raise ImportError(
                "TenSEAL is required for HE inference. Install with `pip install tenseal`."
            ) from e

        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_mod_degree=self.poly_mod_degree,
            coeff_mod_bit_sizes=[60, 40, 40, 60],  # demo-friendly chain
        )
        ctx.generate_galois_keys()
        ctx.global_scale = self.scale
        return ctx

    @staticmethod
    def _sigmoid_poly5(z: float) -> float:
        """
        5th-order odd polynomial approximation of the logistic sigmoid around 0.
        This avoids the true non-polynomial sigmoid, which is not HE-friendly.

        Returns:
            Approximate σ(z) in [0, 1].
        """
        return 0.5 + 0.2159198 * z - 0.008217625 * (z**3) + 0.00018256 * (z**5)

    def encrypt_batch(self, X: np.ndarray) -> Tuple[List[Any], Any]:
        """
        Encrypt a batch of feature vectors.

        Args:
            X: Feature matrix of shape (n_samples, n_features). Will be cast to float64.

        Returns:
            (encrypted_vectors, ctx)
              - encrypted_vectors: list of CKKS vectors (TenSEAL objects)
              - ctx: TenSEAL context needed for operations on those vectors
        """
        import tenseal as ts  # safe: raises if not installed
        ctx = self._ctx()

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        enc = [ts.ckks_vector(ctx, x.ravel().tolist()) for x in X]
        return enc, ctx

    def predict_batch(
        self,
        enc_batch: List[Any],
        ctx: Any,
        w: np.ndarray,
        b: float,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Encrypted inference with a linear model and polynomial sigmoid (demo).

        Args:
            enc_batch: List of CKKS vectors (as returned by `encrypt_batch`).
            ctx: TenSEAL context (unused directly here but kept for API symmetry).
            w: Weight vector (shape [n_features], dtype float64 recommended).
            b: Bias term (float).
            threshold: Probability threshold for class 1 (default 0.5).

        Returns:
            Array of hard-label predictions (0/1) with shape (n_samples,).
        """
        # Ensure weights are a flat python list for TenSEAL dot-product
        w = np.asarray(w, dtype=np.float64).ravel()
        preds: List[int] = []

        for enc_vec in enc_batch:
            # Compute <x, w> + b under encryption (ciphertext)
            enc_score = enc_vec.dot(w.tolist()) + b

            # DEMO: decrypt on the server side to obtain a scalar score.
            # In a real deployment, only the client should decrypt.
            score = enc_score.decrypt()[0]

            # Map score -> [0, 1] with a polynomial approximation to sigmoid
            p = self._sigmoid_poly5(float(score))
            preds.append(1 if p >= threshold else 0)

        return np.array(preds, dtype=int)

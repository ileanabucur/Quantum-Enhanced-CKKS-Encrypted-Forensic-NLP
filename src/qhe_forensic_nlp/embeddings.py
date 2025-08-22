"""
Embeddings utilities for simple NLP pipelines.

- TFIDFEmbeddings: thin wrapper around scikit-learn's TfidfVectorizer.
- Word2VecEmbeddings: trains a local gensim Word2Vec model and
  represents each document by the mean of its word vectors.

Both classes expose a consistent minimal interface with:
    - fit_transform(texts)
    - transform(texts)

Notes:
- `batch_clean` is used to lightly normalize input text (lowercase, strip, etc.).
- TF-IDF returns a **sparse** matrix; Word2Vec returns a **dense** NumPy array.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import batch_clean


# --------------------------------------------------------------------------- #
# TF-IDF
# --------------------------------------------------------------------------- #
@dataclass
class TFIDFEmbeddings:
    """
    Wrapper for scikit-learn's TfidfVectorizer.

    Args:
        max_features: Cap on vocabulary size (highest TF-IDF scores kept).
        vectorizer: Populated after `fit_transform`.
    """
    max_features: int = 2048
    vectorizer: Optional[TfidfVectorizer] = None

    def fit_transform(self, texts: List[str]):
        """
        Fit the vectorizer on training texts and return the TF-IDF matrix.

        Returns:
            scipy.sparse matrix of shape (n_samples, n_features).
        """
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        X = self.vectorizer.fit_transform(batch_clean(texts))
        return X

    def transform(self, texts: List[str]):
        """
        Transform new texts using the fitted vectorizer.

        Returns:
            scipy.sparse matrix of shape (n_samples, n_features).

        Raises:
            RuntimeError: if called before `fit_transform`.
        """
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted. Call `fit_transform` first.")
        return self.vectorizer.transform(batch_clean(texts))

    def get_feature_names(self) -> List[str]:
        """
        Return the learned vocabulary (feature names) as a list of strings.

        Raises:
            RuntimeError: if called before `fit_transform`.
        """
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted. Call `fit_transform` first.")
        return self.vectorizer.get_feature_names_out().tolist()


# --------------------------------------------------------------------------- #
# Word2Vec (gensim)
# --------------------------------------------------------------------------- #
@dataclass
class Word2VecEmbeddings:
    """
    Train a local Word2Vec model and embed each document by averaging
    its token vectors (ignoring OOV tokens).

    Args:
        size: Dimensionality of the word vectors.
        window: Window size for skip-gram/CBOW context.
        min_count: Minimum token frequency to be included in the vocab.
        workers: Number of worker threads (parallel training).
        model: Populated after `fit` (lazy import of gensim).
        vocab_: Cached set of tokens in the trained vocabulary.
    """
    size: int = 100
    window: int = 5
    min_count: int = 1
    workers: int = 1
    model: Optional[object] = None
    vocab_: Optional[set] = None

    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        """Very light tokenization after cleaning: whitespace split."""
        return [t.split() for t in batch_clean(texts)]

    def fit(self, texts: List[str]) -> "Word2VecEmbeddings":
        """
        Train a Word2Vec model on the provided (cleaned) corpus.

        Returns:
            self (so you can chain `.fit(...).transform(...)`).
        """
        from gensim.models import Word2Vec  # lazy import

        tokens = self._tokenize(texts)
        self.model = Word2Vec(
            tokens,
            vector_size=self.size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )
        # Cache vocabulary membership for quick lookup during transform
        self.vocab_ = set(self.model.wv.key_to_index.keys())
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Convert documents to dense vectors by averaging word vectors.

        For any document with no known tokens, returns a zero vector.

        Returns:
            np.ndarray of shape (n_samples, size).

        Raises:
            RuntimeError: if called before `fit`.
        """
        if self.model is None or self.vocab_ is None:
            raise RuntimeError("Word2Vec model not fitted. Call `fit` first.")

        tokens_list = self._tokenize(texts)
        vecs: List[np.ndarray] = []

        for tokens in tokens_list:
            if not tokens:
                vecs.append(np.zeros(self.size, dtype=np.float32))
                continue

            # Collect in-vocab word vectors
            in_vocab = [self.model.wv[w] for w in tokens if w in self.vocab_]
            if in_vocab:
                vecs.append(np.mean(in_vocab, axis=0))
            else:
                vecs.append(np.zeros(self.size, dtype=np.float32))

        return np.vstack(vecs)

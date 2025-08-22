"""
Lightweight text preprocessing utilities.

- `clean_text`: lowercase, remove non-alphanumeric characters, collapse spaces,
  and drop English stopwords (via NLTK).
- `batch_clean`: apply `clean_text` over a list of documents.

Notes:
- We lazily ensure NLTK corpora ('stopwords' and 'punkt') are available.
- Stopword removal uses the English list from NLTK.
"""

import re
from typing import List

import nltk
from nltk.corpus import stopwords

# Pre-compile regexes for speed
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_MULTISPACE = re.compile(r"\s+")


def _ensure_nltk() -> None:
    """
    Ensure required NLTK resources are available. If missing, download them.
    """
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        nltk.download("punkt")


# Initialize NLTK resources and stopword set once at import time
_ensure_nltk()
STOP = set(stopwords.words("english"))


def clean_text(t: str) -> str:
    """
    Normalize a single document:
      1) Lowercase
      2) Remove non-alphanumeric characters (keep spaces)
      3) Collapse multiple spaces
      4) Remove English stopwords

    Args:
        t: Raw text.

    Returns:
        A cleaned string suitable for simple bag-of-words pipelines.
    """
    if not isinstance(t, str):
        t = "" if t is None else str(t)

    t = t.lower()
    t = _NON_ALNUM.sub(" ", t)
    t = _MULTISPACE.sub(" ", t).strip()
    tokens = [w for w in t.split() if w not in STOP]
    return " ".join(tokens)


def batch_clean(texts: List[str]) -> List[str]:
    """
    Apply `clean_text` to a list of documents.

    Args:
        texts: List of raw text strings.

    Returns:
        List of cleaned text strings.
    """
    return [clean_text(t) for t in texts]

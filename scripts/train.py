# Make the package importable when running the script directly
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import os
import time

import joblib
import numpy as np

from qhe_forensic_nlp.utils import ensure_dir, save_json, set_seed
from qhe_forensic_nlp.data import load_csv, train_test_split_df
from qhe_forensic_nlp.embeddings import TFIDFEmbeddings, Word2VecEmbeddings
from qhe_forensic_nlp.eval import compute_metrics
from qhe_forensic_nlp.models import (
    HEEncryptedLinear,
    QuantumVQCClassifier,
    predict_linearsvc,
    predict_logreg,
    predict_rf,
    train_linearsvc,
    train_logreg,
    train_rf,
)


def build_embeddings(kind: str, args, texts_train, texts_test):
    """
    Create the embedding object and produce train/test matrices.

    Returns:
        (emb, X_train, X_test, fmt)
        - emb: fitted embedding object
        - X_train: features for training set
        - X_test: features for test set
        - fmt: "sparse" for TF-IDF, "dense" for Word2Vec
    """
    if kind == "tfidf":
        emb = TFIDFEmbeddings(max_features=args.max_features)
        X_train = emb.fit_transform(texts_train)
        X_test = emb.transform(texts_test)
        return emb, X_train, X_test, "sparse"

    elif kind == "w2v":
        emb = Word2VecEmbeddings(
            size=args.w2v_dim, window=5, min_count=1, workers=1
        )
        emb.fit(texts_train)
        X_train = emb.transform(texts_train)
        X_test = emb.transform(texts_test)
        return emb, X_train, X_test, "dense"

    else:
        raise ValueError("Unknown embedding kind")


def run_classic(args):
    """
    Train and evaluate a classical model (LogReg / LinearSVC / RandomForest)
    on the chosen embedding (TF-IDF / Word2Vec).
    """
    set_seed(args.seed)

    # Load data and split
    df = load_csv(args.data)
    tr, te = train_test_split_df(df, test_size=args.test_size, seed=args.seed)

    # Build features
    emb, X_train, X_test, fmt = build_embeddings(
        args.emb, args, tr.text.tolist(), te.text.tolist()
    )
    y_train, y_test = tr.label.values, te.label.values

    # Train + predict
    if args.model == "logreg":
        model = train_logreg(X_train, y_train, C=args.C, max_iter=args.max_iter)
        y_pred = predict_logreg(model, X_test)

    elif args.model == "linearsvc":
        model = train_linearsvc(X_train, y_train, C=args.C, max_iter=args.max_iter)
        y_pred = predict_linearsvc(model, X_test)

    elif args.model == "randomforest":
        # RF expects dense; convert TF-IDF to dense when needed
        model = train_rf(
            X_train if fmt == "dense" else X_train.toarray(),
            y_train,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.seed,
        )
        y_pred = predict_rf(model, X_test if fmt == "dense" else X_test.toarray())

    else:
        raise ValueError("Unknown model")

    # Metrics + artifacts
    metrics = compute_metrics(y_test, y_pred)
    ensure_dir(args.models_dir)
    ensure_dir(args.results_dir)

    joblib.dump(model, os.path.join(args.models_dir, f"classic_{args.model}.joblib"))
    joblib.dump(emb, os.path.join(args.models_dir, f"{args.emb}.joblib"))
    save_json(
        os.path.join(
            args.results_dir, f"classic_{args.model}_{args.emb}_metrics.json"
        ),
        metrics,
    )

    print(f"{args.model.upper()} metrics:", metrics)
    return metrics


def run_he(args):
    """
    Train a plaintext LogisticRegression to obtain weights (w, b),
    then perform **encrypted** inference on the test set using CKKS
    (via TenSEAL). We report the same metrics, plus a rough encryption time.
    """
    set_seed(args.seed)

    # Load data and split
    df = load_csv(args.data)
    tr, te = train_test_split_df(df, test_size=args.test_size, seed=args.seed)

    # Build features
    emb, X_train, X_test, fmt = build_embeddings(
        args.emb, args, tr.text.tolist(), te.text.tolist()
    )

    # CKKS prefers float64; and we need dense arrays to pack vectors
    if fmt == "sparse":
        X_train = X_train.astype(np.float64)
        X_test_arr = X_test.astype(np.float64).toarray()
    else:
        X_test_arr = X_test.astype(np.float64)

    y_train, y_test = tr.label.values, te.label.values

    # Train plaintext linear model (LogReg) -> weights for HE inference
    model = train_logreg(X_train, y_train, C=args.C, max_iter=args.max_iter)
    w = model.coef_.ravel().astype(np.float64)
    b = float(model.intercept_[0])

    # Encrypt batch and classify under encryption
    he = HEEncryptedLinear(poly_mod_degree=args.poly_mod_degree, scale=2**40)
    t0 = time.time()
    enc_batch, ctx = he.encrypt_batch(X_test_arr)
    y_pred = he.predict_batch(enc_batch, ctx, w, b, threshold=0.5)
    enc_ms = (time.time() - t0) * 1000.0

    # Metrics + artifacts
    metrics = compute_metrics(y_test, y_pred)
    ensure_dir(args.models_dir)
    ensure_dir(args.results_dir)
    save_json(
        os.path.join(args.results_dir, f"he_{args.emb}_metrics.json"),
        {**metrics, "encryption_ms": round(enc_ms, 2)},
    )

    print("HE metrics:", metrics, "| encryption_ms:", round(enc_ms, 2))
    return metrics


def run_quantum(args):
    """
    Train and evaluate a tiny VQC (PennyLane) on dense features.
    TF-IDF will be densified before training/inference.
    """
    set_seed(args.seed)

    # Load data and split
    df = load_csv(args.data)
    tr, te = train_test_split_df(df, test_size=args.test_size, seed=args.seed)

    # Build features
    emb, X_train, X_test, fmt = build_embeddings(
        args.emb, args, tr.text.tolist(), te.text.tolist()
    )

    # VQC expects dense arrays
    X_train = X_train if fmt == "dense" else X_train.toarray()
    X_test = X_test if fmt == "dense" else X_test.toarray()
    y_train, y_test = tr.label.values.astype(float), te.label.values

    # Train + predict
    clf = QuantumVQCClassifier(
        n_qubits=args.qubits,
        layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        entangle=args.entangle,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics + artifacts
    metrics = compute_metrics(y_test, y_pred)
    ensure_dir(args.models_dir)
    ensure_dir(args.results_dir)
    save_json(
        os.path.join(args.results_dir, f"quantum_{args.emb}_metrics.json"), metrics
    )

    print("Quantum metrics:", metrics)
    return metrics


def main() -> None:
    """CLI entry point: train/evaluate classic, HE, or quantum variants."""
    parser = argparse.ArgumentParser(description="Extended training CLI (EN)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(a: argparse.ArgumentParser) -> None:
        a.add_argument("--data", type=str, required=True)
        a.add_argument("--emb", type=str, default="tfidf", choices=["tfidf", "w2v"])
        a.add_argument("--max-features", type=int, default=2048)
        a.add_argument("--w2v-dim", type=int, default=100)
        a.add_argument("--test-size", type=float, default=0.25)
        a.add_argument("--seed", type=int, default=42)
        a.add_argument("--models-dir", type=str, default="models")
        a.add_argument("--results-dir", type=str, default="results")

    # classic
    c = sub.add_parser("classic", help="Train a classical baseline.")
    add_common(c)
    c.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "linearsvc", "randomforest"],
    )
    c.add_argument("--C", type=float, default=1.0)
    c.add_argument("--max-iter", type=int, default=200)
    c.add_argument("--n-estimators", type=int, default=200)
    c.add_argument("--max-depth", type=int, default=None)

    # HE
    h = sub.add_parser("he", help="Run encrypted inference (CKKS).")
    add_common(h)
    h.add_argument("--C", type=float, default=1.0)
    h.add_argument("--max-iter", type=int, default=200)
    h.add_argument("--poly-mod-degree", type=int, default=8192)

    # quantum
    q = sub.add_parser("quantum", help="Train a small VQC.")
    add_common(q)
    q.add_argument("--qubits", type=int, default=4)
    q.add_argument("--layers", type=int, default=3)
    q.add_argument("--epochs", type=int, default=40)
    q.add_argument("--lr", type=float, default=0.1)
    q.add_argument("--entangle", type=str, default="chain", choices=["chain", "full"])

    args = parser.parse_args()

    if args.cmd == "classic":
        run_classic(args)
    elif args.cmd == "he":
        run_he(args)
    elif args.cmd == "quantum":
        run_quantum(args)


if __name__ == "__main__":
    main()

# Quantum-Enhanced, CKKS-Encrypted Forensic NLP

**A compact research toolkit** for sensitive-text classification that compares:
- **Classical baselines** (LogReg, LinearSVC, RandomForest)
- **Privacy-preserving inference** with **Homomorphic Encryption (CKKS via TenSEAL)**
- A tiny **Variational Quantum Classifier (VQC)** built with PennyLane

Comes with **CLI scripts**, **unit tests**, **GitHub Pages** docs, and **pdoc** API generation.

> **HE note:** TenSEAL is optional and may not build on every OS/Python toolchain.  
> The classical and quantum paths run regardless.

---

## Why this project?

Modern forensic NLP may handle highly sensitive content. This repo shows—end to end—how to:
1) Train solid classical baselines;  
2) Perform **encrypted** linear inference with CKKS (weights in clear, features encrypted), and  
3) Explore **quantum-inspired** baselines with a small VQC—keeping everything reproducible and CI-friendly.

---

## Requirements

- **Python ≥ 3.10**
- Linux/macOS recommended. (Windows works for classical/quantum; TenSEAL support varies.)
- Install system deps for TenSEAL if you want HE:
  - Ubuntu/Debian: `build-essential cmake python3-dev`
  - macOS: `brew install cmake`

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python - <<'PY'
import nltk
nltk.download('punkt'); nltk.download('stopwords')
PY

# Data (CSV with columns: text,label)
python scripts/generate_dataset.py --out data/samples/synthetic.csv --n 600

# Baselines
python scripts/train.py classic --data data/samples/synthetic.csv --model logreg      --emb tfidf
python scripts/train.py classic --data data/samples/synthetic.csv --model linearsvc  --emb tfidf
python scripts/train.py classic --data data/samples/synthetic.csv --model randomforest --emb w2v

# Encrypted inference (CKKS)
python scripts/train.py he --data data/samples/synthetic.csv --emb tfidf

# Quantum (VQC)
python scripts/train.py quantum --data data/samples/synthetic.csv --emb tfidf --qubits 4 --layers 3 --epochs 40

# Produce example artifacts used in docs
python scripts/examples_report.py
````

**Artifacts written to `results/`:**

* `classic_*_metrics.json`, `he_*_metrics.json`, `quantum_*_metrics.json`
* `confusion_matrix.png`, `he_timing.json` (via `examples_report.py`)

---

## Dataset format

Bring your own CSV with **two columns**:

* `text` — string
* `label` — integer (0/1)

You can also synthesize a small toy set:

```bash
python scripts/generate_dataset.py --out data/samples/synthetic.csv --n 600
```

---

## What you get (at a glance)

* **Embeddings:** TF-IDF (sparse), Word2Vec (dense)
* **Models:** Logistic Regression, LinearSVC, RandomForest
* **HE path:** Train LogReg in plaintext → run **CKKS-encrypted dot products** using TenSEAL → polynomial sigmoid approx
* **Quantum path:** Small **VQC** with angle encoding, configurable entanglement (chain/full), gradient descent

---

## Reproducibility, Tests & CI

```bash
# Run tests (smoke + quick checks)
pytest -q

# Optional: coverage
pytest --cov=src --cov-report=term-missing
```

GitHub Actions workflow (`.github/workflows/ci.yml`) runs tests on pushes/PRs.
Codecov upload is included (provide `CODECOV_TOKEN` secret to enable).

---

## Documentation & GitHub Pages

Generate **API docs** with **pdoc** and serve from **/docs**:

```bash
pip install -r dev-requirements.txt
python -m pdoc --html --output-dir docs/api src/qhe_forensic_nlp
```

Enable **GitHub Pages**:
**Settings → Pages → Source = “Deploy from a branch” → Branch: `main` → Folder: `/docs`**.
Your site will serve `docs/index.md` and host pdoc at `/api/…`.

---

## Project structure

```
src/qhe_forensic_nlp/       # Package: utils, data, preprocess, embeddings, eval, models/
  models/                   # baseline, svm, random_forest, he_linear, quantum_vqc
scripts/                    # train.py, generate_dataset.py, run_experiments.py, report.py, examples_report.py
tests/                      # smoke tests for pipelines
docs/                       # GitHub Pages (Jekyll) + pdoc build output under docs/api
results/                    # metrics and figures are written here
```

---

## Design notes

* **HE (CKKS) path**

  * Trains a standard LogisticRegression to obtain `(w, b)`
  * Encrypts features with TenSEAL **client-style** API for a demo
  * Uses a **5th-order polynomial** to approximate sigmoid (HE-friendly)
  * ⚠️ Decryption happens in-process here for simplicity; in a real system it should remain **client-side**

* **Quantum (VQC) path**

  * Angle-encodes up to `n_qubits` features via `RX`
  * Variational layers: `RX` + `RZ` per qubit + **chain/full** entanglement
  * Output is `expval(PauliZ(0)) ∈ [-1,1]` → mapped to `[0,1]` then thresholded

---

## Troubleshooting

* **TenSEAL install issues**

  * Ensure `cmake` and a C++ toolchain are present
  * Try creating a clean venv and `pip install --no-cache-dir tenseal`
  * As a fallback, skip HE by running only classical/quantum paths

* **PennyLane missing**

  * `pip install pennylane` (already in requirements)

* **Sparse vs dense features**

  * RandomForest & VQC expect **dense** inputs; TF-IDF is densified where needed

---

## License

**MIT** — see `LICENSE`.

---

## Citation 

If this helped your work/applications:

```
@software{qhe_forensic_nlp,
  title   = {Quantum-Enhanced, CKKS-Encrypted Forensic NLP},
  author  = {Bucur, Ileana},
  year    = {2025},
  url     = {https://github.com/ileanabucur/quantum-he-forensic-nlp}
}
```

```
```

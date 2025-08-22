# Getting started

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Typical runs
```bash
python scripts/generate_dataset.py --out data/samples/synthetic.csv --n 600
python scripts/train.py classic --data data/samples/synthetic.csv --model logreg --emb tfidf
python scripts/train.py he --data data/samples/synthetic.csv --emb tfidf
python scripts/train.py quantum --data data/samples/synthetic.csv --emb tfidf --qubits 4 --layers 3 --epochs 40
```

## Build API docs locally (pdoc)
```bash
pip install -r dev-requirements.txt
python -m pdoc --html --output-dir docs/api src/qhe_forensic_nlp
```

Then **enable GitHub Pages**: Settings → Pages → Source = **main /docs**.

# Temporal Locations — single-file simulation

Reproduces the paper’s synthetic results with one script.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python TL_all_in_one.py --seed 1
git add -A
git commit -m "Reproducible snapshot for PRA review"
git tag -a v0.1-review -m "Review snapshot"
git push && git push --tags
python TL_all_in_one.py --no_coherence

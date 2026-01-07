# Yelp Rating Risk Monitor

An end-to-end analytics project that predicts **which restaurants are at risk of a meaningful rating drop** and explains *why* using review text signals.
Built on the Yelp Academic Dataset with reproducible notebooks and an action-oriented dashboard.

## What this project does
- Creates a **rating-drop label** and leakage-safe features from historical review behavior
- Trains and evaluates a classification model to flag **high-risk restaurants**
- Generates **root-cause categories** from review text (e.g., service, food quality, wait time)
- Produces a dashboard-ready risk table for monitoring and prioritization

## Notebooks (recommended order)
1. `notebooks/data_prep.ipynb` — Ingest + cleaning + feature engineering
2. `notebooks/rating_drop_model.ipynb` — Model training + evaluation (Random Forest)
3. `notebooks/llm_root_cause.ipynb` — Review-text root cause categorization
4. `notebooks/risk_dashboard.ipynb` — Risk scoring + dashboard outputs
5. `notebooks/end_to_end_pipeline.ipynb` — One-run end-to-end execution

## Repo structure
```text
yelp-rating-risk-monitor/
├─ notebooks/
├─ reports/
├─ src/
├─ data/
├─ results/
├─ figures/
└─ requirements.txt
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Then follow `data/README.md` to place the Yelp JSON files under `data/raw/`, and run the notebooks.

## Notes
- Raw Yelp JSON is not committed (see `.gitignore`).
- Outputs are written to `results/` and `figures/`.

## License
MIT

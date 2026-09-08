# AGENTS.md — Telecom Customer Intelligence Platform (v2)
**This is the canonical source of truth. Tech stack, architecture, and conventions all live here. Read this before every task in TASKS.md.**

## Tech stack
- Python 3.13
- pandas, numpy — data handling
- scikit-learn — LogisticRegression, RandomForestClassifier, RandomForestRegressor, KMeans, StandardScaler, train_test_split
- shap — explainability (TreeExplainer only)
- joblib — model persistence (`.pkl` files)
- fastapi + uvicorn + pydantic — API layer
- streamlit + plotly — dashboard
- pytest — sanity tests

**Explicitly NOT in this stack:** lifelines, any survival-analysis library, any causal-inference library (dowhy, econml), XGBoost (dropped for the 1-day trim — do not add it), imbalanced-learn/SMOTE, any deep learning framework, any hyperparameter-search library beyond what's already in scikit-learn (and we're not using search this round anyway).

## Architecture

```
telecom-v2/
├── PRD.md              # what to build
├── AGENTS.md           # this file — how to build it
├── CLAUDE.md            # pointer to this file
├── TASKS.md             # checkbox task list, derived from PRD
├── MEMORY.md            # first-time error observations
├── DECISIONS.md         # architecture-level decisions log
├── data/
│   ├── raw/telecom_customer_churn.csv
│   └── processed/       # generated artifacts (gitignored except .gitkeep)
├── models/               # saved .pkl files (gitignored except .gitkeep)
├── src/
│   ├── feature_engineer.py
│   ├── churn_model.py
│   ├── clv_model.py
│   ├── segmentation.py
│   ├── priority_matrix.py
│   └── run_pipeline.py
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Conventions

- Every function that trains or transforms data must have a docstring explaining the
  business reason for it (not just the mechanics) — this project's whole point is
  explainability, so undocumented code defeats the purpose.
- Every model-choice line of code should have an inline comment answering "why this
  and not something fancier" where relevant (e.g. `# class_weight='balanced' instead
  of SMOTE — keeps training data unchanged and easy to explain`).
- Feature engineering is capped at 8-10 derived features. Do not add more without
  updating PRD.md first.
- No notebook is required for this trimmed 1-day build — write directly in `src/` and
  run via `run_pipeline.py`, since notebooks cost review time we don't have today.
- Random seed = 42 everywhere, for reproducibility.
- All file paths are relative to repo root; scripts must be runnable via
  `python -m src.run_pipeline` from repo root.

## Verification command (run after every task in TASKS.md)

```bash
python -m src.run_pipeline && python -m pytest tests/ -v
```

If this command fails, the task is NOT done — fix before moving to the next task.

## Deployment specifics

- **Streamlit Community Cloud**: sign in with GitHub, connect this repo, entry point
  `dashboard/app.py`. Free tier apps sleep after ~12 hours idle — this is expected
  behavior, not a bug, and should be noted in the submitted description.
- **Render**: sign in with GitHub, "New Web Service", connect this repo, start command:
  `uvicorn api.main:app --host 0.0.0.0 --port $PORT`. Free tier cold-starts can take
  1-2 minutes on first request after idling — note this in the description too.
- **Fallback if Render setup stalls with less than 2 hours to deadline**: compute
  predictions directly inside `dashboard/app.py` (no API call), ship the dashboard
  alone, note in README that the API is built and tested locally, deployment pending.

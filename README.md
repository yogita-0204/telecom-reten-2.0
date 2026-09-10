# Telecom Customer Intelligence Platform (v2)

**Predict churn, price customer value, segment the base, and rank who to save
first — with B.Tech-level, fully explainable models.**

Rebuild of the old Cox-PH / BG-NBD / S-learner project using standard,
interview-explainable techniques on the Maven Analytics Telecom Churn dataset
(7,043 customers, 38 columns). One command reproduces the entire pipeline;
a live dashboard and API expose the results.

---

## 🚀 Live links

| Service | URL | Status |
|---|---|---|
| Streamlit dashboard (Executive Overview + Customer Explorer) | **https://telecom-reten-2-0.streamlit.app** | Live — 2 pages, real data |
| FastAPI service (`/health`, `/predict`) on Render | **https://telecom-reten-2-0.onrender.com** | Live — `/predict` verified |
| Source repository | https://github.com/yogita-0204/telecom-reten-2.0 | Public |

> **Expected free-tier behavior (not a bug):** Streamlit Community Cloud apps
> sleep after ~12 hours idle — the first visit shows a *"Yes, get this app back
> up!"* button and takes ~30 s to wake. Render free services cold-start 1–2
> minutes on the first request after idling. Pipeline artifacts ship with the
> repo (see *Artifacts & clean checkout*), so no request ever has to wait for
> model training.

---

## Key results

| Problem | Model (why) | Result |
|---|---|---|
| Churn prediction | Logistic Regression (scaled pipeline) vs Random Forest, `class_weight='balanced'` | **Deployed LR: ROC-AUC 0.9186, Recall 0.8556, F1 0.7298** (RF: ROC-AUC 0.9072, F1 0.7132) |
| Customer value (CLV) | RandomForestRegressor on total revenue | **RMSE $136.56 · MAE $89.51 · R² 0.9976** |
| Segmentation | StandardScaler + KMeans (k=4, elbow + silhouette 0.4249) | 4 named segments, 0–2,529 customers each |
| Retention priority | 2×2 median rule (churn prob × predicted CLV) | 7,043/7,043 labeled, 0 nulls |

### Business summary (from `python -m src.run_pipeline`)

- **Total customers:** 7,043 (4,720 stayed · 1,869 churned · 454 joined)
- **Predicted churners:** 3,522 (at/above median churn probability 0.3075)
- **Immediate Save:** 1,171 · Nurture 2,351 · Monitor 2,351 · Low Priority 1,170
- **CLV at risk:** $7,246,149 across the 3,522 at-risk customers

### Segments (named from k=4 centroids)

| Segment | Customers | Immediate Save |
|---|---|---|
| Loyal Premium Customers | 2,225 | 695 |
| Established Standard Customers | 2,529 | 430 |
| Loyal Budget Customers | 918 | 38 |
| New Budget Customers | 1,371 | 8 |

---

## Why these techniques

| Technique | Why this, not something fancier |
|---|---|
| **Logistic Regression + Random Forest** (`class_weight='balanced'`, default hyperparameters) | Binary churn is a classification problem; two standard models with a comparison table are fully explainable in an interview. `class_weight='balanced'` handles the 28% churn rate without SMOTE, keeping training data unchanged and easy to explain. LR is wrapped in a `StandardScaler` pipeline (MEMORY.md: raw-scale LR does not converge). |
| **RandomForestRegressor for CLV** | Predicts total revenue directly; SHAP/feature importance (tenure 0.39, tenure months 0.32, monthly charge 0.21) make every dollar traceable, unlike probabilistic BG/NBD. |
| **K-Means k=4** (elbow + silhouette sweep, k=2–8) | Silhouette peaks at k=4 (0.4249, scored on a fixed 2,000-customer sample) and the elbow bends around k=4 — the data and the PRD's four business segments agree; every customer gets exactly one named cluster. |
| **Rule-based 2×2 priority matrix** | A transparent median-threshold rule (churn prob ≥ 0.3075, predicted CLV ≥ $2,117.54) yields deterministic, exhaustive labels — honest where uplift models would need simulated treatment data. |
| **SHAP TreeExplainer** | Global summary plot on the Random Forest makes feature contributions visible per prediction. |

---

## Repository layout

```
├── src/                 # pipeline: feature engineering → churn → CLV → segments → priority
├── api/main.py          # FastAPI: GET /health, POST /predict
├── dashboard/           # Streamlit: Executive Overview + Customer Explorer (2 pages)
├── data/raw/            # telecom_customer_churn.csv (committed — clean-checkout contract)
├── data/processed/      # pipeline artifacts (committed, regenerable at seed 42)
├── models/              # *.pkl (committed, joblib-compressed; regenerable)
├── tests/               # 35 pytest sanity gates
└── requirements.txt     # pinned deps (Python 3.13)
```

## Run it locally

```bash
pip install -r requirements.txt
python -m src.run_pipeline            # reproduces every artifact (seed 42)
python -m pytest tests/ -v            # 35 sanity gates
uvicorn api.main:app --reload         # API on :8000, docs at /docs
streamlit run dashboard/app.py        # dashboard on :8501
```

## Artifacts & clean checkout

Pipeline artifacts (`data/processed/*`, `models/*`) **are committed** — the
model pickles are compressed with joblib level 3, so the whole set is ~14 MB
(`clv_model.pkl` ~11 MB instead of ~49 MB). Two reasons:

1. Both deployables serve instantly — no first-visitor pipeline run on a shared
   free-tier CPU.
2. The pipeline remains fully reproducible: `python -m src.run_pipeline` rebuilds
   every artifact from the committed raw CSV at seed 42, and the tests assert the
   committed pickles match the committed features.

If the artifacts are ever stripped from a checkout, both apps regenerate them on
first launch (`dashboard/app.py` → `_bootstrap_artifacts`, `api/main.py` →
`_bootstrap_artifacts_if_needed`), so a clean clone still boots to real numbers.

## Deployment

**Streamlit Community Cloud** (dashboard): sign in at share.streamlit.io with
GitHub, *New app* → select `yogita-0204/telecom-reten-2.0`, branch `main`,
main file `dashboard/app.py`.

> **Python version note:** Community Cloud currently forces Python 3.14 on new
> deploys and ignores `runtime.txt` / `.python-version`
> ([streamlit#15326](https://github.com/streamlit/streamlit/issues/15326)).
> Every dependency is therefore pinned to a release that publishes Linux cp314
> wheels, so the app builds on 3.13 and 3.14 alike.

**Render** (API): sign in at render.com with GitHub, *New Web Service* →
select the same repo, start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**Fallback (AGENTS.md):** the dashboard never depends on the API — it reads the
same committed artifacts directly — so even if the Render service is asleep or
unreachable, the Streamlit app alone stays fully functional.

---

*Stack: Python 3.13 · pandas · scikit-learn · SHAP · FastAPI · Streamlit · plotly · pytest*
*Dataset: Maven Analytics Telecom Customer Churn — 7,043 customers · 38 columns*
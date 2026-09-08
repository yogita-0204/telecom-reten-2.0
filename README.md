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
| Streamlit dashboard (Executive Overview + Customer Explorer) | **https://telecom-reten-2-0.streamlit.app** | Deployment pending GitHub/Streamlit sign-in — see [Deployment](#deployment) |
| FastAPI service (`/health`, `/predict`) on Render | **https://telecom-reten-2-0.onrender.com** | Deployment pending GitHub/Render sign-in — see [Deployment](#deployment) |
| Source repository | https://github.com/yogita-0204/telecom-reten-2.0 | Public |

> **Expected free-tier behavior (not a bug):** Streamlit Community Cloud apps
> sleep after ~12 hours idle and cold-start on the next visit; Render free
> services cold-start 1–2 minutes on first request after idling. On a fresh
> checkout both deployables regenerate artifacts from the committed raw CSV on
> first launch (see *Clean-checkout strategy* below), so the first request can
> take a few minutes.

---

## Key results

| Problem | Model (why) | Result |
|---|---|---|
| Churn prediction | Logistic Regression (scaled pipeline) vs Random Forest, `class_weight='balanced'` | **Deployed LR: ROC-AUC 0.9186, Recall 0.8556, F1 0.7298** (RF: ROC-AUC 0.9072, F1 0.7132) |
| Customer value (CLV) | RandomForestRegressor on total revenue | **RMSE $136.56 · MAE $89.51 · R² 0.9976** |
| Segmentation | StandardScaler + KMeans (k=4, elbow + silhouette 0.4176) | 4 named segments, 0–2,529 customers each |
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
| **K-Means k=4** (elbow + silhouette sweep, k=2–8) | Silhouette peaks at k=3; k=4 is the best ≥4 and matches the PRD's four business segments; every customer gets exactly one named cluster. |
| **Rule-based 2×2 priority matrix** | A transparent median-threshold rule (churn prob ≥ 0.3075, predicted CLV ≥ $2,117.54) yields deterministic, exhaustive labels — honest where uplift models would need simulated treatment data. |
| **SHAP TreeExplainer** | Global summary plot on the Random Forest makes feature contributions visible per prediction. |

---

## Repository layout

```
├── src/                 # pipeline: feature engineering → churn → CLV → segments → priority
├── api/main.py          # FastAPI: GET /health, POST /predict
├── dashboard/           # Streamlit: Executive Overview + Customer Explorer (2 pages)
├── data/raw/            # telecom_customer_churn.csv (committed — clean-checkout contract)
├── data/processed/      # pipeline artifacts (generated, gitignored)
├── models/              # *.pkl (generated, gitignored)
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

## Clean-checkout strategy

The generated artifacts (`data/processed/*`, `models/*`) are **not** committed —
`models/clv_model.pkl` alone is ~51 MB. Instead:

1. `data/raw/telecom_customer_churn.csv` **is** committed, so a clean clone can
   re-run the whole pipeline (PRD §7 acceptance criterion).
2. The dashboard bootstraps missing artifacts on first launch
   (`dashboard/app.py` → `_bootstrap_artifacts`, cached once per app process).
3. The API does the same on first `/predict` (`api/main.py` →
   `_bootstrap_artifacts_if_needed`), while `/health` stays artifact-free for
   probes.

With the pipeline deterministic at seed 42, bootstrapped artifacts match a
local run byte-for-byte.

## Deployment

**Streamlit Community Cloud** (dashboard): sign in at share.streamlit.io with
GitHub, *New app* → select `yogita-0204/telecom-reten-2.0`, branch `main`,
main file `dashboard/app.py`.

**Render** (API): sign in at render.com with GitHub, *New Web Service* →
select the same repo, start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**Fallback (AGENTS.md):** if the Render setup stalls before the deadline, the
dashboard already computes everything from local artifacts — it never depends
on the API — so the Streamlit app alone remains fully functional; the API stays
built and locally tested, deployment pending.

---

*Stack: Python 3.13 · pandas · scikit-learn · SHAP · FastAPI · Streamlit · plotly · pytest*
*Dataset: Maven Analytics Telecom Customer Churn — 7,043 customers · 38 columns*
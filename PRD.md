# PRD — Telecom Customer Intelligence Platform (v2)

## 1. Problem statement
The existing telecom churn project uses PhD-level techniques (Cox Proportional Hazards
survival analysis, BG/NBD probabilistic CLV, causal S-learner uplift modeling) that the
project owner cannot personally explain in a technical interview. We are rebuilding the
same 4 business capabilities using B.Tech-level, standard, fully explainable ML
techniques — without looking like a toy project.

**Hard deadline:** live link + project description due tomorrow 10:00 AM. Scope below
is already trimmed for a 1-day build. Do not add anything not listed here.

## 2. Goals
- Solve the same 4 business problems as the old project, with interpretable models
- Ship a live, working dashboard + API by the deadline
- Every technical choice must be explainable by the owner in 1-2 plain sentences

## 3. Non-goals (explicitly excluded — do not build these)
- No survival analysis (Cox PH, Kaplan-Meier)
- No probabilistic CLV (BG/NBD, Gamma-Gamma)
- No causal inference / uplift modeling (S-learner, propensity matching)
- No deep learning
- No hyperparameter search / tuning (use sensible defaults)
- No SMOTE or synthetic oversampling (use `class_weight='balanced'` instead)
- No 3rd churn model (XGBoost) — 2 models only (Logistic Regression + Random Forest)
- No 3rd dashboard page — 2 pages only

## 4. User stories

1. **As a recruiter/interviewer**, I want to open a live dashboard link and immediately
   see churn risk, customer value, and segment for the customer base, so I can judge
   the project's completeness in under a minute.
2. **As Yogita (the owner)**, I want every model's output to be traceable to specific
   features (via SHAP or coefficients), so I can explain any prediction on the spot.
3. **As a business stakeholder (framing)**, I want customers ranked into a clear
   priority list (who to save first), not just a raw probability score.

## 5. Scope — the 4 problems and what "done" looks like

| # | Problem | Built as | Done when |
|---|---|---|---|
| 1 | Churn prediction | Logistic Regression + Random Forest (`class_weight='balanced'`, default hyperparameters) | Both trained, ROC-AUC/Precision/Recall/F1 reported in a comparison table, SHAP global summary plot generated |
| 2 | Customer value (CLV) | Random Forest Regressor predicting historical total revenue | RMSE/MAE/R² reported, feature importance plot generated |
| 3 | Segmentation | K-Means (k=4, chosen via elbow + silhouette) on tenure/monthly charge/services count | Clusters generated and named from centroid characteristics |
| 4 | Retention priority | Rule-based matrix combining churn probability + predicted CLV against their medians | Every customer tagged with one of: Immediate Save / Monitor / Nurture / Low Priority |

## 6. Deliverables
- `src/run_pipeline.py` — runs all 4 problems end-to-end, saves artifacts to `data/processed/` and `models/`
- `api/main.py` — FastAPI with `/health` and `/predict` only
- `dashboard/app.py` — Streamlit with 2 pages: Executive Overview, Customer Explorer
- Live Streamlit Community Cloud link
- Live Render link (or embedded-in-Streamlit fallback if time runs out — see AGENTS.md)
- `README.md` with live links, results table, and a short "why these techniques" rationale

## 7. Acceptance criteria (must all pass before calling this "done")
- [ ] Pipeline runs top-to-bottom with zero errors on a clean checkout
- [ ] Both churn models trained, comparison table has real (not placeholder) numbers
- [ ] SHAP global summary plot renders correctly
- [ ] CLV model trained, metrics reported
- [ ] Segmentation produces 4 named clusters
- [ ] Priority matrix covers all 7,043 customers, no nulls in the priority label
- [ ] `/predict` endpoint returns a valid JSON response for a sample customer
- [ ] Dashboard's 2 pages both load without error and show real data
- [ ] Dashboard is reachable via a public URL
- [ ] README has live link(s) at the top

## 8. Dataset
Maven Analytics Telecom Customer Churn — `data/raw/telecom_customer_churn.csv`
(7,043 customers, 38 columns). Already available; no new data collection needed.

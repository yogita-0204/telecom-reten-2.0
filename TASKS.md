# TASKS.md — Telecom v2 (1-day build, deadline tomorrow 10:00 AM)

Work top to bottom. Do not skip ahead. Run the verification command (see AGENTS.md)
after each task before ticking it off. If a task's verification fails twice with the
same error, log it per the two-strikes rule (see MEMORY.md / DECISIONS.md below).

## 0. Setup (do first, before any code)
- [ ] Sign in to streamlit.io with GitHub (no separate signup)
- [ ] Sign in to render.com with GitHub (no separate signup)
- [ ] Copy `data/raw/telecom_customer_churn.csv` from the old project into this repo

## 1. Data pipeline foundation
- [ ] Write `src/feature_engineer.py` — load CSV, validate (nulls/duplicates/types),
      create the 8-10 derived features listed in PRD.md, one-hot encode categoricals
- [ ] Verify: features.csv is generated, no nulls, correct row count (7,043)

## 2. Churn model (Problem 1)
- [ ] Write `src/churn_model.py` — stratified 80/20 split, train LogisticRegression
      and RandomForestClassifier (class_weight='balanced', default hyperparameters)
- [ ] Generate comparison table: Accuracy, Precision, Recall, F1, ROC-AUC for both models
- [ ] Generate confusion matrix for the better model
- [ ] Generate SHAP TreeExplainer global summary plot for the Random Forest model
- [ ] Save model as `models/churn_model.pkl`
- [ ] Verify: comparison table has real numbers, SHAP plot renders without error

## 3. CLV model (Problem 2)
- [ ] Write `src/clv_model.py` — RandomForestRegressor predicting
      `total_revenue_to_date`, report RMSE/MAE/R², feature importance plot
- [ ] Save model as `models/clv_model.pkl`
- [ ] Verify: metrics are real numbers, not NaN

## 4. Segmentation (Problem 3)
- [ ] Write `src/segmentation.py` — StandardScaler + KMeans, elbow method (k=2-8),
      silhouette score (k=2-8), fit final model at k=4
- [ ] Name each cluster from its centroid characteristics
- [ ] Save model as `models/segmentation_model.pkl`
- [ ] Verify: 4 clusters generated, every customer assigned exactly one

## 5. Priority matrix (Problem 4)
- [ ] Write `src/priority_matrix.py` — median-threshold rule combining churn
      probability and predicted CLV into 4 labels (Immediate Save / Monitor / Nurture
      / Low Priority)
- [ ] Verify: every one of 7,043 customers has a non-null priority label

## 6. Orchestration
- [x] Write `src/run_pipeline.py` — runs steps 1-5 in order, prints business summary
      (total customers, predicted churners, customers in Immediate Save, CLV at risk)
- [x] Verify: full pipeline runs start to finish with zero errors

## ✅ MINIMUM VIABLE CHECKPOINT — everything below is only if time remains

## 7. API
- [ ] Write `api/main.py` — `/health` and `/predict` only (see PRD.md — no `/insights`
      in this trimmed scope)
- [ ] Verify: `uvicorn api.main:app --reload` runs locally, `/predict` returns valid JSON

## 8. Dashboard
- [ ] Write `dashboard/app.py` — 2 pages only: Executive Overview (KPIs, priority
      chart, model comparison table, SHAP summary plot), Customer Explorer (search +
      prediction)
- [ ] Verify: `streamlit run dashboard/app.py` loads both pages with real data

## 9. Deployment
- [ ] Push repo to GitHub
- [ ] Deploy dashboard to Streamlit Community Cloud, get live URL
- [ ] Deploy API to Render, get live URL (or apply the AGENTS.md fallback if <2 hours remain)
- [ ] Verify: both live URLs load correctly from a fresh browser tab

## 10. Documentation
- [ ] Write README.md per PRD.md deliverables section — live links at top, results
      table, "why these techniques" rationale
- [ ] Fill in the ready-to-submit project description with real numbers

## 11. Final check
- [ ] Re-read PRD.md Section 7 (Acceptance Criteria) and confirm every box is checked

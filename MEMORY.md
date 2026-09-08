# MEMORY.md — Observed issues (two-strikes rule)

**Purpose:** the FIRST time an error/gotcha shows up, log it here as a one-line
observation. Do NOT create a formal rule in AGENTS.md yet — that would add noise for
something that might be a one-off. If the SAME error happens a SECOND time, promote it
to a formal rule in AGENTS.md's Conventions section, then remove or mark it resolved
here.

Format:
```
- [DATE] Observed: <what happened> — Context: <which task/file>
```

---
- [2026-09-08] Observed: literal string 'None' written to CSV comes back as NaN on pd.read_csv (it is on pandas' default NA-token list), breaking the null-free gate — fill sentinels must avoid NA tokens ('No Offer'/'No Internet'/'No Churn') — Context: task 1, src/feature_engineer.py
- [2026-09-08] Observed: 682 accounts carry negative Monthly Charge (bill credits), which pushes naive bill-share ratios outside [0,1] — ratio features need a monthly-charge>0 guard — Context: task 1, src/feature_engineer.py (_long_distance_dependency)
- [2026-09-08] Observed: orchestrator/verifier.py task_1 gate checks lowercase data/processed/features.csv, not the 'Features.csv' spelling in TASKS.md/brief — Context: task 1 (use lowercase artifact name for cross-platform gate compliance)
- [2026-09-08] Observed: sklearn lbfgs LogisticRegression does NOT converge on raw mixed-scale features (0/1 flags vs bills in the thousands) even at max_iter=1000, and metrics drift with the iteration cutoff (ROC-AUC 0.855@100 -> 0.892@1000, still climbing) — wrap LR in a StandardScaler pipeline (fit on train split only); with scaled features the default 100 iterations converge (ROC-AUC 0.919). RF is scale-free and stays on raw features — Context: task 2, src/churn_model.py

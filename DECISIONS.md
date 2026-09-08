# DECISIONS.md — Architecture decision log

**Purpose:** when an architecture-level decision is made (not a small implementation
detail), append a versioned block here. The next agent/session reads this FIRST so the
same debate doesn't happen twice.

Format:
```
## Decision N: <title>
- Date:
- Who: Yogita + Claude (planning) / Antigravity (implementation)
- Decision:
- Why:
- Alternative considered:
```

---

## Decision 1: Replace PhD-level techniques with explainable ML
- Date: 2026-09-09
- Who: Yogita + Claude (planning)
- Decision: Rebuild the churn/CLV/segmentation/targeting pipeline using Logistic
  Regression, Random Forest, K-Means, regression-based CLV, and a rule-based priority
  matrix — instead of the old project's Cox Proportional Hazards, BG/NBD, and causal
  S-learner uplift model.
- Why: The old techniques could not be personally explained by the project owner in a
  technical interview. Interpretability is a hard requirement for this project, even
  at some cost to modeling sophistication.
- Alternative considered: Keep the old models and just study them harder before the
  interview — rejected due to the short prep timeline and depth of the underlying
  statistics required (survival analysis, probabilistic modeling, causal inference).

## Decision 2: Drop XGBoost comparison and hyperparameter tuning for this build
- Date: 2026-09-09
- Who: Yogita + Claude (planning)
- Decision: Churn model comparison is Logistic Regression vs Random Forest only, both
  trained with default hyperparameters (no RandomizedSearchCV/GridSearchCV).
- Why: Hard deadline of 1 day. A working 2-model comparison with clean metrics beats a
  3-model tuned comparison that risks not finishing.
- Alternative considered: Keep XGBoost + tuning as originally planned — deferred to a
  future iteration once the deadline pressure is gone (see the full untrimmed spec in
  TELECOM_V2_EXPLAINABLE_REBUILD_SPEC.md for the complete version).

## Decision 3: Use rule-based priority matrix instead of causal uplift modeling
- Date: 2026-09-09
- Who: Yogita + Claude (planning)
- Decision: Combine churn probability and predicted CLV against their medians into 4
  priority labels, rather than modeling causal treatment effects.
- Why: The old project's uplift model used simulated (not real) treatment data, which
  was already a known weakness. A transparent rule-based matrix is honest about what
  the data can support and is trivially explainable.
- Alternative considered: Real A/B test based uplift modeling — not possible without
  actual campaign/treatment data, which doesn't exist for this dataset.

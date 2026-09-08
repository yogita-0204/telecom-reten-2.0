"""End-to-end pipeline runner for the Telecom Customer Intelligence Platform (v2).

Runs the five TASKS.md stages in order from repo root
(`python -m src.run_pipeline`): feature engineering -> churn model -> CLV model
-> segmentation -> priority matrix. Each stage executes through its module's
own entry point (`run_*_stage` / `build_features` + `save_features`), writes
its artifacts to the established lowercase locations under ``data/processed/``
(features.csv, churn_model_comparison.csv, clv_metrics.csv,
segment_profiles.csv, customer_segments.csv, priority_matrix.csv, ...) and
``models/`` (churn_model.pkl, clv_model.pkl, segmentation_model.pkl), and this
runner prints the business summary (total customers, predicted churners,
Immediate Save count, CLV at risk) that README.md and the submitted project
description quote.

Why this module exists (business reason): one command must reproduce the whole
analysis end-to-end so the project is auditable and the PRD.md Section 7
acceptance criterion — "pipeline runs top-to-bottom with zero errors on a
clean checkout" — can be checked on a fresh clone. Chaining the stages through
their public entry points (instead of re-implementing any modeling here) keeps
every artifact owner single: whichever module creates a file is the only
module that writes it.

Stack guard: this runner fits nothing itself, derives no new features, and
introduces no library beyond what the five stage modules already use — the
pipeline is orchestration only, so the approved stack (pandas/scikit-learn/
shap/joblib, random_state=42 inside every fitted model) is preserved by
construction.
"""

from __future__ import annotations

from src.churn_model import run_churn_stage
from src.clv_model import run_clv_stage
from src.feature_engineer import (
    FEATURES_PATH,
    build_features,
    save_features,
)
from src.priority_matrix import run_priority_matrix_stage
from src.segmentation import run_segmentation_stage


def main() -> None:
    """Run every stage in TASKS.md order and print the business summary.

    Business reason: the four downstream problems all read the same stage-1
    artifact (data/processed/features.csv) and stages 3-5 consume the pkl
    files of the stages before them, so the runner executes stages strictly in
    dependency order — churn and CLV train from features.csv, segmentation
    assigns every customer from features.csv, and the priority matrix scores
    the full population with the saved churn/CLV models joined to the saved
    segment assignments. Each stage's returned summary dict is the runner's
    only source for the final numbers, so the printed summary always reflects
    what was just written to disk.
    """
    # --- Stage 1: data pipeline foundation (TASKS.md #1) --------------------
    # The null-free, model-ready table every later stage reads; saved with the
    # exact lowercase artifact name data/processed/features.csv (MEMORY.md:
    # the task-1 gate is case-sensitive).
    df = build_features()
    save_features(df)

    churned = int((df["Customer_Status"] == "Churned").sum())
    stayed = int((df["Customer_Status"] == "Stayed").sum())
    print(
        f"[pipeline] stage 1 done: {len(df)} customers "
        f"({stayed} stayed / {churned} churned), "
        f"{df.shape[1]} columns, 0 nulls -> {FEATURES_PATH}"
    )

    # --- Stage 2: churn model (TASKS.md #2) ----------------------------------
    churn = run_churn_stage()
    print(
        f"[pipeline] stage 2 done: deployed {churn['better_model']} "
        f"(ROC-AUC {churn['roc_auc']:.4f}, F1 {churn['f1']:.4f}) -> "
        f"models/churn_model.pkl"
    )

    # --- Stage 3: CLV model (TASKS.md #3) ------------------------------------
    clv = run_clv_stage()
    print(
        f"[pipeline] stage 3 done: {clv['model']} "
        f"RMSE=${clv['rmse']:,.2f} MAE=${clv['mae']:,.2f} R2={clv['r2']:.4f} "
        f"-> models/clv_model.pkl"
    )

    # --- Stage 4: segmentation (TASKS.md #4) ---------------------------------
    seg = run_segmentation_stage()
    print(
        f"[pipeline] stage 4 done: {seg['k']} named segments "
        f"(silhouette {seg['silhouette_at_k4']:.4f} at k=4) -> "
        f"models/segmentation_model.pkl"
    )

    # --- Stage 5: priority matrix (TASKS.md #5) -------------------------------
    matrix = run_priority_matrix_stage()
    counts = matrix["priority_counts"]

    # --- Business summary (TASKS.md #6) --------------------------------------
    # One coherent story for the whole base: the priority matrix is the
    # deliverable that turns the churn model into action, so its own rule
    # (churn probability at/above the population median = the at-risk set,
    # priority_matrix.py) defines 'predicted churners' here too — the same
    # customers whose predicted CLV is summed as CLV at risk. Using that
    # shared definition keeps the four quoted numbers mutually consistent,
    # unlike a 0.5 hard threshold that would count a different set than the
    # Immediate Save/Nurture quadrants do.
    total = int(matrix["customers"])
    immediate_save = int(counts["Immediate Save"])
    nurture = int(counts["Nurture"])
    predicted_churners = immediate_save + nurture
    clv_at_risk = float(matrix["at_risk_clv_dollars"])

    print("\n[pipeline] ===== BUSINESS SUMMARY =====")
    print(f"[pipeline] total customers:        {total:,}")
    print(
        f"[pipeline] predicted churners:      {predicted_churners:,} "
        "(at/above-median churn probability from the deployed "
        f"{churn['better_model']})"
    )
    print(f"[pipeline] Immediate Save count:   {immediate_save:,}")
    print(
        f"[pipeline] CLV at risk:             ${clv_at_risk:,.0f} "
        f"(predicted CLV of the {predicted_churners:,} at-risk customers)"
    )
    print(
        f"[pipeline] ============================\n"
        f"[pipeline] full label counts: "
        + ", ".join(f"{label}={counts[label]:,}" for label in counts)
    )


if __name__ == "__main__":
    main()

"""Retention priority matrix for the Telecom Customer Intelligence Platform (v2).

Stage-5 module (TASKS.md #5, PRD.md Problem 4): turns the outputs of stages
2-4 into one actionable list — who to save first. Every one of the 7,043
customers is scored with the deployed stage-2 churn model
(``models/churn_model.pkl`` -> P(churned)) and the stage-3 CLV model
(``models/clv_model.pkl`` -> predicted ``Total_Revenue`` dollars), joined with
the stage-4 segment assignment (``data/processed/customer_segments.csv``), and
tagged with exactly one of four priority labels by a transparent rule: each
score is compared against its population *median*, and the two
high/low decisions combine into the classic 2x2 retention matrix
(PRD.md Section 5, DECISIONS.md Decision 3). The complete per-customer result
is saved as ``data/processed/priority_matrix.csv`` (lowercase artifact name,
matching the ``features.csv`` convention).

Why this module exists (business reason): a churn probability alone does not
say *what to do* — saving a $12,000 account and a $100 account at the same
churn risk are different decisions with different budgets. The matrix merges
urgency (churn risk) with worth (predicted lifetime revenue) into the four
actions a retention team can actually schedule: stop the valuable leavers
first (Immediate Save), try cheap saves on the small leavers (Nurture), keep
the valuable stayers happy (Monitor), and leave the quiet tail alone
(Low Priority). That is the "clear priority list" of PRD user story 3, and the
labels are rule-based, so every label is explainable from two numbers on a
customer's record.

Why medians as thresholds (and why a plain rule at all): PRD Decision 3 chose
a rule-based matrix over causal uplift modeling — the dataset has no real
treatment/campaign data, so a treatment-effect model would be built on
simulated numbers, while a median split is honest about what the data supports
and needs no tuning (PRD Decision 2). The median is the threshold that needs
no business guess: half the customer base is above it and half below on each
axis, which reads in one sentence and is stable against the right-skewed
revenue distribution that would pull a mean threshold upward.

Boundary rule (no customer may fall through): "high" on each axis means
*at or above* the median (``>=``), so every customer resolves to a boolean on
each axis and the 2x2 rule is exhaustive — a customer sitting exactly on a
median is classified as high on that axis, never left unclassified. Scores
are thresholded at full precision and the CSV stores the same full-precision
values, so the labels reproduce exactly from the artifact.

Scope guard (PRD.md non-goals): no new model is trained here and nothing is
fitted — the stage consumes the two saved stage-2/3 models and a pandas
median/boolean-rule join, i.e. only pandas/numpy/joblib from the approved
stack. No survival analysis, no causal inference, no XGBoost, no SMOTE, no
hyperparameter search, and no new derived features are added. Random
seed 42 is not needed in this module because it introduces no randomness —
the scores come from models already fitted with seed 42, so every run of this
stage reproduces identical thresholds and labels (AGENTS.md).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.churn_model import select_predictor_columns as churn_predictor_columns
from src.clv_model import select_predictor_columns as clv_predictor_columns

FEATURES_PATH = Path("data/processed/features.csv")
SEGMENTS_PATH = Path("data/processed/customer_segments.csv")
CHURN_MODEL_PATH = Path("models/churn_model.pkl")
CLV_MODEL_PATH = Path("models/clv_model.pkl")
MATRIX_PATH = Path("data/processed/priority_matrix.csv")

# Output columns of the artifact (join key first, then the two scores the
# rule is applied to, then the segment context, then the action label).
CUSTOMER_ID_COLUMN = "Customer_ID"
CHURN_PROB_COLUMN = "Churn_Probability"
PREDICTED_CLV_COLUMN = "Predicted_CLV"
SEGMENT_COLUMN = "Segment"
PRIORITY_COLUMN = "Priority_Label"

# The four actions of the 2x2 retention matrix. PRIORITY_ORDER doubles as the
# row ranking of the saved list and the legend order: churn is the trigger
# for any save action, so the two churning quadrants lead (value then sets the
# budget — full save campaign vs. cheap nurture offers), followed by the
# valuable stayers to protect and the quiet tail to leave alone.
PRIORITY_ORDER = ["Immediate Save", "Nurture", "Monitor", "Low Priority"]
PRIORITY_LABELS = set(PRIORITY_ORDER)
EXPECTED_CUSTOMER_COUNT = 7043  # PRD.md acceptance: all 7,043 customers


# ---------------------------------------------------------------------------
# Scoring the full population with the deployed stage-2/3 models
# ---------------------------------------------------------------------------

def score_full_population(df: pd.DataFrame) -> pd.DataFrame:
    """Score all customers with the saved churn and CLV models.

    Business reason: the matrix needs a churn probability and a predicted
    lifetime value for every account, and the only artifacts that carry those
    models are models/churn_model.pkl and models/clv_model.pkl — the same
    files the API and dashboard will load. Scoring here, once, on the full
    feature table, keeps the priority list consistent with what a customer's
    record would show at prediction time.

    Drift guard: each model records the feature list it was trained on
    (feature_names_in_). If features.csv has changed since the models were
    fitted, that list no longer matches the stage's canonical predictor
    columns — a silent mismatch would rank customers by a stale model, so we
    fail loudly instead of predicting on drifted features.
    """
    churn_model = joblib.load(CHURN_MODEL_PATH)
    clv_model = joblib.load(CLV_MODEL_PATH)

    expected_churn = churn_predictor_columns(df)
    if list(churn_model.feature_names_in_) != expected_churn:
        raise ValueError(
            "models/churn_model.pkl was trained on a different feature table "
            "than features.csv — re-run the churn stage before the matrix."
        )
    expected_clv = clv_predictor_columns(df)
    if list(clv_model.feature_names_in_) != expected_clv:
        raise ValueError(
            "models/clv_model.pkl was trained on a different feature table "
            "than features.csv — re-run the CLV stage before the matrix."
        )

    # P(churned) is the positive class of the deployed classifier; the CLV
    # model predicts Total_Revenue dollars directly (the target name mapped
    # in clv_model.py).
    churn_prob = churn_model.predict_proba(df[expected_churn])[:, 1]
    predicted_clv = clv_model.predict(df[expected_clv])

    scores = pd.DataFrame(
        {
            CUSTOMER_ID_COLUMN: df[CUSTOMER_ID_COLUMN],
            CHURN_PROB_COLUMN: churn_prob,
            PREDICTED_CLV_COLUMN: predicted_clv,
        }
    )
    numeric = scores[[CHURN_PROB_COLUMN, PREDICTED_CLV_COLUMN]]
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Model scores contain NaN/inf — matrix would mislabel.")
    if not scores[CHURN_PROB_COLUMN].between(0, 1).all():
        raise ValueError("Churn probabilities outside [0, 1] — model is broken.")
    print(
        f"[priority_matrix] scored {len(scores)} customers: churn prob "
        f"[{scores[CHURN_PROB_COLUMN].min():.4f}, "
        f"{scores[CHURN_PROB_COLUMN].max():.4f}], predicted CLV "
        f"[${scores[PREDICTED_CLV_COLUMN].min():,.0f}, "
        f"${scores[PREDICTED_CLV_COLUMN].max():,.0f}]"
    )
    return scores


# ---------------------------------------------------------------------------
# Median thresholds and the 2x2 label rule
# ---------------------------------------------------------------------------

def compute_median_thresholds(scores: pd.DataFrame) -> dict[str, float]:
    """Median of each score over the full customer base.

    Business reason: the median is the self-explanatory cutoff — "half of our
    customers carry more churn risk than this, half less" — so the split is
    defensible in one sentence, needs no tuning, and is not dragged upward by
    the few very-high-revenue accounts the way a mean would be. Computed on
    the full 7,043 (descriptive rule, no split — same design as stage 4).
    """
    thresholds = {
        CHURN_PROB_COLUMN: float(scores[CHURN_PROB_COLUMN].median()),
        PREDICTED_CLV_COLUMN: float(scores[PREDICTED_CLV_COLUMN].median()),
    }
    print(
        f"[priority_matrix] median thresholds: churn "
        f"{thresholds[CHURN_PROB_COLUMN]:.4f}, predicted CLV "
        f"${thresholds[PREDICTED_CLV_COLUMN]:,.2f}"
    )
    return thresholds


def assign_priority_labels(
    scores: pd.DataFrame, thresholds: dict[str, float]
) -> pd.DataFrame:
    """Tag every customer with exactly one of the four priority labels.

    Business reason: this is the 2x2 retention decision itself — urgency on
    one axis (is churn risk at/above the base median?), worth on the other
    (is predicted CLV at/above the base median?). A high-risk high-value
    leaver is an Immediate Save (the revenue we must fight for now); a
    high-risk low-value leaver is a Nurture (still leaving, but worth only a
    cheap save offer); a low-risk high-value stayer is a Monitor (the base's
    quiet value engine to protect and grow); a low-risk low-value stayer is
    Low Priority (no campaign spend). Every quadrant is a real action, so the
    matrix doubles as the campaign planner.

    Boundary rule: 'high' means at-or-above the median on both axes, so each
    customer resolves to a boolean per axis and the four combinations cover
    every row — nobody falls through, and a customer sitting exactly on a
    median is deterministically counted as high (documented in the module
    docstring, and asserted below so a future change cannot silently unlabel
    anyone).
    """
    churn_prob = scores[CHURN_PROB_COLUMN]
    predicted_clv = scores[PREDICTED_CLV_COLUMN]
    high_churn = churn_prob >= thresholds[CHURN_PROB_COLUMN]
    high_value = predicted_clv >= thresholds[PREDICTED_CLV_COLUMN]

    labels = np.select(
        [
            high_churn & high_value,      # at risk, worth the full save effort
            high_churn & ~high_value,     # at risk, cheap-save territory
            ~high_churn & high_value,     # safe, valuable — protect
        ],
        ["Immediate Save", "Nurture", "Monitor"],
        default="Low Priority",           # the remaining low/low quadrant
    )
    labeled = scores.copy()
    labeled[PRIORITY_COLUMN] = pd.Series(labels, index=scores.index)

    # Gate (PRD acceptance): all customers labeled, none twice, only the four
    # agreed labels, labels deterministically reproducible from the columns.
    if len(labeled) != len(scores):
        raise ValueError("Labeling changed the row count.")
    if labeled[PRIORITY_COLUMN].isna().any():
        raise ValueError("Some customers were left without a priority label.")
    unknown = set(labeled[PRIORITY_COLUMN]) - PRIORITY_LABELS
    if unknown:
        raise ValueError(f"Unknown priority labels produced: {unknown}")
    return labeled


# ---------------------------------------------------------------------------
# Segment context and artifact persistence
# ---------------------------------------------------------------------------

def attach_segments(labeled: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Join the stage-4 segment name onto each customer.

    Business reason: the matrix file is also the dashboard's customer view,
    and a segment name ("Loyal Premium Customers") tells the retention team
    *what kind of account* each priority action applies to — e.g. whether
    Immediate Save offers concentrate in premium households. The join is
    one-to-one on Customer_ID against the stage-4 assignment artifact, and it
    must cover the whole base: a missing segment would mean stage 4 and
    stage 5 disagree about who the customers are.
    """
    segments = pd.read_csv(SEGMENTS_PATH)
    if len(segments) != len(df) or not segments[CUSTOMER_ID_COLUMN].is_unique:
        raise ValueError(
            "data/processed/customer_segments.csv must hold one row per "
            "customer (7,043 unique IDs) to join the matrix."
        )
    missing_cols = [c for c in (CUSTOMER_ID_COLUMN, SEGMENT_COLUMN)
                    if c not in segments.columns]
    if missing_cols:
        raise ValueError(
            f"customer_segments.csv missing join columns: {missing_cols}"
        )
    matrix = labeled.merge(
        segments[[CUSTOMER_ID_COLUMN, SEGMENT_COLUMN]],
        on=CUSTOMER_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if len(matrix) != len(labeled):
        raise ValueError("Segment join changed the customer count.")
    if matrix[SEGMENT_COLUMN].isna().any():
        raise ValueError("Customers without a segment after the join.")
    return matrix


def save_matrix(matrix: pd.DataFrame) -> None:
    """Persist the ranked priority list to data/processed/priority_matrix.csv.

    Business reason: the artifact is the gate for this task (every one of
    7,043 customers has a non-null label) and the file the README, dashboard
    and interview answers quote from, so it is sorted as an actual priority
    list — the most urgent, most valuable accounts first — and stored at full
    score precision (no rounding) so the label rule reproduces exactly from
    the file. Customer_ID and the label column are the required core; the
    scores and segment ride along so each row is self-explanatory.
    """
    rank = {label: i for i, label in enumerate(PRIORITY_ORDER)}
    ranked = matrix.copy()
    ranked["_rank"] = ranked[PRIORITY_COLUMN].map(rank)
    ranked = ranked.sort_values(
        by=["_rank", CHURN_PROB_COLUMN], ascending=[True, False]
    ).drop(columns="_rank")

    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(MATRIX_PATH, index=False)
    counts = ranked[PRIORITY_COLUMN].value_counts()
    print(
        f"[priority_matrix] wrote {MATRIX_PATH} ({len(ranked)} rows)\n"
        f"[priority_matrix] label counts: "
        + ", ".join(
            f"{label}={int(counts.get(label, 0))}" for label in PRIORITY_ORDER
        )
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def run_priority_matrix_stage() -> dict:
    """Run the full priority-matrix stage and save the artifact.

    Business reason: one entry point (a) makes the stage reproducible with a
    single command for orchestration in run_pipeline.py and (b) keeps the
    deliverable fixed — one ranked CSV joining churn risk, predicted value,
    segment and the action label for the whole customer base.
    """
    df = pd.read_csv(FEATURES_PATH)
    if len(df) != EXPECTED_CUSTOMER_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CUSTOMER_COUNT} customers in features.csv, "
            f"got {len(df)}."
        )

    scores = score_full_population(df)
    thresholds = compute_median_thresholds(scores)
    labeled = assign_priority_labels(scores, thresholds)
    matrix = attach_segments(labeled, df)

    # PRD.md Section 7 acceptance: the matrix covers all 7,043 customers and
    # no priority label is null — re-checked on the saved rows' columns.
    if len(matrix) != EXPECTED_CUSTOMER_COUNT:
        raise ValueError(f"Matrix covers {len(matrix)} customers, not 7,043.")
    if matrix[PRIORITY_COLUMN].isna().any():
        raise ValueError("Null priority labels in the final matrix.")
    if not matrix[CUSTOMER_ID_COLUMN].is_unique:
        raise ValueError("Duplicate customers in the priority matrix.")
    if set(matrix[PRIORITY_COLUMN]) != PRIORITY_LABELS:
        empty = PRIORITY_LABELS - set(matrix[PRIORITY_COLUMN])
        raise ValueError(f"Priority labels with zero customers: {empty}")

    save_matrix(matrix)

    # Business summary: where the save budget should land. 'At-risk value' is
    # the predicted CLV of everyone with at-or-above-median churn — the
    # revenue the business would lose if nothing is done this quarter.
    counts = matrix[PRIORITY_COLUMN].value_counts()
    at_risk = matrix.loc[
        matrix[PRIORITY_COLUMN].isin(["Immediate Save", "Nurture"]),
        PREDICTED_CLV_COLUMN,
    ].sum()
    print(
        f"[priority_matrix] at-risk predicted CLV "
        f"(${at_risk:,.0f} across "
        f"{int(counts.get('Immediate Save', 0) + counts.get('Nurture', 0))} "
        f"at-risk customers)"
    )
    segment_view = (
        matrix.groupby(SEGMENT_COLUMN)
        .agg(
            customers=(PRIORITY_COLUMN, "size"),
            immediate_save=(PRIORITY_COLUMN, lambda s: int((s == "Immediate Save").sum())),
        )
        .reset_index()
        .sort_values("immediate_save", ascending=False)
    )
    print(
        "[priority_matrix] Immediate Save by segment:\n"
        f"{segment_view.round(4).to_string(index=False)}"
    )

    return {
        "customers": len(matrix),
        "thresholds": thresholds,
        "priority_counts": {
            label: int(counts.get(label, 0)) for label in PRIORITY_ORDER
        },
        "at_risk_clv_dollars": float(at_risk),
        "segments": len(segment_view),
    }


def main() -> None:
    """CLI entry point: python -m src.priority_matrix"""
    summary = run_priority_matrix_stage()
    print(f"[priority_matrix] summary: {summary}")


if __name__ == "__main__":
    main()

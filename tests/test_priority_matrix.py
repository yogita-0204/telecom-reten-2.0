"""Sanity tests for the retention priority matrix stage (TASKS.md #5).

Each test encodes one acceptance line from the task brief / PRD.md Section 7:
the artifact data/processed/priority_matrix.csv covers all 7,043 customers,
no priority label is null, every label is one of the four agreed actions, the
labels follow the median-threshold 2x2 rule exactly (values at the median
count as high, so nobody falls through), customer IDs match the feature table
and the segment artifact, and the stored churn probabilities / predicted CLV
are exactly the deployed models' scores on the current feature table.

These tests deliberately avoid retraining models — the artifact is produced
by `python -m src.priority_matrix` (and later by run_pipeline.py), and this
file checks the contract it must satisfy.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.feature_engineer import (
    EXPECTED_CUSTOMER_COUNT,
    FEATURES_PATH,
    build_features,
)
from src.priority_matrix import (
    CHURN_MODEL_PATH,
    CHURN_PROB_COLUMN,
    CLV_MODEL_PATH,
    CUSTOMER_ID_COLUMN,
    MATRIX_PATH,
    PREDICTED_CLV_COLUMN,
    PRIORITY_COLUMN,
    PRIORITY_LABELS,
    SEGMENTS_PATH,
    SEGMENT_COLUMN,
)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """Rebuild the feature table from the raw CSV (source of truth)."""
    return build_features()


@pytest.fixture(scope="module")
def matrix() -> pd.DataFrame:
    """The priority artifact on disk, as every downstream consumer reads it."""
    assert MATRIX_PATH.exists(), "run python -m src.priority_matrix first"
    return pd.read_csv(MATRIX_PATH)


# ---------------------------------------------------------------------------
# PRD acceptance: full coverage, no null labels, four agreed labels
# ---------------------------------------------------------------------------

def test_artifact_exists():
    assert MATRIX_PATH.exists()


def test_matrix_covers_all_customers_with_non_null_labels(matrix):
    assert len(matrix) == EXPECTED_CUSTOMER_COUNT == 7043
    assert int(matrix[PRIORITY_COLUMN].isna().sum()) == 0, (
        "customers left without a priority label"
    )
    assert matrix[CUSTOMER_ID_COLUMN].is_unique


def test_labels_are_exactly_the_four_agreed_actions(matrix):
    assert set(matrix[PRIORITY_COLUMN]) == PRIORITY_LABELS == {
        "Immediate Save", "Nurture", "Monitor", "Low Priority"
    }
    counts = matrix[PRIORITY_COLUMN].value_counts()
    assert int(counts.sum()) == EXPECTED_CUSTOMER_COUNT
    # A 2x2 median split cannot leave a quadrant empty on 7,043 customers —
    # the dashboard's priority chart needs all four bars.
    assert (counts > 0).all()


def test_matrix_ids_match_feature_and_segment_artifacts(matrix, features):
    assert set(matrix[CUSTOMER_ID_COLUMN]) == set(features[CUSTOMER_ID_COLUMN])
    segments = pd.read_csv(SEGMENTS_PATH)
    assert set(matrix[CUSTOMER_ID_COLUMN]) == set(segments[CUSTOMER_ID_COLUMN])
    # Segment context joined cleanly: no nulls, all four stage-4 names present.
    assert int(matrix[SEGMENT_COLUMN].isna().sum()) == 0
    assert set(matrix[SEGMENT_COLUMN]) == set(segments[SEGMENT_COLUMN])


# ---------------------------------------------------------------------------
# The rule: >= median counts as high on each axis (nobody falls through)
# ---------------------------------------------------------------------------

def test_labels_reproduce_from_stored_scores_by_median_rule(matrix):
    """The label of every row must follow from its own stored scores and the
    stored median thresholds: at-or-above the median is 'high' on that axis,
    so exact-median customers are deterministically classified (never null,
    never ambiguous)."""
    churn_med = matrix[CHURN_PROB_COLUMN].median()
    clv_med = matrix[PREDICTED_CLV_COLUMN].median()
    expected = np.select(
        [
            (matrix[CHURN_PROB_COLUMN] >= churn_med)
            & (matrix[PREDICTED_CLV_COLUMN] >= clv_med),
            (matrix[CHURN_PROB_COLUMN] >= churn_med)
            & (matrix[PREDICTED_CLV_COLUMN] < clv_med),
            (matrix[CHURN_PROB_COLUMN] < churn_med)
            & (matrix[PREDICTED_CLV_COLUMN] >= clv_med),
        ],
        ["Immediate Save", "Nurture", "Monitor"],
        default="Low Priority",
    )
    assert (matrix[PRIORITY_COLUMN] == expected).all()
    # The median split is genuinely 50/50 on each axis.
    assert len(matrix) % 2 == 1  # 7,043 is odd, so the split is 3,522/3,521
    assert int((matrix[CHURN_PROB_COLUMN] >= churn_med).sum()) == 3522
    assert int((matrix[PREDICTED_CLV_COLUMN] >= clv_med).sum()) == 3522


# ---------------------------------------------------------------------------
# Model contract: scores stored in the artifact are the pkl models' output
# on the current feature table (drift guard: re-run stages 2-3, then this).
# ---------------------------------------------------------------------------

def test_stored_scores_match_deployed_models_on_current_features(matrix, features):
    churn_model = joblib.load(CHURN_MODEL_PATH)
    clv_model = joblib.load(CLV_MODEL_PATH)
    churn_prob = churn_model.predict_proba(
        features[list(churn_model.feature_names_in_)]
    )[:, 1]
    predicted_clv = clv_model.predict(features[list(clv_model.feature_names_in_)])
    assert len(churn_prob) == EXPECTED_CUSTOMER_COUNT
    # The artifact is a ranked list (not in features.csv row order), so align
    # both sides by Customer_ID before comparing scores.
    stored = matrix.set_index(CUSTOMER_ID_COLUMN)
    np.testing.assert_allclose(
        stored.loc[features[CUSTOMER_ID_COLUMN], CHURN_PROB_COLUMN].to_numpy(),
        churn_prob,
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        stored.loc[features[CUSTOMER_ID_COLUMN], PREDICTED_CLV_COLUMN].to_numpy(),
        predicted_clv,
        rtol=1e-12,
        atol=0,
    )
    # Both score columns are finite and in valid ranges.
    assert np.isfinite(matrix[[CHURN_PROB_COLUMN, PREDICTED_CLV_COLUMN]]).all().all()
    assert matrix[CHURN_PROB_COLUMN].between(0, 1).all()
    assert (matrix[PREDICTED_CLV_COLUMN] > 0).all()

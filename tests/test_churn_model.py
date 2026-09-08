"""Sanity tests for the churn model stage (TASKS.md #2).

Each test encodes one acceptance line from the task brief: the comparison
table artifact holds two models with real (0..1) metrics, the confusion
matrix artifact belongs to the model the table says is better, the SHAP
summary plot file renders as a non-trivial PNG, the saved
models/churn_model.pkl scores the full 7,043-row population with the exact
feature list it was trained on, and the training population excludes the
454 Joined customers (no churn outcome yet) while features.csv itself keeps
every row.

These tests deliberately avoid retraining models or recomputing SHAP — the
artifacts are produced by `python -m src.churn_model` (and later by
run_pipeline.py), and this file checks the contract they must satisfy.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.churn_model import (
    COMPARISON_PATH,
    CONFUSION_MATRIX_PATH,
    CONFUSION_PLOT_PATH,
    MODEL_PATH,
    NON_PREDICTOR_COLUMNS,
    SHAP_SUMMARY_PATH,
    build_training_data,
    select_predictor_columns,
)
from src.feature_engineer import (
    EXPECTED_CUSTOMER_COUNT,
    FEATURES_PATH,
    build_features,
)

TEST_SPLIT_SIZE = 1318  # 20% of the 6,589 labeled customers (Stayed + Churned)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """Rebuild the feature table from the raw CSV (source of truth)."""
    return build_features()


# ---------------------------------------------------------------------------
# Artifacts exist on disk
# ---------------------------------------------------------------------------

def test_churn_artifacts_exist():
    assert COMPARISON_PATH.exists(), "run python -m src.churn_model first"
    assert CONFUSION_MATRIX_PATH.exists()
    assert CONFUSION_PLOT_PATH.exists()
    assert SHAP_SUMMARY_PATH.exists()
    assert MODEL_PATH.exists()


def test_shap_plot_is_a_real_png():
    """A zero-byte or non-image file would mean the plot silently failed."""
    assert SHAP_SUMMARY_PATH.stat().st_size > 10_000, "PNG suspiciously small"
    with open(SHAP_SUMMARY_PATH, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n", "not a PNG header"


# ---------------------------------------------------------------------------
# Comparison table contract (the verifier gate's core check)
# ---------------------------------------------------------------------------

def test_comparison_table_has_both_models_with_real_metrics():
    df = pd.read_csv(COMPARISON_PATH)
    assert set(df["model"]) == {"Logistic Regression", "Random Forest"}
    metric_cols = [c for c in df.columns if c != "model"]
    assert metric_cols == ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = df[metric_cols].to_numpy()
    assert not np.isnan(values).any(), "metrics contain NaN"
    assert (values > 0).all() and (values <= 1).all(), "metrics outside (0, 1]"
    assert df.loc[df["model"] == "Random Forest", "roc_auc"].iloc[0] > 0.85


# ---------------------------------------------------------------------------
# Confusion matrix belongs to the model the table declares better
# ---------------------------------------------------------------------------

def test_confusion_matrix_matches_better_model_and_split_size():
    comparison = pd.read_csv(COMPARISON_PATH)
    better = comparison.sort_values(
        by=["roc_auc", "f1"], ascending=False
    ).iloc[0]["model"]
    cm = pd.read_csv(CONFUSION_MATRIX_PATH)
    assert set(cm["model"]) == {better}, (
        f"confusion matrix is for {set(cm['model'])} but the better "
        f"model per the table is {better}"
    )
    assert len(cm) == 4, "expected the 2x2 confusion cells"
    assert int(cm["count"].sum()) == TEST_SPLIT_SIZE, (
        "confusion cells must cover the whole test split"
    )


# ---------------------------------------------------------------------------
# Saved model contract: scores the full population with its own features
# ---------------------------------------------------------------------------

def test_model_pkl_scores_full_population(features):
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba"), "saved object is not a classifier"
    trained_on = list(model.feature_names_in_)
    assert trained_on == select_predictor_columns(features), (
        "saved model feature list drifted from features.csv"
    )
    X_all = features[trained_on]
    assert len(X_all) == EXPECTED_CUSTOMER_COUNT  # all 7,043 customers
    proba = model.predict_proba(X_all)
    assert proba.shape == (EXPECTED_CUSTOMER_COUNT, 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Training-population design: labels only where an outcome exists
# ---------------------------------------------------------------------------

def test_training_data_excludes_joined_but_keeps_labeled_balance(features):
    X, y = build_training_data(features)
    assert len(X) == 6589  # 4,720 stayed + 1,869 churned
    assert int(y.sum()) == 1869, "churned positives must be preserved"
    assert int((1 - y).sum()) == 4720
    # features.csv itself is untouched: full population remains for scoring.
    assert len(features) == EXPECTED_CUSTOMER_COUNT
    assert set(features["Customer_Status"]) == {"Stayed", "Churned", "Joined"}


def test_predictor_columns_exclude_identifiers_geo_and_outcomes(features):
    predictors = select_predictor_columns(features)
    assert len(predictors) == 48
    assert not (set(predictors) & NON_PREDICTOR_COLUMNS), (
        "leak/identity columns leaked into the predictors"
    )
    assert all(
        pd.api.types.is_numeric_dtype(features[c]) for c in predictors
    ), "all predictors must be numeric"


def test_saved_model_was_retrained_against_current_features(features):
    """The pkl must have been fit on the same features.csv this repo builds —
    if stage-1 changed, the pipeline must be re-run before this passes."""
    model = joblib.load(MODEL_PATH)
    assert pd.read_csv(FEATURES_PATH).shape == features.shape
    assert list(model.feature_names_in_) == select_predictor_columns(features)

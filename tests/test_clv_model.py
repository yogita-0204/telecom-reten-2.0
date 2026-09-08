"""Sanity tests for the CLV model stage (TASKS.md #3).

Each test encodes one acceptance line from the task brief: the metrics
artifact holds real, finite RMSE/MAE/R-squared values, the feature-importance
CSV + PNG exist and are internally consistent, the saved models/clv_model.pkl
is a RandomForestRegressor that scores the full 7,043-row population with the
exact feature list it was trained on, and the predictor set excludes the
target (Total_Revenue), customer identifiers and the cumulative charge lines
that are bookkeeping components of the target.

These tests deliberately avoid retraining the model — the artifacts are
produced by `python -m src.clv_model` (and later by run_pipeline.py), and this
file checks the contract they must satisfy.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.clv_model import (
    IMPORTANCE_CSV_PATH,
    IMPORTANCE_PLOT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NON_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    build_training_data,
    select_predictor_columns,
)
from src.feature_engineer import (
    EXPECTED_CUSTOMER_COUNT,
    FEATURES_PATH,
    build_features,
)

# The cumulative bill lines excluded from the predictors: Total_Revenue is a
# bookkeeping sum of them, so including them would make the model a tautology.
REVENUE_COMPONENT_COLUMNS = {
    "Total_Charges", "Total_Refunds",
    "Total_Extra_Data_Charges", "Total_Long_Distance_Charges",
}

EXPECTED_PREDICTOR_COUNT = 43  # 60 feature columns - 17 excluded (see clv_model.py)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """Rebuild the feature table from the raw CSV (source of truth)."""
    return build_features()


# ---------------------------------------------------------------------------
# Artifacts exist on disk
# ---------------------------------------------------------------------------

def test_clv_artifacts_exist():
    assert METRICS_PATH.exists(), "run python -m src.clv_model first"
    assert IMPORTANCE_CSV_PATH.exists()
    assert IMPORTANCE_PLOT_PATH.exists()
    assert MODEL_PATH.exists()


def test_importance_plot_is_a_real_png():
    """A zero-byte or non-image file would mean the plot silently failed."""
    assert IMPORTANCE_PLOT_PATH.stat().st_size > 10_000, "PNG suspiciously small"
    with open(IMPORTANCE_PLOT_PATH, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n", "not a PNG header"


# ---------------------------------------------------------------------------
# Metrics artifact contract (the verifier gate's core check)
# ---------------------------------------------------------------------------

def test_metrics_are_real_finite_numbers():
    df = pd.read_csv(METRICS_PATH)
    assert list(df.columns) == ["rmse", "mae", "r2"]
    row = df.iloc[0]
    assert np.isfinite(row[["rmse", "mae", "r2"]]).all(), "metrics not finite"
    assert row["rmse"] > 0 and row["mae"] > 0, "errors must be positive"
    # RMSE >= MAE always (RMS >= mean absolute value); both are dollar errors,
    # so they must be far below the mean revenue of ~$3,034 — anything else
    # means the model learned nothing.
    assert row["rmse"] >= row["mae"]
    assert row["rmse"] < 2000 and row["mae"] < 2000
    assert 0.9 < row["r2"] <= 1.0, "model must explain most of the variance"


# ---------------------------------------------------------------------------
# Feature-importance artifact contract
# ---------------------------------------------------------------------------

def test_importance_csv_matches_model_features(features):
    model = joblib.load(MODEL_PATH)
    imp = pd.read_csv(IMPORTANCE_CSV_PATH)
    assert list(imp.columns) == ["feature", "importance"]
    # The CSV ranks features by importance (descending), so compare as sets
    # and align each importance value by feature name against the model.
    assert set(imp["feature"]) == set(model.feature_names_in_)
    assert len(imp) == len(model.feature_names_in_)
    assert imp["importance"].is_monotonic_decreasing
    by_feature = imp.set_index("feature")["importance"]
    for feature, value in zip(model.feature_names_in_,
                              model.feature_importances_):
        assert by_feature[feature] == pytest.approx(value)
    assert imp["importance"].between(0, 1).all()
    assert imp["importance"].sum() == pytest.approx(1.0, abs=1e-6)
    # The top driver of customer value must be tenure or monthly bill size,
    # not an accident of column order.
    assert imp["feature"].iloc[0] in {"Tenure_in_Months", "Tenure_Years",
                                      "Monthly_Charge"}


# ---------------------------------------------------------------------------
# Saved model contract: scores the full population with its own features
# ---------------------------------------------------------------------------

def test_model_pkl_is_regressor_scoring_full_population(features):
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, RandomForestRegressor), (
        "saved object is not a RandomForestRegressor"
    )
    assert model.random_state == 42, "model must use the agreed seed"
    trained_on = list(model.feature_names_in_)
    assert trained_on == select_predictor_columns(features), (
        "saved model feature list drifted from features.csv"
    )
    X_all = features[trained_on]
    assert len(X_all) == EXPECTED_CUSTOMER_COUNT  # all 7,043 customers
    preds = model.predict(X_all)
    assert preds.shape == (EXPECTED_CUSTOMER_COUNT,)
    assert np.isfinite(preds).all(), "predictions contain NaN/inf"
    assert (preds > 0).all(), "predicted revenue must be positive"


def test_target_and_components_never_become_predictors(features):
    predictors = select_predictor_columns(features)
    assert len(predictors) == EXPECTED_PREDICTOR_COUNT
    assert not (set(predictors) & NON_PREDICTOR_COLUMNS), (
        "leak columns leaked into the predictors"
    )
    assert TARGET_COLUMN not in predictors
    assert "Customer_ID" not in predictors
    assert not (set(predictors) & REVENUE_COMPONENT_COLUMNS), (
        "cumulative revenue components must not be predictors (tautology)"
    )
    assert all(
        pd.api.types.is_numeric_dtype(features[c]) for c in predictors
    ), "all predictors must be numeric"


# ---------------------------------------------------------------------------
# Training-population design: every account has a revenue value
# ---------------------------------------------------------------------------

def test_training_data_covers_full_population(features):
    X, y = build_training_data(features)
    assert len(X) == EXPECTED_CUSTOMER_COUNT == 7043
    assert len(y) == EXPECTED_CUSTOMER_COUNT
    assert np.isfinite(y).all()
    assert y.name == TARGET_COLUMN
    # Unlike churn, no account is excluded: even Joined customers have been
    # billed, so the target is observed for the whole base.
    assert np.allclose(y.values, features[TARGET_COLUMN].values)


def test_saved_model_was_retrained_against_current_features(features):
    """The pkl must have been fit on the same features.csv this repo builds —
    if stage-1 changed, the pipeline must be re-run before this passes."""
    model = joblib.load(MODEL_PATH)
    assert pd.read_csv(FEATURES_PATH).shape == features.shape
    assert list(model.feature_names_in_) == select_predictor_columns(features)

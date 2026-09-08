"""Sanity tests for the data pipeline foundation (TASKS.md #1).

Each test encodes one acceptance line from the task brief: the features
artifact exists at data/processed/features.csv with exactly 7,043 rows, is
null-free, has stable column names, preserves the fields later stages need,
and respects the 8-10 derived-feature cap.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineer import (
    DERIVED_FEATURES,
    EXPECTED_CUSTOMER_COUNT,
    FEATURES_PATH,
    build_features,
)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """Rebuild the feature table from the raw CSV (source of truth)."""
    return build_features()


def test_artifact_file_exists():
    assert FEATURES_PATH.exists(), (
        f"{FEATURES_PATH} not written - run python -m src.run_pipeline"
    )


def test_feature_table_has_expected_row_count(features):
    assert len(features) == EXPECTED_CUSTOMER_COUNT == 7043


def test_feature_table_is_null_free(features):
    nulls = int(features.isna().sum().sum())
    assert nulls == 0, f"features contain {nulls} nulls"


def test_no_duplicate_customer_ids(features):
    assert features["Customer_ID"].is_unique


def test_artifact_on_disk_matches_contract():
    """The saved CSV itself must be null-free with the full population —
    later stages read the artifact, not the in-memory frame."""
    df = pd.read_csv(FEATURES_PATH)
    assert len(df) == EXPECTED_CUSTOMER_COUNT
    assert int(df.isna().sum().sum()) == 0
    assert df["Customer_ID"].is_unique


def test_downstream_required_fields_preserved(features):
    required = {
        "Customer_ID",          # join key for models, API and dashboard
        "Customer_Status",      # churn target source (Churned/Stayed/Joined)
        "Churn_Reason",         # dashboard churn-reason view
        "Total_Revenue",        # CLV regression target
        "Tenure_in_Months",     # segmentation input (PRD)
        "Monthly_Charge",       # segmentation input (PRD)
        "Total_Services",       # segmentation input: services count (PRD)
    }
    missing = required - set(features.columns)
    assert not missing, f"downstream-required columns missing: {missing}"


def test_derived_feature_count_within_cap(features):
    assert 8 <= len(DERIVED_FEATURES) <= 10
    missing = set(DERIVED_FEATURES) - set(features.columns)
    assert not missing, f"declared derived features missing: {missing}"


def test_derived_feature_value_ranges(features):
    """Business-meaningful bounds: a service count can never exceed the
    number of services offered, and flags must stay 0/1."""
    assert features["Total_Services"].between(0, 10).all()
    assert features["Premium_Services"].between(0, 4).all()
    assert features["Streaming_Services"].between(0, 3).all()
    assert set(features["Has_Phone_And_Internet"].unique()) <= {0, 1}
    assert set(features["Is_New_Customer"].unique()) <= {0, 1}
    assert features["Tenure_Years"].ge(0).all()
    assert features["Long_Distance_Dependency"].between(0, 1).all()


def test_binary_encoded_fields_are_0_or_1(features):
    binary_cols = [
        "Gender", "Married", "Phone_Service", "Multiple_Lines",
        "Internet_Service", "Online_Security", "Online_Backup",
        "Device_Protection_Plan", "Premium_Tech_Support", "Streaming_TV",
        "Streaming_Movies", "Streaming_Music", "Unlimited_Data",
        "Paperless_Billing",
    ]
    for col in binary_cols:
        assert set(features[col].unique()) <= {0, 1}, f"{col} not binary"


def test_one_hot_dummies_present(features):
    expected_dummies = {
        "Contract_One_Year", "Contract_Two_Year",
        "Payment_Method_Credit_Card", "Payment_Method_Mailed_Check",
        "Internet_Type_DSL", "Internet_Type_Fiber_Optic",
        "Internet_Type_No_Internet",
        "Offer_Offer_A", "Offer_Offer_B", "Offer_Offer_C",
        "Offer_Offer_D", "Offer_Offer_E",
    }
    missing = expected_dummies - set(features.columns)
    assert not missing, f"expected one-hot columns missing: {missing}"


def test_sentinels_survive_csv_round_trip():
    """'None' is a pandas NA token and would come back as NaN from the CSV;
    the fill sentinels must be plain English instead so the artifact on disk
    stays null-free for every downstream reader."""
    df = pd.read_csv(FEATURES_PATH)
    assert "No Offer" in set(df["Offer"])
    assert "No Internet" in set(df["Internet_Type"])
    assert "No Churn" in set(df["Churn_Category"])
    assert "No Churn" in set(df["Churn_Reason"])


def test_columns_have_stable_names(features):
    """Every column name must be a clean snake_case token so names survive
    CSV round-trips, API payloads, and SHAP plots unchanged."""
    bad = [c for c in features.columns
           if not all(ch.isalnum() or ch == "_" for ch in c)]
    assert not bad, f"unstable column names: {bad}"


def test_churn_population_counts(features):
    counts = features["Customer_Status"].value_counts().to_dict()
    assert counts["Churned"] == 1869
    assert counts["Stayed"] == 4720

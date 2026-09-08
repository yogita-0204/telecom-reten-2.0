"""FastAPI prediction API for the Telecom Customer Intelligence Platform (v2).

Task-7 module (TASKS.md #7, PRD.md Section 6): exposes exactly two endpoints
— ``GET /health`` (liveness for the Render deploy) and ``POST /predict``
(retention decisions for one customer) — loading the deployed stage-2/3/4
artifacts (``models/churn_model.pkl``, ``models/clv_model.pkl``,
``models/segmentation_model.pkl``) plus the stage-5 priority rule from
``data/processed/priority_matrix.csv`` and the stage-4 segment names from
``data/processed/segment_profiles.csv`` (lowercase artifact names, matching
the ``features.csv`` convention — MEMORY.md task-1 note).

Why this module exists (business reason): the dashboard's Customer Explorer
and any external integration need a single, auditable way to turn a raw
customer record into the four retention outputs the PRD promises — churn
probability, predicted CLV, segment and priority label. The route reuses the
stage-1 feature-engineering functions unchanged (fill documented blanks ->
encode binaries -> add the 10 derived features -> one-hot), so a prediction
here is byte-for-byte the same preprocessing the models were trained on, and
every label is traceable to the same rule the priority matrix artifact was
built with.

Scope guard (PRD.md non-goals / task brief): /health and /predict only —
no /insights, no batch scoring, no retraining, no hyperparameter search. The
request schema accepts only raw customer fields that the established pipeline
can transform; outcome columns (Customer_Status / Churn_Category /
Churn_Reason) and identifiers/geo codes (Customer_ID / City / Zip_Code /
Latitude / Longitude) are deliberately NOT requested because the pipeline's
own NON_PREDICTOR_COLUMNS contract excludes them from every model input.

Single-row encoding note: stage 1 one-hot-encodes with
``pandas.get_dummies(..., drop_first=True)`` over the full 7,043-row table,
where the dropped reference level is decided by the whole population's values.
A one-row frame cannot reuse that call directly — get_dummies would decide the
dropped level from the row's single value (e.g. Contract='One Year' would drop
'One Year' itself and emit no Contract dummies at all), silently corrupting
the input. The API therefore rebuilds the *same* dummy columns from the
stage-1 vocabulary constants (EXPECTED_VOCABULARY + ONE_HOT_FIELDS) and the
same alphabetical reference-level rule, and guards the result against the
models' recorded ``feature_names_in_`` contracts before predicting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.churn_model import MODEL_PATH as CHURN_MODEL_PATH
from src.clv_model import MODEL_PATH as CLV_MODEL_PATH
from src.feature_engineer import (
    EXPECTED_VOCABULARY,
    ONE_HOT_FIELDS,
    REQUIRED_RAW_COLUMNS,
    add_derived_features,
    encode_binary_fields,
    fill_documented_blanks,
)
from src.priority_matrix import (
    CHURN_PROB_COLUMN,
    MATRIX_PATH,
    PREDICTED_CLV_COLUMN,
    PRIORITY_COLUMN,
    assign_priority_labels,
    compute_median_thresholds,
)
from src.segmentation import (
    MODEL_PATH as SEGMENTATION_MODEL_PATH,
    PROFILES_PATH,
    SEGMENT_FEATURES,
)

app = FastAPI(
    title="Telecom Customer Intelligence Platform API",
    description=(
        "Retention decision service: churn probability, predicted CLV, "
        "segment and priority label for one customer, from the deployed "
        "stage-2..5 artifacts."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Request schema — raw customer fields the stage-1 pipeline can transform
# ---------------------------------------------------------------------------

YesNo = Literal["Yes", "No"]


class CustomerRecord(BaseModel):
    """One raw customer record, in the human-readable format of the raw CSV.

    Business reason: the request carries exactly the fields the feature
    engineering pipeline consumes *before* a customer leaves — profile, plan,
    usage and bill — in the dictionary's own vocabulary ('Yes'/'No', category
    names), so the route can apply the stage-1 transforms verbatim and the
    owner can explain any prediction from the JSON alone. Optional fields are
    only the dictionary-documented blanks (no phone -> no long-distance
    charges/multiple lines; no internet -> no add-ons/downloads; never
    accepted an offer -> no offer); passing null there reproduces the same
    repair rules that stage 1 applies to the raw CSV.
    """

    # --- profile ------------------------------------------------------------
    gender: Literal["Male", "Female"]
    age: int = Field(ge=0, le=120)
    married: YesNo
    number_of_dependents: int = Field(ge=0)
    number_of_referrals: int = Field(ge=0)

    # --- plan / tenure -------------------------------------------------------
    tenure_in_months: int = Field(ge=0)
    offer: Optional[Literal["No Offer", "Offer A", "Offer B", "Offer C",
                            "Offer D", "Offer E"]] = None
    phone_service: YesNo
    avg_monthly_long_distance_charges: Optional[float] = Field(
        default=None, ge=0
    )
    multiple_lines: Optional[YesNo] = None
    internet_service: YesNo
    internet_type: Optional[Literal["Fiber Optic", "DSL", "Cable",
                                    "No Internet"]] = None
    avg_monthly_gb_download: Optional[float] = Field(default=None, ge=0)
    online_security: Optional[YesNo] = None
    online_backup: Optional[YesNo] = None
    device_protection_plan: Optional[YesNo] = None
    premium_tech_support: Optional[YesNo] = None
    streaming_tv: Optional[YesNo] = None
    streaming_movies: Optional[YesNo] = None
    streaming_music: Optional[YesNo] = None
    unlimited_data: Optional[YesNo] = None
    contract: Literal["Month-to-Month", "One Year", "Two Year"]
    paperless_billing: YesNo
    payment_method: Literal["Bank Withdrawal", "Credit Card", "Mailed Check"]

    # --- bill ----------------------------------------------------------------
    # Monthly_Charge may be negative: 682 real accounts carry bill credits
    # (MEMORY.md task-1 note), so no ge=0 bound here.
    monthly_charge: float
    total_charges: float = Field(ge=0)
    total_refunds: float = Field(ge=0)
    total_extra_data_charges: float = Field(ge=0)
    total_long_distance_charges: float = Field(ge=0)
    # The churn model was trained with Total_Revenue among its predictors, so
    # a prediction needs it (the stage-2 NON_PREDICTOR_COLUMNS contract keeps
    # it — the CLV stage excludes it as its own target, churn does not).
    total_revenue: float = Field(ge=0)

    # optional join key echoed back in the response (never a model input)
    customer_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """The four retention outputs of PRD.md Section 5 for one customer."""

    customer_id: Optional[str] = None
    churn_probability: float = Field(ge=0, le=1)
    predicted_clv: float = Field(ge=0)
    segment: str
    priority_label: Literal["Immediate Save", "Nurture", "Monitor",
                            "Low Priority"]


# ---------------------------------------------------------------------------
# Single-row transform — the stage-1 pipeline, verbatim, plus a population-
# faithful one-hot rebuild (see module docstring for why it cannot reuse
# get_dummies directly).
# ---------------------------------------------------------------------------

# The raw columns a prediction needs: every schema field mapped to its
# exact raw-CSV spelling (snake_case lowercased: 'streaming_tv' ->
# 'Streaming_TV', 'avg_monthly_gb_download' -> 'Avg_Monthly_GB_Download'),
# plus the two outcome placeholders stage 1's blank-repair fills with the
# 'No Churn' sentinel for any prospective (never-yet-churned) customer.
_TRANSFORM_RAW_COLUMNS = [
    c for c in REQUIRED_RAW_COLUMNS
    if c not in {"Customer_ID", "City", "Zip_Code", "Latitude", "Longitude",
                 "Customer_Status"}
]
_COLUMN_BY_FIELD = {c.lower(): c for c in _TRANSFORM_RAW_COLUMNS}

# String columns must keep the raw CSV's object dtype: a column built from a
# single NaN would otherwise come out float64, and stage 1's blank-repair
# would then warn on every categorical write instead of matching the
# pipeline's dtype exactly.
_STRING_COLUMNS = [
    c for c in _TRANSFORM_RAW_COLUMNS
    if c not in {
        "Age", "Number_of_Dependents", "Number_of_Referrals",
        "Tenure_in_Months", "Avg_Monthly_Long_Distance_Charges",
        "Avg_Monthly_GB_Download", "Monthly_Charge", "Total_Charges",
        "Total_Refunds", "Total_Extra_Data_Charges",
        "Total_Long_Distance_Charges", "Total_Revenue",
    }
]

# Canonical one-hot dummy columns are the population's non-reference levels
# (same set stage 1 produces on the full table): every vocabulary value except
# the alphabetically-first (the dropped reference), with the stage-1
# non-alphanumeric->underscore renaming applied.
DUMMY_COLUMNS: list[str] = [
    re.sub(r"[^0-9A-Za-z]+", "_", f"{field}_{value}")
    for field in ONE_HOT_FIELDS
    for value in sorted(EXPECTED_VOCABULARY[field])[1:]
]


def _build_request_frame(payload: CustomerRecord) -> pd.DataFrame:
    """Turn the validated request into the stage-1 raw frame shape.

    Business reason: every stage-1 transform expects the raw CSV's exact
    column names and vocabularies, so the JSON is mapped onto that skeleton
    first (optionally filled with NaN where the dictionary says the field can
    be blank) — after this point the route runs the *same* code paths as the
    pipeline, nothing API-specific.
    """
    data = payload.model_dump(exclude={"customer_id"})
    row = {_COLUMN_BY_FIELD[f]: v for f, v in data.items()}
    out = {}
    for col in _TRANSFORM_RAW_COLUMNS:
        value = row.get(col)
        if value is None:
            value = np.nan  # documented blank -> stage-1 repair rule applies
        out[col] = value
    frame = pd.DataFrame([out])
    return frame.astype({c: object for c in _STRING_COLUMNS})


def _encode_one_hot_from_vocabulary(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the four multi-category fields from the vocabulary.

    Business reason: stage 1's ``get_dummies(drop_first=True)`` decides its
    dropped reference level from the *population* (first value alphabetically,
    e.g. 'Month-to-Month' for Contract). A one-row frame has a population of
    one, so the same call would drop the row's own value whenever it is the
    only one present (verified: Contract='One Year' yields zero Contract
    dummies), which would silently flip 'One Year' into the baseline. This
    encoder instead emits the full canonical dummy set — the vocabulary minus
    the alphabetical reference level, renamed exactly like stage 1 — setting
    1 only for the row's own category, which reproduces the training-time
    representation for any legal input.
    """
    df = df.copy()
    for field in ONE_HOT_FIELDS:
        allowed = EXPECTED_VOCABULARY[field]
        value = df[field].iloc[0]
        if value not in allowed:
            raise ValueError(
                f"Column '{field}' has unexpected value {value!r}; "
                f"expected one of {sorted(allowed)}"
            )
        if value == min(allowed):
            continue  # reference level: stays all-zero across its dummies
        dummy = re.sub(r"[^0-9A-Za-z]+", "_", f"{field}_{value}")
        if dummy not in DUMMY_COLUMNS:
            raise ValueError(
                f"Value {value!r} of '{field}' maps to dummy '{dummy}' "
                "outside the canonical dummy set — vocab drift."
            )
        df[dummy] = 1
    for dummy in DUMMY_COLUMNS:  # every non-reference column must exist (0)
        if dummy not in df.columns:
            df[dummy] = 0
    return df


def transform_customer_row(payload: CustomerRecord) -> pd.DataFrame:
    """Apply the stage-1 feature engineering to a single customer record.

    Business reason: a prediction is only explainable if the API feeds the
    models the same numbers training saw. This runs the pipeline's own
    functions in the pipeline's own order — fill documented blanks, encode
    binaries as 0/1, add the 10 derived features, one-hot from the vocabulary
    — so the derived features (Total_Services, Long_Distance_Dependency,
    Is_New_Customer, ...) and their business meanings carry over verbatim.
    """
    frame = _build_request_frame(payload)
    frame = fill_documented_blanks(frame)
    frame = encode_binary_fields(frame)
    frame = add_derived_features(frame)
    frame = _encode_one_hot_from_vocabulary(frame)
    return frame


def model_input(frame: pd.DataFrame, model: object, label: str) -> pd.DataFrame:
    """Select exactly the columns a fitted model was trained on.

    Business reason: every deployed artifact records its feature contract
    (sklearn ``feature_names_in_``); selecting by it — and failing loudly if
    the transform produced anything less — is the API's drift guard, the same
    one the priority matrix uses, so a quietly mismatched feature set can
    never reach a predict call.
    """
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{label} expects columns the transform did not produce: {missing}"
        )
    return frame[expected]


# ---------------------------------------------------------------------------
# Artifact loading (lazy, cached) — /health stays alive even if artifacts are
# missing; /predict gives a clear 503 instead of an import-time crash.
# ---------------------------------------------------------------------------

_artifacts: dict = {}


def _bootstrap_artifacts_if_needed() -> None:
    """Train and persist every artifact on a clean checkout (first /predict).

    Business reason: the repo intentionally ships without generated artifacts
    (models/*.pkl, data/processed/* — AGENTS.md convention), so a Render
    clean-git-checkout build contains only data/raw/telecom_customer_churn.csv.
    Regenerating the artifacts once here lets the API serve real predictions
    straight from a fresh clone — the pipeline is seed-42 deterministic, so
    the bootstrap artifacts are byte-identical to a local run — while /health
    stays independent of their presence, so the deploy greets probes before
    the first training pass finishes.
    """
    needed = [
        CHURN_MODEL_PATH,
        CLV_MODEL_PATH,
        SEGMENTATION_MODEL_PATH,
        PROFILES_PATH,
        MATRIX_PATH,
    ]
    if all(p.exists() for p in needed):
        return
    from src.run_pipeline import main as run_pipeline_main

    run_pipeline_main()
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Pipeline completed but artifacts still missing: {missing}"
        )


def _load_artifacts() -> dict:
    """Load the deployed models and stage-4/5 artifacts once, then cache.

    Business reason: model files are static between deployments — reloading
    them per request would waste the majority of each call's latency for zero
    freshness — so they load lazily on first use and stay cached; the explicit
    lowercase artifact paths (MEMORY.md convention) keep the API aligned with
    the pipeline's artifact contract. On a clean checkout the lazy loader
    first bootstraps the artifacts from the committed raw CSV (see
    _bootstrap_artifacts_if_needed), which is what makes a fresh Render
    deploy functional without committing generated files.
    """
    if _artifacts:
        return _artifacts
    _bootstrap_artifacts_if_needed()
    churn_model = joblib.load(CHURN_MODEL_PATH)
    clv_model = joblib.load(CLV_MODEL_PATH)
    segmentation_model = joblib.load(SEGMENTATION_MODEL_PATH)

    profiles = pd.read_csv(PROFILES_PATH)
    if "cluster" not in profiles.columns or "segment" not in profiles.columns:
        raise RuntimeError(
            f"{PROFILES_PATH} must hold 'cluster' and 'segment' columns."
        )
    segment_names = {
        int(row.cluster): str(row.segment) for row in profiles.itertuples()
    }
    if len(segment_names) != len(profiles):
        raise RuntimeError("Duplicate cluster ids in the segment profiles.")

    # Stage-5 thresholds: medians of the full base, read from the artifact the
    # pipeline itself wrote (identical values at full precision).
    matrix = pd.read_csv(MATRIX_PATH)
    thresholds = compute_median_thresholds(matrix)

    _artifacts.update(
        churn_model=churn_model,
        clv_model=clv_model,
        segmentation_model=segmentation_model,
        segment_names=segment_names,
        thresholds=thresholds,
    )
    return _artifacts


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness probe for the deployment (Render health check / verifier).

    Business reason: an infrastructure probe must never depend on model files
    being present — it answers "is the process up and able to serve" — so it
    reports process state only, which keeps the deploy healthy even while
    artifacts are being regenerated upstream.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerRecord) -> dict:
    """Score one customer: churn probability, predicted CLV, segment, label.

    Business reason: this is the single auditable path from a raw record to
    the four retention outputs PRD user story 3 promises. Churn probability
    comes from the deployed classifier's positive class, CLV dollars from the
    deployed regressor, the segment name from the stage-4 cluster assignment,
    and the priority label from the same median-threshold 2x2 rule that
    produced data/processed/priority_matrix.csv — so the API's answer matches
    the dashboard's exactly, and both are explainable from two numbers.
    """
    try:
        artifacts = _load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model or matrix artifacts not found — run "
                "python -m src.run_pipeline first. "
                f"({Path(exc.filename).name if exc.filename else exc})"
            ),
        ) from exc
    except Exception as exc:  # unreadable artifact -> 503, not a 500 crash
        raise HTTPException(
            status_code=503, detail=f"Artifact load failed: {exc}"
        ) from exc

    try:
        frame = transform_customer_row(customer)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Customer record cannot be transformed: {exc}",
        ) from exc

    churn_model = artifacts["churn_model"]
    clv_model = artifacts["clv_model"]
    segmentation_model = artifacts["segmentation_model"]

    churn_prob = float(
        churn_model.predict_proba(
            model_input(frame, churn_model, "churn_model")
        )[:, 1][0]
    )
    predicted_clv = float(
        clv_model.predict(model_input(frame, clv_model, "clv_model"))[0]
    )
    cluster = int(segmentation_model.predict(frame[SEGMENT_FEATURES])[0])
    segment_names = artifacts["segment_names"]
    if cluster not in segment_names:
        raise HTTPException(
            status_code=500,
            detail=f"Segment model returned unknown cluster {cluster}.",
        )
    segment = segment_names[cluster]

    # Same 2x2 median rule as stage 5: at-or-above the base median counts as
    # 'high' on each axis, so the label is deterministic and exhaustive.
    one_row = pd.DataFrame(
        {CHURN_PROB_COLUMN: [churn_prob], PREDICTED_CLV_COLUMN: [predicted_clv]}
    )
    labeled = assign_priority_labels(one_row, artifacts["thresholds"])
    priority = str(labeled[PRIORITY_COLUMN].iloc[0])

    return {
        "customer_id": customer.customer_id,
        "churn_probability": churn_prob,
        "predicted_clv": predicted_clv,
        "segment": segment,
        "priority_label": priority,
    }
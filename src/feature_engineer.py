"""Feature engineering for the Telecom Customer Intelligence Platform (v2).

Stage-1 contract of the whole build: this module turns the raw Maven
Analytics telecom churn CSV (7,043 customers) into a single null-free table,
``data/processed/features.csv``, that every later stage (churn, CLV,
segmentation, priority matrix) reads as its input.

Why this module exists (business reason): the four downstream problems all
need the same cleaned, model-ready view of the customer base — churn needs
contract/service signals, CLV needs the revenue record, segmentation needs
tenure/charge/service breadth, and the priority matrix needs the customer
identifier to join everything together. Building that view once, with stable
column names and documented definitions, keeps every later model explainable
and prevents each stage from silently handling missing data differently.

The pipeline is: load -> validate schema -> repair documented missing values
-> coerce types -> encode Yes/No fields as 0/1 flags -> add 10 business
features -> one-hot encode multi-category fields -> assert null-free output.
Only pandas/numpy from the approved stack are used here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/raw/telecom_customer_churn.csv")
FEATURES_PATH = Path("data/processed/features.csv")
EXPECTED_CUSTOMER_COUNT = 7043  # fixed size of the Maven telecom dataset (PRD.md)

# ---------------------------------------------------------------------------
# Schema constants — these define the contract later stages build on.
# ---------------------------------------------------------------------------

# Columns that must survive untouched (downstream join keys / targets).
CUSTOMER_ID_COLUMN = "Customer_ID"
OUTCOME_COLUMNS = ["Customer_Status", "Churn_Category", "Churn_Reason"]

# Raw numeric columns that must exist and must end up as real numbers.
NUMERIC_REQUIRED_COLUMNS = [
    "Age", "Number_of_Dependents", "Zip_Code", "Latitude", "Longitude",
    "Number_of_Referrals", "Tenure_in_Months", "Avg_Monthly_Long_Distance_Charges",
    "Avg_Monthly_GB_Download", "Monthly_Charge", "Total_Charges", "Total_Refunds",
    "Total_Extra_Data_Charges", "Total_Long_Distance_Charges", "Total_Revenue",
]

# Columns where the raw CSV legitimately contains blanks (dictionary-documented),
# because the customer does not subscribe to the product the field describes.
# Any blank OUTSIDE this set is a data-quality failure, not a missing value.
DOCUMENTED_BLANK_COLUMNS = {
    "Offer",                        # never accepted an offer -> 'None'
    "Avg_Monthly_Long_Distance_Charges",  # no phone service -> 0
    "Multiple_Lines",               # no phone service -> 'No'
    "Internet_Type",                # no internet service -> 'None'
    "Avg_Monthly_GB_Download",      # no internet service -> 0
    "Online_Security", "Online_Backup", "Device_Protection_Plan",
    "Premium_Tech_Support", "Streaming_TV", "Streaming_Movies",
    "Streaming_Music", "Unlimited_Data",  # no internet service -> 'No'
    "Churn_Category", "Churn_Reason",     # never churned -> 'None'
}

# Yes/No fields encoded as 0/1 flags. Kept as plain 0/1 (not one-hot pairs):
# a one-hot pair for a binary field is redundant, and 0/1 reads cleanly in
# SHAP plots and regression coefficients when we explain a prediction.
YES_NO_FIELDS = [
    "Married", "Phone_Service", "Multiple_Lines", "Internet_Service",
    "Online_Security", "Online_Backup", "Device_Protection_Plan",
    "Premium_Tech_Support", "Streaming_TV", "Streaming_Movies",
    "Streaming_Music", "Unlimited_Data", "Paperless_Billing",
]

# Multi-category fields -> one-hot encoded (baseline category dropped so
# every remaining coefficient is a contrast vs. a natural reference group).
ONE_HOT_FIELDS = ["Offer", "Internet_Type", "Contract", "Payment_Method"]

# Reason vocabulary for churners only (raw file; non-churners are blank and
# get the 'No Churn' sentinel during fill).
CHURN_CATEGORIES = {"Attitude", "Competitor", "Dissatisfaction", "Other", "Price"}

# Expected vocabulary per categorical column (validated before encoding so a
# typo like 'yes ' or 'Y' fails loudly at the source instead of silently
# producing a new dummy column). Note: the data dictionary spells the
# "no offer / no internet" cases as 'None', but the literal string 'None' is
# on pandas' default NA-token list, so a CSV round-trip would silently turn
# it back into a missing value. The plain-English sentinels below keep the
# same meaning and survive round-trips.
EXPECTED_VOCABULARY = {
    "Gender": {"Male", "Female"},
    "Customer_Status": {"Stayed", "Churned", "Joined"},
    "Contract": {"Month-to-Month", "One Year", "Two Year"},
    "Payment_Method": {"Bank Withdrawal", "Credit Card", "Mailed Check"},
    "Internet_Type": {"Fiber Optic", "DSL", "Cable", "No Internet"},
    "Offer": {"No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"},
}

# The 10 business features derived in this module (AGENTS.md cap: 8-10).
# Listed in the order they are appended to the table.
DERIVED_FEATURES = [
    "Total_Services",
    "Premium_Services",
    "Streaming_Services",
    "Has_Phone_And_Internet",
    "Monthly_Charge_per_Service",
    "Tenure_Years",
    "Referrals_per_Tenure_Year",
    "Long_Distance_Dependency",
    "Is_New_Customer",
    "Senior_Citizen",
]

# Full expected column list after snake-casing the raw header (38 columns).
REQUIRED_RAW_COLUMNS = [
    "Customer_ID", "Gender", "Age", "Married", "Number_of_Dependents",
    "City", "Zip_Code", "Latitude", "Longitude", "Number_of_Referrals",
    "Tenure_in_Months", "Offer", "Phone_Service",
    "Avg_Monthly_Long_Distance_Charges", "Multiple_Lines", "Internet_Service",
    "Internet_Type", "Avg_Monthly_GB_Download", "Online_Security",
    "Online_Backup", "Device_Protection_Plan", "Premium_Tech_Support",
    "Streaming_TV", "Streaming_Movies", "Streaming_Music", "Unlimited_Data",
    "Contract", "Paperless_Billing", "Payment_Method", "Monthly_Charge",
    "Total_Charges", "Total_Refunds", "Total_Extra_Data_Charges",
    "Total_Long_Distance_Charges", "Total_Revenue", "Customer_Status",
    "Churn_Category", "Churn_Reason",
]


# ---------------------------------------------------------------------------
# Load + validation
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw telecom CSV and normalize its header to snake_case.

    Business reason: every downstream artifact, model pickled later, and API
    payload key must refer to one canonical spelling of each field, so the
    header is normalized here, once, before any validation or modeling.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def validate_raw_data(df: pd.DataFrame) -> dict:
    """Validate row count, duplicates, required columns, types, and vocabulary.

    Business reason: the models and the dashboard promise "7,043 customers,
    every one with a non-null prediction". If the raw file drifts from the
    documented schema (wrong row count, duplicated IDs, blank values in
    fields that must always be populated, unknown category strings), every
    downstream number would be silently wrong — so we fail fast here instead.
    """
    report: dict = {}
    report["rows"] = len(df)
    if len(df) != EXPECTED_CUSTOMER_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CUSTOMER_COUNT} customers, got {len(df)}. "
            "Refusing to build features on an unexpected population."
        )

    missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw file missing required columns: {missing_cols}")

    extra_cols = [c for c in df.columns if c not in REQUIRED_RAW_COLUMNS]
    if extra_cols:
        raise ValueError(f"Raw file has unexpected columns: {extra_cols}")

    dup_ids = int(df[CUSTOMER_ID_COLUMN].duplicated().sum())
    report["duplicate_customer_ids"] = dup_ids
    if dup_ids:
        raise ValueError(f"Found {dup_ids} duplicated customer IDs.")

    # Type check: numeric fields must coerce cleanly to numbers. Embedded
    # junk (e.g. '1,234' or ' ') would otherwise poison every model silently.
    for col in NUMERIC_REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            new_blanks = int(coerced.isna().sum() - df[col].isna().sum())
            if new_blanks:
                raise ValueError(
                    f"Column '{col}' contains {new_blanks} non-numeric values "
                    "that cannot be parsed."
                )
            df[col] = coerced

    # Blank check: blanks may only appear where the data dictionary says the
    # customer simply does not subscribe to the underlying product.
    blank_cols = df.columns[df.isna().any()].tolist()
    report["blank_columns_before_fill"] = {
        c: int(df[c].isna().sum()) for c in blank_cols
    }
    unexpected_blanks = set(blank_cols) - DOCUMENTED_BLANK_COLUMNS
    if unexpected_blanks:
        raise ValueError(
            f"Unexpected blanks in columns that must always be populated: "
            f"{sorted(unexpected_blanks)}"
        )

    # Vocabulary check for categorical fields.
    for col, allowed in EXPECTED_VOCABULARY.items():
        present = set(df[col].dropna().unique())
        unknown = present - allowed
        if unknown:
            raise ValueError(
                f"Column '{col}' contains unexpected values: {sorted(unknown)}"
            )

    # Churn category vocabulary (blank for non-churners is allowed).
    unknown_cats = set(df["Churn_Category"].dropna().unique()) - CHURN_CATEGORIES
    if unknown_cats:
        raise ValueError(
            f"Column 'Churn_Category' contains unexpected values: "
            f"{sorted(unknown_cats)}"
        )

    report["customer_status_counts"] = (
        df["Customer_Status"].value_counts().to_dict()
    )
    return report


# ---------------------------------------------------------------------------
# Missing-value repair (documented business rules only)
# ---------------------------------------------------------------------------

def fill_documented_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the dictionary-documented blanks with their stated meaning.

    Business reason: every blank in this CSV is informative — it means "the
    customer does not have this product" — and the data dictionary tells us
    exactly what value to substitute (0 for no-phone long-distance charges,
    'No' for add-ons the customer cannot buy without internet, 'No Offer' for
    an offer never accepted, 'No Churn' for a reason that does not exist
    because the customer never left). Encoding that meaning explicitly keeps
    the feature table null-free without inventing information.

    Sentinel strings deliberately avoid 'None': the literal 'None' is on
    pandas' default NA-token list, so a CSV round-trip would silently turn it
    back into a missing value and break the null-free contract downstream.
    """
    df = df.copy()

    # No phone service -> no long-distance charges, no second line.
    df.loc[df["Phone_Service"] == "No", "Avg_Monthly_Long_Distance_Charges"] = 0.0
    df.loc[df["Phone_Service"] == "No", "Multiple_Lines"] = "No"

    # No internet service -> no downloads, no internet add-ons, type
    # 'No Internet' (dictionary meaning of 'None', in round-trip-safe words).
    no_internet = df["Internet_Service"] == "No"
    df.loc[no_internet, "Avg_Monthly_GB_Download"] = 0.0
    df.loc[no_internet, "Internet_Type"] = "No Internet"
    internet_addon_cols = [
        "Online_Security", "Online_Backup", "Device_Protection_Plan",
        "Premium_Tech_Support", "Streaming_TV", "Streaming_Movies",
        "Streaming_Music", "Unlimited_Data",
    ]
    df.loc[no_internet, internet_addon_cols] = "No"

    # Offer never accepted -> 'No Offer'.
    df["Offer"] = df["Offer"].fillna("No Offer")

    # Customer never churned -> no category/reason (kept as strings for the
    # dashboard's churn-reason view; filled, not dropped, so the table is
    # null-free and every row remains traceable to the raw record).
    df["Churn_Category"] = df["Churn_Category"].fillna("No Churn")
    df["Churn_Reason"] = df["Churn_Reason"].fillna("No Churn")

    if df.isna().any().any():
        raise ValueError(
            f"Unrepaired blanks remain: {dict(df.isna().sum()[df.isna().sum() > 0])}"
        )
    return df


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_binary_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Yes/No fields (and Gender) to 0/1 integer flags.

    Business reason: models need numbers, and for an interviewer-facing
    project the flags must stay self-explanatory — 'Phone_Service = 1' means
    "subscribes to phone service". A plain 0/1 flag is the explainable choice
    for binary fields; one-hot encoding them would only add a redundant
    column of the exact opposite information.
    """
    df = df.copy()
    for col in YES_NO_FIELDS:
        unknown = set(df[col].unique()) - {"Yes", "No"}
        if unknown:
            raise ValueError(f"'{col}' has unexpected values: {unknown}")
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype(np.int8)

    # Gender encoded Male=1/Female=0 (single binary flag, same rationale).
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).astype(np.int8)
    return df


def one_hot_encode_multi_category(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Offer, Internet Type, Contract, and Payment Method.

    Business reason: these fields have 3-6 unordered categories each, and
    their distinctions (two-year vs. month-to-month contracts, bank
    withdrawal vs. mailed checks) are exactly the signals churn and CLV
    models should separate. The first category alphabetically is dropped as
    the reference level — 'No Offer' for Offer, 'Month-to-Month' for
    Contract, 'Bank Withdrawal' for Payment Method, 'Cable' for Internet
    Type — so every coefficient a model learns is a plain contrast against
    that baseline customer, which keeps coefficients explainable in the
    interview.
    """
    df = df.copy()
    dummies = pd.get_dummies(
        df[ONE_HOT_FIELDS], prefix_sep="_", drop_first=True
    ).astype(np.int8)
    # Stable dummy names: 'Contract_Month-to-Month' -> 'Contract_Month_to_Month',
    # so the same column names survive CSV round-trips and API payloads.
    dummies.columns = [re.sub(r"[^0-9A-Za-z]+", "_", c) for c in dummies.columns]
    return pd.concat([df, dummies], axis=1)


# ---------------------------------------------------------------------------
# Derived features (one function per feature, AGENTS.md cap: 8-10)
# ---------------------------------------------------------------------------

def _total_services(df: pd.DataFrame) -> pd.Series:
    """Total Services (0-10): breadth of subscriptions the customer pays for.

    Business reason: every extra service is a switching cost — leaving means
    replacing ten products, not one — and breadth also drives monthly value,
    so this is a core input for churn, CLV, and segmentation alike.
    """
    service_cols = [
        "Phone_Service", "Internet_Service", "Online_Security",
        "Online_Backup", "Device_Protection_Plan", "Premium_Tech_Support",
        "Streaming_TV", "Streaming_Movies", "Streaming_Music", "Unlimited_Data",
    ]
    return df[service_cols].sum(axis=1).astype(np.int8)


def _premium_services(df: pd.DataFrame) -> pd.Series:
    """Premium Services (0-4): paid protection add-ons.

    Business reason: customers who pay for security, backup, device
    protection, or priority support have already invested in the
    relationship, so this count measures stickiness that generic service
    breadth alone does not capture.
    """
    cols = ["Online_Security", "Online_Backup", "Device_Protection_Plan",
            "Premium_Tech_Support"]
    return df[cols].sum(axis=1).astype(np.int8)


def _streaming_services(df: pd.DataFrame) -> pd.Series:
    """Streaming Services (0-3): entertainment add-ons in use.

    Business reason: streaming usage is an engagement signal — households
    that stream TV, movies, or music through their plan use the connection
    daily and are far less likely to cancel it than pure-data customers.
    """
    cols = ["Streaming_TV", "Streaming_Movies", "Streaming_Music"]
    return df[cols].sum(axis=1).astype(np.int8)


def _has_phone_and_internet(df: pd.DataFrame) -> pd.Series:
    """Has Phone And Internet (0/1): takes both core services.

    Business reason: a phone+internet bundle is the classic retention anchor
    — the account covers two household needs and is harder to switch in one
    move — so it is flagged explicitly rather than left for a model to
    rediscover from the two separate flags.
    """
    return ((df["Phone_Service"] == 1) & (df["Internet_Service"] == 1)).astype(np.int8)


def _monthly_charge_per_service(df: pd.DataFrame) -> pd.Series:
    """Monthly Charge per Service: how expensive the average service is.

    Business reason: total bill size mixes "many cheap services" with "few
    expensive ones". Dividing the bill by service count separates the
    premium-heavy customer (high per-service price, harder to replicate
    cheaply elsewhere) from the price-sensitive one — a distinction churn
    models should see. Zero for the rare customer with no services.
    """
    denom = df["Total_Services"].where(df["Total_Services"] > 0)
    return np.where(denom.notna(), df["Monthly_Charge"] / denom, 0.0)


def _tenure_years(df: pd.DataFrame) -> pd.Series:
    """Tenure Years: months of loyalty expressed in years.

    Business reason: tenure is measured in months, but churn risk and value
    accrue on a year scale; expressing it in years gives regressions and
    clusters a coefficient that reads naturally ("each extra year of tenure
    lowers churn odds by X") without changing the information.
    """
    return df["Tenure_in_Months"] / 12.0


def _referrals_per_tenure_year(df: pd.DataFrame) -> pd.Series:
    """Referrals per Tenure Year: how actively the customer evangelizes.

    Business reason: raw referral counts reward long-tenured customers
    automatically. Dividing by years of tenure measures advocacy intensity —
    a customer who brought in 3 referrals in one year is worth more than one
    who brought 3 in ten — and advocacy is a leading indicator of a healthy,
    low-churn relationship.
    """
    tenure_years = df["Tenure_in_Months"] / 12.0
    return df["Number_of_Referrals"] / tenure_years


def _long_distance_dependency(df: pd.DataFrame) -> pd.Series:
    """Long Distance Dependency (0-1): LD share of the monthly bill.

    Business reason: customers whose bill is mostly long-distance minutes
    rely on a legacy product that competitors price aggressively — flagging
    their dependency tells retention teams who needs an LD-to-internet
    upsell before a cheaper offer pulls them away.
    """
    ld = df["Avg_Monthly_Long_Distance_Charges"]
    monthly = df["Monthly_Charge"]
    # Guard: the share is only meaningful when the customer is actually
    # billed for base services. A handful of accounts carry negative monthly
    # charges (bill credits), which would otherwise push the ratio outside
    # [0, 1] or divide by zero; those customers have no LD dependency either.
    meaningful = (monthly > 0) & ((ld + monthly) > 0)
    return np.where(meaningful, ld / (ld + monthly), 0.0)


def _is_new_customer(df: pd.DataFrame) -> pd.Series:
    """Is New Customer (0/1): tenure of six months or less.

    Business reason: churn concentrates in the first months of a contract,
    before habits and switching costs form. A flag for the risky onboarding
    window lets the model (and a recruiter reading a SHAP plot) see exactly
    that step-change instead of inferring it from raw tenure.
    """
    return (df["Tenure_in_Months"] <= 6).astype(np.int8)


def _senior_citizen(df: pd.DataFrame) -> pd.Series:
    """Senior Citizen (0/1): customer aged 60 or older.

    Business reason: older households have distinct needs — simpler bills,
    more support contact, higher loyalty once served well — and the raw data
    dictionary's own churn categories show service experience matters. This
    flag captures that demographic segment the way the classic telco churn
    benchmark does with its senior-citizen indicator.
    """
    return (df["Age"] >= 60).astype(np.int8)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the 10 business features in DERIVED_FEATURES order.

    Business reason: derived features translate raw usage facts into the
    business concepts the four downstream problems reason about (switching
    costs, price sensitivity, loyalty, advocacy, dependency), so every model
    and every explanation speaks the business's language.
    """
    df = df.copy()
    builders = {
        "Total_Services": _total_services,
        "Premium_Services": _premium_services,
        "Streaming_Services": _streaming_services,
        "Has_Phone_And_Internet": _has_phone_and_internet,
        "Monthly_Charge_per_Service": _monthly_charge_per_service,
        "Tenure_Years": _tenure_years,
        "Referrals_per_Tenure_Year": _referrals_per_tenure_year,
        "Long_Distance_Dependency": _long_distance_dependency,
        "Is_New_Customer": _is_new_customer,
        "Senior_Citizen": _senior_citizen,
    }
    for name in DERIVED_FEATURES:
        df[name] = builders[name](df)
    return df


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _final_column_order(df: pd.DataFrame) -> list[str]:
    """Stable column order: raw profile/spend columns in their original
    order, then the 10 derived features, then one-hot dummies, then the
    outcome fields — so the artifact reads the way the business story is
    told and later stages can rely on DERIVED_FEATURES' positions."""
    raw_kept = [c for c in REQUIRED_RAW_COLUMNS
                if c in df.columns and c not in OUTCOME_COLUMNS]
    outcome_kept = [c for c in OUTCOME_COLUMNS if c in df.columns]
    dummies = [c for c in df.columns
               if c not in set(raw_kept) | set(DERIVED_FEATURES) | set(outcome_kept)]
    return raw_kept + DERIVED_FEATURES + dummies + outcome_kept


def build_features(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Run the full feature-engineering pipeline and return the final table.

    Business reason: one entry point that guarantees every consumer of
    features.csv receives the same validated, null-free, model-ready view of
    the customer base — the single source of truth for all four problems.
    """
    df = load_raw_data(path)
    report = validate_raw_data(df)
    print("[feature_engineer] validation OK:", report)

    df = fill_documented_blanks(df)
    df = encode_binary_fields(df)
    df = add_derived_features(df)
    df = one_hot_encode_multi_category(df)

    if len(df) != EXPECTED_CUSTOMER_COUNT:
        raise ValueError(f"Row count changed during engineering: {len(df)}")
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Duplicate column names produced during engineering.")
    null_cols = df.columns[df.isna().any()].tolist()
    if null_cols:
        raise ValueError(
            f"Feature table is NOT null-free; blanks remain in {null_cols}"
        )
    return df[_final_column_order(df)]


def save_features(df: pd.DataFrame, path: Path = FEATURES_PATH) -> None:
    """Persist the feature table to data/processed/features.csv.

    Business reason: the artifact is the agreed hand-off point between stage
    1 and every later stage, so it is written with index=False and plain 0/1
    integers (no bools) to keep column types stable across CSV round-trips.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[feature_engineer] wrote {path} ({len(df)} rows x {len(df.columns)} cols)")


def main() -> None:
    """CLI entry point: python -m src.feature_engineer"""
    df = build_features()
    save_features(df)


if __name__ == "__main__":
    main()

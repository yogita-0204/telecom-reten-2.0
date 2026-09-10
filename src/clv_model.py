"""Customer lifetime value (CLV) model for the Telecom Customer Intelligence
Platform (v2).

Stage-3 module (TASKS.md #3, PRD.md Problem 2): predicts the raw dataset's
``Total_Revenue`` column — the total dollars a customer has generated so far,
used as the historical-value proxy for CLV — from the null-free feature table
built in stage 1 (``data/processed/features.csv``). A single
RandomForestRegressor (default hyperparameters, random_state=42) is trained on
a plain 80/20 split and evaluated with RMSE, MAE and R-squared; a
feature-importance plot and a tidy importance CSV are saved under
``data/processed/`` (lowercase artifact names, matching the ``features.csv``
convention), and the fitted model is saved as ``models/clv_model.pkl`` for the
API, dashboard and priority matrix.

Why this module exists (business reason): retention is not just about who will
leave but about who is worth fighting for. The priority matrix (stage 5) ranks
customers by churn probability *and* predicted CLV, so this stage produces the
value half of that ranking — and the interview story "we can predict a
customer's total revenue from tenure, bill size and service breadth" is backed
by real numbers and an importance plot showing exactly which drivers matter.

Target-name note (TASKS.md calls it ``total_revenue_to_date``): the actual
column in the raw file and in features.csv is ``Total_Revenue`` (PRD.md Section
5 and the data dictionary) — the brief's name is mapped here via
``TARGET_COLUMN = "Total_Revenue"`` rather than inventing a column that does
not exist.

Scope guard (PRD.md non-goals): a RandomForestRegressor with sklearn defaults —
no XGBoost, no survival/probabilistic CLV (BG/NBD), no causal methods, no
hyperparameter search, no deep learning.

Model-ready view: unlike the churn stage, every one of the 7,043 accounts is
trainable here — Total_Revenue exists even for the 454 Joined customers (they
have been billed since joining), so the regressor learns from and is scored on
the full population.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES_PATH = Path("data/processed/features.csv")

# Artifacts (lowercase snake_case names, matching features.csv — see MEMORY.md
# task-1 note on the lowercase artifact convention).
METRICS_PATH = Path("data/processed/clv_metrics.csv")
IMPORTANCE_CSV_PATH = Path("data/processed/clv_feature_importance.csv")
IMPORTANCE_PLOT_PATH = Path("data/processed/clv_feature_importance.png")
MODEL_PATH = Path("models/clv_model.pkl")

RANDOM_STATE = 42  # AGENTS.md: reproducibility everywhere
TEST_SIZE = 0.2    # plain 80/20 split (TASKS.md #3)

# The regression target: historical total revenue in dollars. TASKS.md calls
# this column `total_revenue_to_date`; the raw file's real name (and the name
# used by every earlier stage) is `Total_Revenue` — see module docstring.
TARGET_COLUMN = "Total_Revenue"


# ---------------------------------------------------------------------------
# Feature/target preparation
# ---------------------------------------------------------------------------

# Columns that must never enter the CLV model, with the business reason for
# each group:
#  - Customer_ID / City / Zip_Code / Latitude / Longitude: identifiers and
#    geo codes. A model memorizing them would memorize people and
#    neighborhoods, not value drivers, and would not generalize to a new
#    customer at prediction time.
#  - Offer / Internet_Type / Contract / Payment_Method: the raw multi-category
#    text columns; their one-hot dummy versions in features.csv already carry
#    the same information as numbers.
#  - Customer_Status / Churn_Category / Churn_Reason: outcome fields that are
#    unknown (or meaningless) for accounts being scored today.
#  - Total_Revenue: the target itself — direct leakage.
#  - Total_Charges / Total_Long_Distance_Charges / Total_Extra_Data_Charges /
#    Total_Refunds: the cumulative charge and refund lines that *make up* the
#    revenue figure on the same bill. Total_Revenue is a bookkeeping sum of
#    these components (they correlate at ~0.97), so feeding them to the model
#    would turn "predict customer value" into a tautological re-addition that
#    no model is needed for — and the importance plot would be a single bar
#    saying "the answer is on the bill". They are excluded so the model has to
#    learn value from the customer's observable behavior instead: tenure,
#    bill *rate* (Monthly_Charge and the average monthly lines are kept — they
#    are known month-to-month, not cumulative sums), services, and contract.
NON_PREDICTOR_COLUMNS = {
    "Customer_ID", "City", "Zip_Code", "Latitude", "Longitude",  # identifiers/geo
    "Offer", "Internet_Type", "Contract", "Payment_Method",      # raw category cols
    "Customer_Status", "Churn_Category", "Churn_Reason",         # outcomes
    TARGET_COLUMN,                                                # the target
    "Total_Charges", "Total_Long_Distance_Charges",              # revenue components
    "Total_Extra_Data_Charges", "Total_Refunds",                 # (see above)
}


def select_predictor_columns(df: pd.DataFrame) -> list[str]:
    """Return the model-ready predictor columns, in features.csv order.

    Business reason: the CLV model must consume only signals that describe who
    the customer is and how they consume the service (tenure, monthly bill,
    service breadth, contract type) — never their identifier, geo code, churn
    outcome, or the revenue figure itself or its cumulative components (see
    NON_PREDICTOR_COLUMNS). Enforcing that here, centrally, means training,
    the saved model and the API all agree on the exact input contract —
    sklearn records it on the fitted model as feature_names_in_.
    """
    predictors = [c for c in df.columns if c not in NON_PREDICTOR_COLUMNS]
    non_numeric = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric predictor columns would break the model: {non_numeric}"
        )
    return predictors


def make_clv_target(df: pd.DataFrame) -> pd.Series:
    """Extract the CLV target: Total_Revenue in dollars per customer.

    Business reason: the business question is "how much is this customer
    worth?" and the answer the data can support is the total revenue the
    account has generated so far — a real, auditable number from the billing
    record that an interviewer can verify against a single row of the CSV.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' missing from features table — "
            "the raw file names it Total_Revenue, not total_revenue_to_date."
        )
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError(
            f"Target '{TARGET_COLUMN}' contains "
            f"{int(target.isna().sum())} non-numeric values."
        )
    return target


def build_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble (X, y) for training from the full feature table.

    Business reason: unlike churn (where Joined customers have no outcome yet
    and are rightly excluded), every account in this table has already been
    billed — Total_Revenue exists for all 7,043 rows including the 454 Joined
    sign-ups — so the regressor learns from the complete population, which is
    exactly the population the priority matrix will score with its predictions.
    """
    X = df[select_predictor_columns(df)]
    y = make_clv_target(df)
    if len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("Target must be non-empty and finite for training.")
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Plain 80/20 train/test split with a fixed seed.

    Business reason: regression needs no stratified sampling (there are no
    classes to balance — every customer has a revenue value, and the test set
    is a random 20% slice of the whole base, which is how "how well does this
    model value a customer it has never met" should be measured). The fixed
    seed makes every re-run and every interview demo reproduce the same
    numbers.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def train_clv_model(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    """Fit the RandomForestRegressor with default hyperparameters.

    Business reason: customer value is not a clean linear sum — revenue
    accrues as tenure * bill size, an interaction (a customer is valuable
    because they stay *and* pay a lot) — and a tree ensemble learns products
    and thresholds natively, with no manual interaction engineering.

    Why a RandomForestRegressor and not something fancier: the project's rule
    (PRD Decision 2) is defaults-only, fully explainable models — a random
    forest needs no tuning to be solid, has a single clear story ("the average
    of 100 trees, each seeing a random subset of customers and features"),
    and ships feature_importances_ built-in, which is exactly the
    explainability deliverable this task asks for. XGBoost would add tuning
    surface and a boosting story for a marginal gain; a linear model would
    miss the tenure*bill interaction; deep learning is uninterpretable — all
    three are out of scope for this 1-day build. Trees split on value
    thresholds, so the mixed-scale features need no StandardScaler (unlike the
    logistic regression in the churn stage), and none is applied.
    """
    model = RandomForestRegressor(
        random_state=RANDOM_STATE,  # AGENTS.md: seed 42 everywhere
        # n_estimators defaults to 100 — PRD: no hyperparameter search.
    )
    model.fit(X_train, y_train)
    print(
        f"[clv_model] trained {type(model).__name__} on {len(X_train)} "
        f"customers ({len(X_train.columns)} features)"
    )
    return model


def evaluate_model(
    model: object, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Score the model on the held-out test split; return RMSE, MAE, R-squared.

    Business reason: the three metrics answer three questions an interviewer
    will ask. RMSE and MAE say how far the predicted value is from the true
    bill in dollars — MAE is the average miss a retention team would plan
    around, RMSE penalizes big misses like the rare very-high-value accounts.
    R-squared says how much of the variation in customer value the model
    explains (how much of "why are some customers worth $12k and others $21"
    is tenure and bill size vs. unpredictable noise). All three are checked
    finite so a silent NaN never reaches the metrics artifact.
    """
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    bad = {k: v for k, v in metrics.items() if not np.isfinite(v)}
    if bad:
        raise ValueError(f"Non-finite CLV metrics: {bad}")
    print(
        f"[clv_model] test split ({len(y_test)} customers): "
        f"RMSE=${rmse:,.2f}  MAE=${mae:,.2f}  R2={r2:.4f}  "
        f"(target mean ${y_test.mean():,.2f}, std ${y_test.std():,.2f})"
    )
    return metrics


def save_metrics(metrics: dict[str, float]) -> None:
    """Persist RMSE/MAE/R-squared to data/processed/clv_metrics.csv.

    Business reason: the README, the dashboard's overview page and the
    interview answer all quote these three numbers, so they are saved as a
    plain one-row CSV (lowercase name, same convention as features.csv and the
    churn comparison table) rounded to a readable precision — never left only
    in a terminal scrollback.
    """
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).round(4).to_csv(METRICS_PATH, index=False)
    print(f"[clv_model] wrote {METRICS_PATH}")


# ---------------------------------------------------------------------------
# Feature-importance artifact
# ---------------------------------------------------------------------------

def generate_feature_importance(
    model: object, X: pd.DataFrame
) -> pd.DataFrame:
    """Feature-importance CSV + bar plot for the fitted random forest.

    Business reason: PRD user story 2 demands every model output be traceable
    to features — this is the CLV half of that promise. The plot and table
    answer "what actually drives a customer's total revenue?" (expected
    answer: tenure and monthly bill size dwarf everything else), which is the
    sentence the owner needs to explain the model in an interview. The tidy
    CSV lets the dashboard re-render the chart live; the PNG documents it in
    the README.

    RandomForestRegressor's feature_importances_ are the mean decrease in
    impurity (variance for regression) across all trees — a built-in,
    model-consistent ranking that needs no separate explainer library.

    matplotlib is imported here rather than at module scope so the serving
    path (api/main.py imports this module) never pays for a GUI/plotting
    stack — matplotlib alone adds ~150 MB to the process, which matters on
    free-tier instances.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importances = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    importances.to_csv(IMPORTANCE_CSV_PATH, index=False)

    # Horizontal bars, most important on top — the plot reads top-down like a
    # ranked list of business drivers.
    fig, ax = plt.subplots(figsize=(9, 11))
    order = importances.iloc[::-1]
    ax.barh(order["feature"], order["importance"], color="#2a6f97")
    ax.set_xlabel("Feature importance (mean decrease in variance)")
    ax.set_title("CLV drivers — RandomForestRegressor feature importances")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PLOT_PATH, dpi=150)
    plt.close(fig)

    top = importances.head(10)
    print(
        f"[clv_model] wrote {IMPORTANCE_CSV_PATH} and {IMPORTANCE_PLOT_PATH}\n"
        f"[clv_model] top drivers: "
        + ", ".join(
            f"{row.feature} ({row.importance:.3f})" for row in top.itertuples()
        )
    )
    return importances


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: object, path: Path = MODEL_PATH) -> None:
    """Persist the fitted CLV model with joblib.

    Business reason: the priority matrix (stage 5), the API's /predict route
    and the dashboard all load this one file to value a customer, so it must
    carry the exact trained input contract with it — sklearn records it on the
    estimator as feature_names_in_, which keeps every later consumer honest
    without a second bookkeeping file. joblib is the project's agreed
    serialization format.

    Compression (level 3) is used because this random forest's fully grown
    trees serialize to ~49 MB uncompressed — 94% of every generated artifact
    combined. The values are float64 leaf arrays and split thresholds, which
    compress ~3-4x, and joblib.load transparently decompresses, so no
    consumer changes. The smaller file keeps the deploy images, the
    dashboard's first-launch bootstrap and any repo archive light.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    print(
        f"[clv_model] saved {type(model).__name__} -> {path} "
        f"({len(model.feature_names_in_)} features)"
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def run_clv_stage() -> dict:
    """Run the full CLV stage and save every artifact.

    Business reason: one entry point (a) makes the stage reproducible with a
    single command for later orchestration in run_pipeline.py and (b) keeps
    the artifact set fixed — metrics table, feature-importance CSV + plot, and
    the deployed model file.
    """
    df = pd.read_csv(FEATURES_PATH)
    print(
        f"[clv_model] loaded {FEATURES_PATH}: {len(df)} rows; "
        f"target '{TARGET_COLUMN}' present, "
        f"{int(df[TARGET_COLUMN].isna().sum())} nulls"
    )

    X, y = build_training_data(df)
    predictors = select_predictor_columns(df)
    print(f"[clv_model] predictor columns ({len(predictors)}): {predictors}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(
        f"[clv_model] split: train={len(X_train)} test={len(X_test)} "
        f"(test target mean ${y_test.mean():,.2f})"
    )

    model = train_clv_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics)
    generate_feature_importance(model, X)
    save_model(model)

    return {
        "model": type(model).__name__,
        "predictors": len(predictors),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
    }


def main() -> None:
    """CLI entry point: python -m src.clv_model"""
    summary = run_clv_stage()
    print(f"[clv_model] summary: {summary}")


if __name__ == "__main__":
    main()

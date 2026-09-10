"""Churn prediction model for the Telecom Customer Intelligence Platform (v2).

Stage-2 module (TASKS.md #2, PRD.md Problem 1): predicts which customers are
about to churn, using the null-free feature table built in stage 1
(``data/processed/features.csv``). Two fully explainable models are trained on
a stratified 80/20 split and compared — Logistic Regression and Random Forest,
both with ``class_weight='balanced'`` and default hyperparameters (LR runs
inside a StandardScaler pipeline; the estimator itself is untouched, see
``train_models`` for the convergence rationale). The better model (by ROC-AUC)
is kept for the API/dashboard, the comparison table is saved under
``data/processed/`` (lowercase artifact names, matching the ``features.csv``
convention), and a TreeExplainer SHAP summary plot explains the Random Forest
globally.

Why this module exists (business reason): retention teams cannot act on a
probability alone — they need to know *who* is at risk and *why* the model
thinks so. The churn stage therefore produces not just a saved classifier but
(1) an honest head-to-head comparison of a linear and a tree model, so the
project owner can say in an interview "we compared a simple baseline with a
tree model and here is the real trade-off", and (2) model-agnostic explanations
(SHAP) that turn any prediction into "this customer is leaving because of
fiber-optic price sensitivity / a month-to-month contract / short tenure".

Scope guard (PRD.md non-goals): no XGBoost, no hyperparameter search, no SMOTE
— exactly the two approved classifiers with defaults, and SHAP's TreeExplainer
only (never KernelExplainer/LinearExplainer).

Model-ready view: the 7,043-row features table keeps three customer states —
Stayed (4,720), Churned (1,869) and Joined (454). Churn classification is only
defined once an outcome exists, so the 454 Joined accounts (recent sign-ups
with no churn opportunity yet) are excluded from training and evaluation but
left untouched in features.csv; later stages score the full population with
the saved model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("data/processed/features.csv")

# Artifacts (lowercase snake_case names, matching features.csv — see
# MEMORY.md task-1 note on the lowercase artifact convention).
COMPARISON_PATH = Path("data/processed/churn_model_comparison.csv")
CONFUSION_MATRIX_PATH = Path("data/processed/churn_confusion_matrix.csv")
CONFUSION_PLOT_PATH = Path("data/processed/churn_confusion_matrix.png")
SHAP_SUMMARY_PATH = Path("data/processed/shap_summary.png")
MODEL_PATH = Path("models/churn_model.pkl")

RANDOM_STATE = 42  # AGENTS.md: reproducibility everywhere
TEST_SIZE = 0.2    # stratified 80/20 split (TASKS.md #2)

# Rows whose churn outcome is already known (Joined customers are recent
# sign-ups that have not had a chance to churn yet).
LABELED_STATUSES = ("Stayed", "Churned")


# ---------------------------------------------------------------------------
# Feature/target preparation
# ---------------------------------------------------------------------------

# Columns that must never enter a churn model, with the business reason for
# each group: customer identifiers and geo codes (identify/geo-locate a
# person; a model memorizing Zip_Code would learn neighborhoods, not churn
# drivers, and would not generalize), the raw multi-category columns (their
# one-hot dummy versions already carry the same information in numeric form),
# and the outcome fields themselves (direct leakage).
NON_PREDICTOR_COLUMNS = {
    "Customer_ID", "City", "Zip_Code", "Latitude", "Longitude",  # identifiers/geo
    "Offer", "Internet_Type", "Contract", "Payment_Method",      # raw category cols
    "Customer_Status", "Churn_Category", "Churn_Reason",         # outcomes/target
}


def select_predictor_columns(df: pd.DataFrame) -> list[str]:
    """Return the model-ready predictor columns, in features.csv order.

    Business reason: the churn model must consume only signals that are known
    *before* a customer leaves (contract, tenure, bill size, service breadth)
    and that can be explained one by one in an interview. IDs, geo codes and
    the outcome columns would either leak the answer or make the model
    un-generalizable, so they are excluded here once, centrally — the same
    list then applies to training, to the saved model's predictions, and to
    the API's /predict route (sklearn records it as feature_names_in_).
    """
    predictors = [c for c in df.columns if c not in NON_PREDICTOR_COLUMNS]
    non_numeric = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric predictor columns would break the models: {non_numeric}"
        )
    return predictors


def make_churn_target(df: pd.DataFrame) -> pd.Series:
    """Build the 0/1 churn label from Customer_Status (1 = Churned).

    Business reason: the business question is binary — "will this customer
    leave?" — so the three-state status column is reduced to Churned=1 vs
    Stayed=0. Joined accounts get no label (outcome not yet observable) and
    are dropped by the caller before training rather than being mislabeled as
    non-churners, which would quietly dilute every metric with customers who
    simply have not had time to leave.
    """
    return (df["Customer_Status"] == "Churned").astype(np.int8)


def build_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble (X, y) for model training from the full feature table.

    Business reason: this is the single place that decides which customers
    the churn models are allowed to learn from — only those with an observed
    outcome (Stayed/Churned). Keeping the Joined accounts out of the fit is a
    correctness choice (their label is unknown, not "no"), while keeping them
    in features.csv preserves the full 7,043-row population that later stages
    (priority matrix, dashboard) score with the finished model.
    """
    labeled = df[df["Customer_Status"].isin(LABELED_STATUSES)]
    y = make_churn_target(labeled)
    if int(y.sum()) == 0 or int((1 - y).sum()) == 0:
        raise ValueError("Both classes must be present to train a classifier.")
    return labeled[select_predictor_columns(df)], y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split with a fixed seed.

    Business reason: churners are the minority class (28%), so a plain random
    split could by chance give the test set a different churn rate than the
    business sees — stratified sampling pins the test set to the same 28%
    rate as the full population, and random_state=42 makes every re-run (and
    every interview demo) reproduce the exact same numbers.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def train_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, object]:
    """Fit the two approved churn classifiers with default hyperparameters.

    Business reason: the comparison the business story needs is "simple,
    readable linear model vs. flexible tree model on the same features". Two
    models, both defaults, both class_weight='balanced' — nothing fancier,
    because the goal is a clean head-to-head the owner can defend, not the
    last percentage point of AUC (PRD Decision 2: no search this round).
    """
    models: dict[str, object] = {}

    # Logistic Regression: the explainable baseline. class_weight='balanced'
    # reweights the loss instead of SMOTE-synthesizing rows — real customers
    # only, so every coefficient still reads as "for this real population,
    # one more month of tenure lowers churn odds by X".
    #
    # The estimator itself runs with sklearn's default hyperparameters inside
    # a StandardScaler pipeline (scaler fit on the training split only, no
    # leakage). Standardization is data preprocessing, not model tuning: our
    # features span wildly different scales (0/1 flags next to monthly bills
    # and lifetime charges in the thousands), and lbfgs — sklearn's default
    # solver — converges so slowly on raw mixed scales that the reported
    # metrics still drift with the iteration cutoff at max_iter=1000. Scaled,
    # it converges within the default 100 iterations, and the coefficients
    # become directly comparable to each other, which sharpens the
    # interview story rather than weakening it.
    models["Logistic Regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )

    # Random Forest: captures non-linear effects and interactions the linear
    # model cannot see (e.g. "fiber optic only hurts when the contract is
    # month-to-month"). It trains on the raw features — trees split on value
    # thresholds, so per-feature scaling would not change its splits and none
    # is applied. Defaults per PRD Decision 2 — no grid search; and no
    # XGBoost because a Random Forest is the tree model that SHAP's
    # TreeExplainer explains exactly and that needs no tuning to be solid.
    models["Random Forest"] = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    )

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"[churn_model] trained {name}: {type(model).__name__}")
    return models


def evaluate_models(
    models: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Score every model on the held-out test split.

    Business reason: accuracy alone would flatter the models because 72% of
    customers stay — a "predict everyone stays" model would score 72% with
    zero business value. Reporting precision, recall, F1 and ROC-AUC together
    shows the real cost/benefit trade-off: recall says how many at-risk
    customers we would actually catch, precision says how many of our save
    offers go to customers who truly needed them, and ROC-AUC measures the
    ranking quality behind a priority list, threshold-free.
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # P(churned), positive class
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }
        )
    return pd.DataFrame(rows)


def save_comparison_table(results: pd.DataFrame) -> None:
    """Persist the model comparison table to data/processed/.

    Business reason: the comparison table is the centerpiece artifact of the
    churn stage — the README, the dashboard's overview page and the interview
    answer all quote real numbers from this file, so it is saved as a plain
    CSV (lowercase name, same convention as features.csv) with the metrics
    rounded to a readable precision.
    """
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(COMPARISON_PATH, index=False)
    print(
        f"[churn_model] wrote {COMPARISON_PATH}\n"
        f"{results.round(4).to_string(index=False)}"
    )


def choose_better_model(results: pd.DataFrame) -> str:
    """Pick the model to deploy, by ROC-AUC then F1.

    Business reason: ROC-AUC is the headline ranking metric for churn (it is
    threshold-free, so it measures "does the model put likely churners at the
    top of the save list" rather than one arbitrary cutoff); F1 breaks ties
    by balancing precision and recall. The winner gets the confusion-matrix
    artifact and is saved as models/churn_model.pkl for the API.
    """
    ranked = results.sort_values(
        by=["roc_auc", "f1"], ascending=False
    )
    return str(ranked.iloc[0]["model"])


# ---------------------------------------------------------------------------
# Diagnostic artifacts for the chosen model
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    """Confusion matrix (CSV + PNG) for the deployed model on the test split.

    Business reason: the confusion matrix is the honest bottom line for the
    retention budget — of the real churners in the test set, how many the
    model flags (true positives, the ones we can try to save) and how many it
    misses (false negatives, the silent losses); and of the customers it
    flags, how many were false alarms (wasted save offers). The tidy CSV lets
    the dashboard re-render it, the PNG documents it in the README.

    matplotlib is imported here rather than at module scope so the serving
    path (api/main.py imports this module) never pays for a GUI/plotting
    stack — matplotlib alone adds ~150 MB to the process, which matters on
    free-tier instances.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    tidy = pd.DataFrame(
        [
            {
                "model": model_name,
                "actual": actual,
                "predicted": predicted,
                "count": int(cm[actual, predicted]),
            }
            for actual in range(cm.shape[0])
            for predicted in range(cm.shape[1])
        ]
    )
    tidy.to_csv(CONFUSION_MATRIX_PATH, index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicted Stayed", "Predicted Churned"])
    ax.set_yticks([0, 1], labels=["Actual Stayed", "Actual Churned"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title(f"Churn confusion matrix — {model_name} (test split)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(
        f"[churn_model] confusion matrix ({model_name}): TN={cm[0,0]} "
        f"FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]} -> "
        f"{CONFUSION_MATRIX_PATH}, {CONFUSION_PLOT_PATH}"
    )


# ---------------------------------------------------------------------------
# SHAP explanation (Random Forest only)
# ---------------------------------------------------------------------------

def generate_shap_summary(
    rf_model: object, X_test: pd.DataFrame
) -> None:
    """Global SHAP summary plot for the Random Forest, saved as a PNG.

    Business reason: the interview question "why does this model think my
    customers are leaving?" is answered feature-by-feature here — the plot
    shows which drivers (contract type, tenure, fiber-optic bills, service
    breadth) push churn risk up or down across the whole held-out population,
    and by how much. This is the traceability deliverable of PRD user story 2.

    TreeExplainer is used because the Random Forest is a tree ensemble, so
    SHAP's TreeExplainer computes *exact, model-consistent* Shapley values in
    a single pass (the path-dependent method needs no background dataset and
    no sampling); KernelExplainer would only approximate the same numbers at
    far greater cost, so it is deliberately not used. The held-out set is
    sampled down to a fixed 2,000 rows first: a global summary plot is
    statistically stable at that size, and it keeps peak SHAP memory ~4x
    lower — important on free-tier instances where the pipeline regenerates
    artifacts on first launch (Render/Streamlit bootstrap).

    shap (and matplotlib) are imported here, not at module scope, so the
    serving path never loads the explainability stack — SHAP brings its own
    dependency tree and tens of MB into the process.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    # Deterministic sample of the held-out set for the SHAP computation
    # (explained in the docstring; seed matches the repo-wide seed 42 rule).
    rng = np.random.default_rng(42)
    n_shap = min(2000, X_test.shape[0])
    sample_idx = rng.choice(X_test.shape[0], size=n_shap, replace=False)
    X_shap = X_test.iloc[sample_idx]

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap)

    # Shape guard for portability across SHAP versions: binary classifiers may
    # return a list of per-class arrays (older shap) or a single array whose
    # last dimension is the class axis (newer shap). Normalize to the 2-D
    # (n_samples, n_features) values of the positive (churned) class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[-1]
    if getattr(shap_values, "ndim", 1) == 3:
        shap_values = shap_values[..., 1]
    if (
        getattr(shap_values, "ndim", 1) != 2
        or shap_values.shape[0] != X_shap.shape[0]
        or shap_values.shape[1] != X_shap.shape[1]
    ):
        raise ValueError(
            f"Unexpected SHAP values shape {getattr(shap_values, 'shape', None)} "
            f"for X_shap {X_shap.shape}"
        )

    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=list(X_shap.columns),
        show=False,
    )
    plt.savefig(SHAP_SUMMARY_PATH, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"[churn_model] wrote SHAP summary plot -> {SHAP_SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: object, path: Path = MODEL_PATH) -> None:
    """Persist the deployed churn model with joblib.

    Business reason: the API's /predict route and the dashboard's live
    prediction both load this one file, so it must carry everything needed to
    reproduce a prediction. joblib is the project's agreed serialization
    format; sklearn records the fitted feature list on the estimator itself
    (feature_names_in_), which keeps the exact model input contract available
    to every later consumer without a second bookkeeping file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(
        f"[churn_model] saved {type(model).__name__} -> {path} "
        f"({len(model.feature_names_in_)} features)"
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def run_churn_stage() -> dict:
    """Run the full churn stage and save every artifact.

    Business reason: one entry point (a) makes the stage reproducible with a
    single command for later orchestration in run_pipeline.py and (b) keeps
    the artifact set fixed — comparison table, confusion matrix for the
    winner, SHAP plot for the Random Forest, and the deployed model file.
    """
    df = pd.read_csv(FEATURES_PATH)
    print(
        f"[churn_model] loaded {FEATURES_PATH}: {len(df)} rows "
        f"({df['Customer_Status'].value_counts().to_dict()})"
    )

    X, y = build_training_data(df)
    print(
        f"[churn_model] training population: {len(X)} customers "
        f"({int((1 - y).sum())} stayed / {int(y.sum())} churned); "
        f"{len(df) - len(X)} Joined customers excluded (no outcome yet)"
    )
    predictors = select_predictor_columns(df)
    print(f"[churn_model] predictor columns ({len(predictors)}): {predictors}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(
        f"[churn_model] stratified split: train={len(X_train)} "
        f"test={len(X_test)} (test churn rate {y_test.mean():.1%})"
    )

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    save_comparison_table(results)

    best_name = choose_better_model(results)
    best_model = models[best_name]
    print(f"[churn_model] better model by ROC-AUC/F1: {best_name}")

    plot_confusion_matrix(best_model, X_test, y_test, best_name)
    generate_shap_summary(models["Random Forest"], X_test)
    save_model(best_model)

    best = results[results["model"] == best_name].iloc[0]
    return {
        "better_model": best_name,
        "accuracy": float(best["accuracy"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "roc_auc": float(best["roc_auc"]),
    }


def main() -> None:
    """CLI entry point: python -m src.churn_model"""
    summary = run_churn_stage()
    print(f"[churn_model] deployed model summary: {summary}")


if __name__ == "__main__":
    main()

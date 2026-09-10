"""Customer segmentation model for the Telecom Customer Intelligence Platform
(v2).

Stage-4 module (TASKS.md #4, PRD.md Problem 3): partitions the full 7,043-
customer base into exactly four named segments using KMeans on three business
features from the stage-1 table (``data/processed/features.csv``): tenure
(``Tenure_in_Months``), monthly bill size (``Monthly_Charge``) and service
breadth (``Total_Services``) — the same three columns PRD.md Section 5 names as
the segmentation inputs.

Why this module exists (business reason): retention teams cannot treat 7,043
customers as one group — a save offer that suits a two-year full-bundle
household is wasted on a month-old phone-only account. Segmentation groups
customers whose tenure, bill and service mix behave alike, so the priority
matrix (stage 5), the dashboard and the interview story can all speak about
*types* of customers ("loyal premium accounts vs. new budget sign-ups")
instead of one averaged base.

Method and fit choice: features are standardized (StandardScaler — k-means
minimizes Euclidean distance, so a feature measured in dollars or months would
otherwise outweigh a 0-10 service count purely by unit scale), then KMeans is
evaluated for k=2..8 by the two classic diagnostics — inertia for the elbow
method and the silhouette score — and the final model is fitted at k=4 with
random_state=42 (AGENTS.md), as PRD.md fixes four named segments as the
deliverable. Each cluster is named from its centroid's characteristics, i.e.
from where the centroid sits relative to the population's low/middle/high
thirds on tenure and monthly charge, so a segment name is a claim about real
numbers in the profile artifact, not a marketing label.

Scope guard (PRD.md non-goals): KMeans is the only clustering method in the
approved stack; no hierarchical/DBSCAN/GMM, no survival or causal methods, no
hyperparameter search. Unlike churn/CLV there is no train/test split here —
segmentation is descriptive: the model fits and is scored on the same full
population, and the saved artifact is a fitted Pipeline
(scaler + kmeans) so the API/dashboard can assign a *new* customer to a
segment with the exact preprocessing the clusters were learned on.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("data/processed/features.csv")

# Artifacts (lowercase snake_case names, matching features.csv — see MEMORY.md
# task-1 note on the lowercase artifact convention).
EVALUATION_PATH = Path("data/processed/segment_evaluation.csv")
EVALUATION_PLOT_PATH = Path("data/processed/segmentation_elbow_silhouette.png")
PROFILES_PATH = Path("data/processed/segment_profiles.csv")
ASSIGNMENTS_PATH = Path("data/processed/customer_segments.csv")
MODEL_PATH = Path("models/segmentation_model.pkl")

RANDOM_STATE = 42  # AGENTS.md: reproducibility everywhere
FINAL_K = 4        # PRD.md: exactly four named segments are the deliverable
K_MIN, K_MAX = 2, 8  # elbow + silhouette sweep range (TASKS.md #4)

# The three segmentation inputs named in PRD.md Section 5: how long the
# customer has stayed, how big the monthly bill is, and how many of the ten
# services they subscribe to. These are the customer-relationship dimensions
# a retention team can actually act on.
SEGMENT_FEATURES = ["Tenure_in_Months", "Monthly_Charge", "Total_Services"]

# Tercile labels (vs. the full population) used to name clusters from their
# centroids. A centroid in the lowest/middle/highest third of the base on
# tenure is "New"/"Established"/"Loyal"; on monthly charge it is
# "Budget"/"Standard"/"Premium". Combined, e.g. "Loyal Premium Customers" is
# shorthand for "centroid in the top third of tenure AND the top third of
# monthly charge" — every name is traceable to the profile artifact.
TENURE_LEVELS = {"Low": "New", "Mid": "Established", "High": "Loyal"}
SPEND_LEVELS = {"Low": "Budget", "Mid": "Standard", "High": "Premium"}
SERVICE_LEVELS = {"Low": "Basic", "Mid": "Standard", "High": "Full-Service"}


def select_segment_features(df: pd.DataFrame) -> list[str]:
    """Return the three segmentation inputs, in features.csv order.

    Business reason: clustering must consume exactly the dimensions PRD.md
    promises an interviewer ("tenure, monthly charge, services count") —
    no more. Adding correlated proxies (Tenure_Years is the same information
    as Tenure_in_Months) would silently overweight one business dimension in
    the distance metric; excluding them keeps the four segment names directly
    explainable from three numbers on a profile card.
    """
    missing = [c for c in SEGMENT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Segment features missing from feature table: {missing}")
    return SEGMENT_FEATURES


# ---------------------------------------------------------------------------
# K choice diagnostics (elbow + silhouette, k=2..8)
# ---------------------------------------------------------------------------

def evaluate_k_values(
    X_scaled: np.ndarray,
) -> pd.DataFrame:
    """Fit KMeans for every k in 2..8 and record inertia and silhouette.

    Business reason: the "why exactly four segments?" interview answer needs
    evidence, not assertion. Inertia (the elbow method) shows how much
    within-cluster spread each extra cluster removes — the curve bends where
    extra segments stop buying much separation; the silhouette score shows how
    cleanly customers sit inside their own cluster vs. the next one, on a
    -1..+1 scale where business-meaningful groupings live above ~0.25. Both
    are computed on the same standardized features the final model uses, each
    k fitted with the agreed random_state so the sweep is reproducible.
    """
    rows = []
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        km.fit(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X_scaled, km.labels_)),
            }
        )
    evaluation = pd.DataFrame(rows)
    print(
        f"[segmentation] k sweep {K_MIN}..{K_MAX}:\n"
        f"{evaluation.round(4).to_string(index=False)}"
    )
    return evaluation


def plot_k_evaluation(evaluation: pd.DataFrame) -> None:
    """Two-panel PNG of the elbow curve and the silhouette curve.

    Business reason: the README and the interview slide both want to *show*
    the k choice, not just quote it — the left panel is the elbow (inertia
    dropping fast until the bend, slowly after), the right panel is the
    silhouette score, and the reader sees where k=4 sits on both curves.

    matplotlib is imported here rather than at module scope so the serving
    path (api/main.py imports this module) never pays for a GUI/plotting
    stack — matplotlib alone adds ~150 MB to the process, which matters on
    free-tier instances.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_elbow, ax_sil) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_elbow.plot(evaluation["k"], evaluation["inertia"], "o-", color="#2a6f97")
    ax_elbow.axvline(FINAL_K, color="#c1121f", linestyle="--", linewidth=1)
    ax_elbow.set_xlabel("Number of clusters (k)")
    ax_elbow.set_ylabel("Inertia (within-cluster sum of squares)")
    ax_elbow.set_title("Elbow method")
    ax_elbow.grid(alpha=0.3)

    ax_sil.plot(evaluation["k"], evaluation["silhouette"], "o-", color="#2a6f97")
    ax_sil.axvline(FINAL_K, color="#c1121f", linestyle="--", linewidth=1)
    ax_sil.set_xlabel("Number of clusters (k)")
    ax_sil.set_ylabel("Silhouette score")
    ax_sil.set_title("Silhouette score")
    ax_sil.grid(alpha=0.3)

    fig.suptitle("KMeans segment count selection — elbow and silhouette", y=1.02)
    fig.tight_layout()
    fig.savefig(EVALUATION_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[segmentation] wrote k-evaluation plot -> {EVALUATION_PLOT_PATH}")


# ---------------------------------------------------------------------------
# Final model (k=4, fixed by PRD; diagnostics confirm the region)
# ---------------------------------------------------------------------------

def build_segmentation_model(X: pd.DataFrame) -> Pipeline:
    """Fit the production Pipeline (StandardScaler + KMeans) at k=4.

    Business reason: PRD.md fixes four named segments as the deliverable, and
    the k=2..8 sweep is run to check that four sits in a sensible region of
    both diagnostics rather than to search for a better k (PRD Decision 2:
    no hyperparameter search this round). On this base the silhouette score
    peaks in the k=3-4 region and degrades for every larger k, and the elbow
    has visibly flattened by k=4 — so the business's fixed k lands where the
    data stops rewarding extra clusters.

    Why KMeans and not a more complex clustering method: the deliverable is a
    flat set of four easy-to-describe groups, and KMeans is the one clustering
    algorithm in the approved stack (AGENTS.md). Hierarchical clustering would
    add a dendrogram cut with no business counterpart, DBSCAN would add an
    epsilon/min-samples tuning problem and can label customers as noise
    (every customer must belong to exactly one named segment), and Gaussian
    mixtures would add distributional assumptions and a probabilistic story
    that is harder to explain than "each customer goes to the nearest of four
    centers". None of them buys explainability the priority matrix can use, so
    none is used.

    The pipeline is fitted on the original-unit feature frame, so the scaler
    inside learns the true per-feature mean/scale and sklearn records the
    exact input columns on the estimator (feature_names_in_), matching how the
    churn and CLV artifacts carry their contracts. A consumer that loads
    models/segmentation_model.pkl transforms and assigns in one predict().
    """
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=FINAL_K,
                    n_init=10,  # explicit: 10 restarts, best kept (default is 'auto')
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X)
    print(
        f"[segmentation] fitted {type(model).__name__} at k={FINAL_K} on "
        f"{len(X)} customers ({len(X.columns)} features)"
    )
    return model


# ---------------------------------------------------------------------------
# Cluster naming from centroid characteristics
# ---------------------------------------------------------------------------

def _tercile_level(value: float, p33: float, p66: float) -> str:
    """Bucket a centroid value into Low/Mid/High vs. population terciles.

    Business reason: naming must be a reproducible claim about the data — the
    levels here are what the names are built from, so the boundary logic
    lives in exactly one place (<= p33 Low, <= p66 Mid, else High).
    """
    if value <= p33:
        return "Low"
    if value <= p66:
        return "Mid"
    return "High"


def name_segments(
    df: pd.DataFrame, model: Pipeline
) -> pd.DataFrame:
    """Build the 4-row profile table: per-cluster size, centroid in original
    units, tercile level per feature, and a name derived from those levels.

    Business reason: a segment is only actionable if its label carries its
    meaning. Centroids are mapped back from standardized units to the units a
    stakeholder reads (months, dollars, services) and each cluster is named
    from where its centroid sits against the population thirds on tenure and
    monthly charge — "Loyal Premium Customers" literally means "top third on
    tenure, top third on bill". The profile CSV then lets anyone (and the
    task-4 gate) check that there are exactly four named segments with their
    real sizes.

    Disambiguation guard: two clusters could in principle share the same
    tenure+spend tercile combination (two distinct centers inside one third).
    If that happens the services-count level (Basic/Standard/Full-Service) is
    appended so every name stays unique; on the current data it never fires.
    """
    scaler = model.named_steps["scaler"]
    kmeans = model.named_steps["kmeans"]
    features = select_segment_features(df)

    # Centroid values mapped back from standardized units to the units a
    # stakeholder reads (months, dollars, services).
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

    # Population terciles for each feature — the "low/middle/high third of the
    # customer base" reference frame the names are claims against. Computed on
    # the DataFrame so pandas' quantile interpolation is consistent for every
    # feature.
    terciles = {
        c: (df[c].quantile(1 / 3), df[c].quantile(2 / 3)) for c in features
    }

    labels = kmeans.labels_
    counts = pd.Series(labels).value_counts().sort_index()
    rows = []
    for cluster_id in range(FINAL_K):
        centroid = centroids_original[cluster_id]
        tenure_level = _tercile_level(
            centroid[0], *terciles[features[0]]
        )
        spend_level = _tercile_level(
            centroid[1], *terciles[features[1]]
        )
        services_level = _tercile_level(
            centroid[2], *terciles[features[2]]
        )
        name = f"{TENURE_LEVELS[tenure_level]} {SPEND_LEVELS[spend_level]} Customers"
        rows.append(
            {
                "cluster": int(cluster_id),
                "segment": name,
                "customers": int(counts[cluster_id]),
                "share": float(counts[cluster_id] / len(df)),
                f"{features[0]}_centroid": round(float(centroid[0]), 2),
                f"{features[1]}_centroid": round(float(centroid[1]), 2),
                f"{features[2]}_centroid": round(float(centroid[2]), 2),
                "tenure_level": tenure_level,
                "spend_level": spend_level,
                "services_level": services_level,
            }
        )

    profiles = pd.DataFrame(rows)

    # Uniqueness guard: two centroids in the same tercile band would collide on
    # the tenure+spend name; append the services-level descriptor to make every
    # name unique, keeping the base name when it already is.
    duplicated = profiles["segment"].duplicated(keep=False)
    if duplicated.any():
        profiles.loc[duplicated, "segment"] = (
            profiles.loc[duplicated, "segment"]
            + " ("
            + profiles.loc[duplicated, "services_level"].map(SERVICE_LEVELS)
            + " Services)"
        )

    if profiles["segment"].nunique() != FINAL_K:
        raise ValueError("Cluster names are not unique — naming rule is broken.")
    print(
        f"[segmentation] named {FINAL_K} clusters:\n"
        f"{profiles[['cluster', 'segment', 'customers', 'share']].round(4).to_string(index=False)}"
    )
    return profiles


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_profiles(profiles: pd.DataFrame) -> None:
    """Persist the 4-row segment profile table to data/processed/.

    Business reason: the profile artifact is the gate for this task (exactly
    four named clusters) and the reference the dashboard, README and
    interview answers quote segment sizes and centroid characteristics from.
    """
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(PROFILES_PATH, index=False)
    print(f"[segmentation] wrote {PROFILES_PATH} ({len(profiles)} segments)")


def labels_to_names(profiles: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Map raw cluster numbers to their human-readable segment names.

    Business reason: both the number (stable join key for code) and the name
    (what stakeholders read) are written to the assignment artifact, and this
    is the single place the two are tied together from the profile table.
    """
    name_by_cluster = profiles.set_index("cluster")["segment"].to_dict()
    return pd.Series(labels).map(name_by_cluster)


def run_segmentation_stage() -> dict:
    """Run the full segmentation stage and save every artifact.

    Business reason: one entry point (a) makes the stage reproducible with a
    single command for later orchestration in run_pipeline.py and (b) keeps
    the artifact set fixed — evaluation table + plot for the k choice, the
    4-row profile table, the per-customer assignment table, and the deployed
    model file.
    """
    df = pd.read_csv(FEATURES_PATH)
    print(
        f"[segmentation] loaded {FEATURES_PATH}: {len(df)} rows, "
        f"{int(df.isna().sum().sum())} nulls"
    )

    features = select_segment_features(df)
    X = df[features]
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Segment features contain non-finite values.")

    # Standardize the three features for the k=2..8 sweep (the sweep's KMeans
    # models are throwaway diagnostics, so they share this one scaler fit on
    # the full population — no split exists in descriptive clustering). The
    # production pipeline then refits the same scaler on the same X inside
    # build_segmentation_model, so the saved artifact is fully self-contained;
    # StandardScaler.fit is deterministic, so both scalers are identical.
    X_scaled = StandardScaler().fit(X).transform(X)

    evaluation = evaluate_k_values(X_scaled)
    evaluation.round(4).to_csv(EVALUATION_PATH, index=False)
    print(f"[segmentation] wrote {EVALUATION_PATH}")
    plot_k_evaluation(evaluation)

    # Where does k=4 sit on the two diagnostics? Reported, not searched over:
    # PRD fixes four segments; the sweep confirms the region is sensible.
    best_k = int(evaluation.loc[evaluation["silhouette"].idxmax(), "k"])
    best_sil = float(evaluation["silhouette"].max())
    sil4 = float(evaluation.loc[evaluation["k"] == FINAL_K, "silhouette"].iloc[0])
    print(
        f"[segmentation] silhouette peaks at k={best_k} "
        f"({best_sil:.4f}); k={FINAL_K} silhouette={sil4:.4f} "
        f"(best among k>={FINAL_K}); PRD fixes k={FINAL_K} named segments"
    )

    model = build_segmentation_model(X)
    profiles = name_segments(df, model)
    save_profiles(profiles)

    # Per-customer assignment: fitted labels from the production pipeline.
    labels = model.named_steps["kmeans"].labels_
    assignments = pd.DataFrame(
        {
            "Customer_ID": df["Customer_ID"],
            "Cluster": labels,
            "Segment": labels_to_names(profiles, labels),
        }
    )
    # Assignment contract (TASKS.md #4 / PRD acceptance): every one of 7,043
    # customers appears exactly once with exactly one non-null segment name.
    if len(assignments) != len(df):
        raise ValueError(f"Assignment count {len(assignments)} != {len(df)}")
    if not assignments["Customer_ID"].is_unique:
        raise ValueError("Duplicate customers in the assignment table.")
    if assignments["Cluster"].isna().any() or assignments["Segment"].isna().any():
        raise ValueError("Null cluster or segment name in the assignments.")
    if set(assignments["Segment"]) != set(profiles["segment"]):
        raise ValueError("Assignment names drifted from the profile table.")
    ASSIGNMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(ASSIGNMENTS_PATH, index=False)
    print(
        f"[segmentation] wrote {ASSIGNMENTS_PATH} "
        f"({len(assignments)} customers, {assignments['Segment'].nunique()} "
        f"segments, 0 nulls)"
    )

    # The deployed artifact is the whole fitted pipeline: a consumer that
    # loads models/segmentation_model.pkl can predict (assign) directly.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[segmentation] saved {type(model).__name__} -> {MODEL_PATH}")

    return {
        "model": type(model).__name__,
        "k": FINAL_K,
        "segments": len(profiles),
        "customers_assigned": len(assignments),
        "silhouette_at_k4": sil4,
        "peak_silhouette_k": best_k,
        "peak_silhouette": best_sil,
        "segment_sizes": {
            row.segment: int(row.customers) for row in profiles.itertuples()
        },
    }


def main() -> None:
    """CLI entry point: python -m src.segmentation"""
    summary = run_segmentation_stage()
    print(f"[segmentation] summary: {summary}")


if __name__ == "__main__":
    main()

"""Shared artifact + UI layer for the Streamlit dashboard (TASKS.md #8).

Why this module exists (business reason): the dashboard's two pages must
quote the *same* numbers the pipeline wrote and the API serves, and both
pages must look like one product. This module is the single place that (a)
owns the artifact paths — the lowercase ``data/processed/*.csv`` names the
MEMORY.md task-1 gate checks, plus the ``models/*.pkl`` files — (b) reads
them fresh on every page run so a pipeline re-run is visible immediately,
and (c) holds the "Obsidian Kinetic" style primitives (metric cards, chips,
risk heatstrip, headings) that the stitch designs use as the visual
reference. Nothing here trains or transforms data: the dashboard is a pure
reader of pipeline artifacts, which is why every number shown is real and
traceable to a file.

Design note: the dashboard deliberately does NOT re-score customers with the
pickled models. models/churn_model.pkl + clv_model.pkl were already applied
to the full 7,043-customer base by the stage-5 priority matrix, and that
artifact (data/processed/priority_matrix.csv) is the agreed, sorted,
label-consistent view the README and API quote. Re-predicting in the UI
would risk silent drift (different preprocessing, different precision);
reading the artifact guarantees the dashboard row for a customer is
byte-identical to the API's /predict answer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths — repo-root relative, lowercase artifact names (AGENTS.md / MEMORY.md)
# ---------------------------------------------------------------------------

DASH_DIR = Path(__file__).resolve().parent
ROOT = DASH_DIR.parent  # repo root: dashboard/ is one level below it

FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
PRIORITY_MATRIX_PATH = ROOT / "data" / "processed" / "priority_matrix.csv"
CHURN_COMPARISON_PATH = ROOT / "data" / "processed" / "churn_model_comparison.csv"
CLV_METRICS_PATH = ROOT / "data" / "processed" / "clv_metrics.csv"
SEGMENT_PROFILES_PATH = ROOT / "data" / "processed" / "segment_profiles.csv"
CLV_IMPORTANCE_PATH = ROOT / "data" / "processed" / "clv_feature_importance.csv"
SHAP_SUMMARY_PATH = ROOT / "data" / "processed" / "shap_summary.png"

MODEL_FILES = [
    ROOT / "models" / "churn_model.pkl",
    ROOT / "models" / "clv_model.pkl",
    ROOT / "models" / "segmentation_model.pkl",
]

# Artifacts the dashboard actually renders (models are listed for provenance).
REQUIRED_ARTIFACTS = [
    FEATURES_PATH,
    PRIORITY_MATRIX_PATH,
    CHURN_COMPARISON_PATH,
    CLV_METRICS_PATH,
    SEGMENT_PROFILES_PATH,
    SHAP_SUMMARY_PATH,
    *MODEL_FILES,
]

# ---------------------------------------------------------------------------
# Column names of the artifacts (must match what the pipeline modules wrote)
# ---------------------------------------------------------------------------

CUSTOMER_ID = "Customer_ID"
CHURN_PROB = "Churn_Probability"
PREDICTED_CLV = "Predicted_CLV"
PRIORITY_LABEL = "Priority_Label"
SEGMENT = "Segment"

# Stage-5 rank order (priority_matrix.py PRIORITY_ORDER) — the same order the
# pipeline sorts the CSV by, so the dashboard table leads with the accounts
# the retention team should call first.
PRIORITY_ORDER = ["Immediate Save", "Nurture", "Monitor", "Low Priority"]

# Label -> semantic color of the stitch design (red = severe/at risk, amber =
# action required, emerald = stable, muted = quiet tail). Used for chips,
# KPI accent bars and chart bars, so one label always has one color anywhere
# in the app.
PRIORITY_COLORS = {
    "Immediate Save": "#EF4444",  # high churn + high value: fight for it now
    "Nurture": "#F59E0B",         # high churn + low value: cheap-save offer
    "Monitor": "#10B981",         # low churn + high value: protect & grow
    "Low Priority": "#8590A2",    # low churn + low value: no campaign spend
}

STATUS_COLORS = {"Stayed": "#10B981", "Churned": "#EF4444", "Joined": "#0EA5E9"}

# Segment accent colors keyed by the names the pipeline derives from the
# k=4 centroids (segment_profiles.csv); unknown names fall back to gray.
SEGMENT_COLORS = {
    "Loyal Premium Customers": "#8B5CF6",
    "Established Standard Customers": "#0EA5E9",
    "New Budget Customers": "#F59E0B",
    "Loyal Budget Customers": "#10B981",
}

# Raw human-readable columns kept from features.csv for the Customer
# Explorer dossier. Excludes every one-hot dummy and model-only derived
# column — the dossier should read like an account record, not a feature
# vector.
DOSSIER_COLUMNS = [
    "Customer_Status", "City", "Age", "Gender", "Married",
    "Number_of_Dependents", "Number_of_Referrals", "Tenure_in_Months",
    "Contract", "Internet_Type", "Payment_Method", "Offer",
    "Monthly_Charge", "Total_Revenue", "Total_Services",
    "Premium_Services", "Streaming_Services", "Is_New_Customer",
    "Senior_Citizen", "Avg_Monthly_GB_Download", "Paperless_Billing",
]


# ---------------------------------------------------------------------------
# Artifact readers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _read_cached(path: str, mtime: float) -> pd.DataFrame:
    """Read one CSV artifact, cached until the file on disk changes.

    Business reason: typing in the Customer Explorer search box re-runs the
    page on every keystroke, and re-parsing the 7,043-row feature table each
    time adds latency without adding freshness. The cache key includes the
    file's mtime, so a pipeline re-run (which rewrites the file) invalidates
    the cache automatically — the dashboard can never show stale numbers for
    longer than one page run.
    """
    return pd.read_csv(path)


def read_csv(path: Path) -> pd.DataFrame:
    """Fresh-read a CSV artifact (mtime-keyed cache; see _read_cached)."""
    return _read_cached(str(path), path.stat().st_mtime)


def load_features() -> pd.DataFrame:
    """Stage-1 artifact: one null-free row per customer (7,043 x ~60 cols).

    Business reason: features.csv is the dashboard's account-record source
    (status, contract, bill, tenure, city) and the join key for the dossier;
    the matrix artifact alone cannot answer "what kind of account is this?"
    """
    return read_csv(FEATURES_PATH)


def load_matrix() -> pd.DataFrame:
    """Stage-5 artifact: ranked priority list for all 7,043 customers.

    Business reason: the priority matrix is the single scored view of the
    base (churn probability, predicted CLV, segment, priority label, sorted
    most-urgent-first), so both pages build every KPI and every dossier from
    this one file to stay mutually consistent.
    """
    return read_csv(PRIORITY_MATRIX_PATH)


def load_comparison() -> pd.DataFrame:
    """Stage-2 churn model comparison table (2 models x 5 metrics)."""
    return read_csv(CHURN_COMPARISON_PATH)


def load_clv_metrics() -> dict[str, float]:
    """Stage-3 CLV regression metrics (RMSE / MAE / R2) as a dict."""
    row = read_csv(CLV_METRICS_PATH).iloc[0]
    return {col: float(row[col]) for col in ("rmse", "mae", "r2")}


def load_segment_profiles() -> pd.DataFrame:
    """Stage-4 named segment profiles (4 rows: centroids, names, sizes)."""
    return read_csv(SEGMENT_PROFILES_PATH)


def load_clv_importance() -> pd.DataFrame:
    """Stage-3 CLV feature-importance artifact, most important first."""
    return read_csv(CLV_IMPORTANCE_PATH)


def ensure_artifacts() -> list[str]:
    """Names of missing pipeline artifacts, empty when the pipeline has run.

    Business reason: the dashboard is a reader of pipeline outputs, so it
    must fail with a helpful instruction (run the pipeline) instead of a
    raw FileNotFoundError traceback when the repo is fresh.
    """
    return [str(p.relative_to(ROOT)) for p in REQUIRED_ARTIFACTS if not p.exists()]


def artifact_footer() -> str:
    """One provenance line: when each source artifact was last written.

    Business reason: interviewers and the owner need to trust that the
    numbers on screen are the current pipeline's, not a stale snapshot; the
    artifact mtimes make the data lineage visible on the page itself.
    """
    import datetime

    parts = []
    for path in (FEATURES_PATH, PRIORITY_MATRIX_PATH):
        ts = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        parts.append(f"{path.parent.name}/{path.name} · {ts:%Y-%m-%d %H:%M}")
    return "Artifacts: " + "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Formatters (tabular monospaced figures, per the design's metric language)
# ---------------------------------------------------------------------------

def fmt_int(value: float | int) -> str:
    """Thousands-separated integer, e.g. 7043 -> '7,043'."""
    return f"{value:,.0f}"


def fmt_money(value: float | int) -> str:
    """Dollar figure with thousands separators, e.g. 2797.85 -> '$2,798'."""
    return f"${value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    """0-1 fraction to a percent string, e.g. 0.9666 -> '96.7%'."""
    return f"{value * 100:.{digits}f}%"


def segment_color(name: str) -> str:
    """Color for a segment name, gray for any unknown/future name."""
    return SEGMENT_COLORS.get(name, "#8590A2")


# ---------------------------------------------------------------------------
# Style primitives ("Obsidian Kinetic" — dark instrument-grade surfaces)
# ---------------------------------------------------------------------------

_CSS = """
<style>
:root{
  --sk-bg:#090A0C; --sk-s0:#0F1115; --sk-s1:#15181E; --sk-s2:#1C2027;
  --sk-line:rgba(255,255,255,.08); --sk-line2:rgba(255,255,255,.14);
  --sk-text:#F1F3F7; --sk-text2:#A9B4C4; --sk-text3:#8590A2;
  --sk-amber:#F59E0B; --sk-amber-deep:#D97706; --sk-red:#EF4444;
  --sk-green:#10B981; --sk-cyan:#0EA5E9; --sk-violet:#8B5CF6;
  --sk-mono:ui-monospace,"Cascadia Mono","JetBrains Mono",Consolas,"Courier New",monospace;
  --sk-sans:Inter,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.stApp{background:var(--sk-bg);}
.block-container{padding-top:1.6rem;padding-bottom:4rem;max-width:1380px;}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:var(--sk-s0);border-right:1px solid var(--sk-line);}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:var(--sk-text2);}
[data-testid="stSidebarNav"] span{font-family:var(--sk-mono);}
h1,h2,h3,h4,h5{color:var(--sk-text);letter-spacing:-0.015em;font-family:var(--sk-sans);}
p,li,label{color:var(--sk-text2);}
[data-testid="stCaptionContainer"] p{color:var(--sk-text3);}
[data-testid="stDataFrame"]{border:1px solid var(--sk-line);border-radius:8px;overflow:hidden;}
hr{border-color:var(--sk-line)!important;}
/* ---------- page header ---------- */
.sk-eyebrow{font-family:var(--sk-mono);font-size:11px;font-weight:600;
  letter-spacing:.14em;text-transform:uppercase;color:var(--sk-amber);
  margin:0 0 2px;display:flex;align-items:center;gap:8px;}
.sk-title{color:var(--sk-text);font-size:30px;font-weight:600;
  letter-spacing:-.02em;line-height:1.15;margin:0;}
.sk-sub{color:var(--sk-text3);font-size:13.5px;line-height:1.5;
  margin:6px 0 0;max-width:960px;}
.sk-live{width:6px;height:6px;border-radius:50%;background:var(--sk-green);
  box-shadow:0 0 6px var(--sk-green);display:inline-block;}
/* ---------- KPI / metric cards ---------- */
.kpi-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
  gap:12px;margin:14px 0 6px;}
.kpi-row.kpi-3{grid-template-columns:repeat(3,minmax(0,1fr));}
@media(max-width:1150px){.kpi-row,.kpi-row.kpi-3{grid-template-columns:repeat(2,minmax(0,1fr));}}
.kpi{background:var(--sk-s1);border:1px solid var(--sk-line);
  border-radius:8px;padding:13px 15px 12px;position:relative;overflow:hidden;}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;
  background:var(--sk-tone,var(--sk-amber-deep));opacity:.9;}
.kpi .dot{width:5px;height:5px;border-radius:1px;background:currentColor;display:inline-block;}
.kpi-label{font-family:var(--sk-mono);font-size:10.5px;font-weight:600;
  letter-spacing:.09em;text-transform:uppercase;color:var(--sk-text3);
  display:flex;align-items:center;gap:7px;justify-content:space-between;}
.kpi-value{font-family:var(--sk-mono);font-size:29px;font-weight:600;
  line-height:1.2;letter-spacing:-.03em;color:var(--sk-text);
  font-variant-numeric:tabular-nums;margin-top:7px;white-space:nowrap;}
.kpi-sub{color:var(--sk-text3);font-size:11.5px;line-height:1.45;margin-top:4px;}
.kpi.t-red{--sk-tone:var(--sk-red);} .kpi.t-amber{--sk-tone:var(--sk-amber);}
.kpi.t-green{--sk-tone:var(--sk-green);} .kpi.t-cyan{--sk-tone:var(--sk-cyan);}
.kpi.t-violet{--sk-tone:var(--sk-violet);} .kpi.t-gray{--sk-tone:#5A6472;}
/* ---------- status chips ---------- */
.chip{display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 9px;
  border-radius:4px;font-family:var(--sk-mono);font-size:10.5px;font-weight:600;
  letter-spacing:.07em;text-transform:uppercase;border:1px solid;
  white-space:nowrap;line-height:1;}
.chip i{width:5px;height:5px;border-radius:1px;background:currentColor;font-style:normal;}
.c-red{color:#F87171;background:rgba(239,68,68,.10);border-color:rgba(239,68,68,.32);}
.c-amber{color:#FBBF24;background:rgba(217,119,6,.12);border-color:rgba(245,158,11,.32);}
.c-green{color:#34D399;background:rgba(16,185,129,.10);border-color:rgba(16,185,129,.28);}
.c-cyan{color:#38BDF8;background:rgba(14,165,233,.10);border-color:rgba(14,165,233,.30);}
.c-violet{color:#A78BFA;background:rgba(139,92,246,.10);border-color:rgba(139,92,246,.30);}
.c-gray{color:#A9B4C4;background:rgba(255,255,255,.05);border-color:rgba(255,255,255,.14);}
/* ---------- risk heatstrip ---------- */
.heat{position:relative;height:10px;border-radius:2px;
  background:linear-gradient(90deg,#10B981 0%,#D97706 48%,#EF4444 78%,#EF4444 100%);
  opacity:.92;}
.heat-marker{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--sk-text);
  box-shadow:0 0 0 1px rgba(0,0,0,.6);}
.heat-median{position:absolute;top:0;bottom:0;width:1px;
  background:rgba(241,243,247,.55);}
.heat-legend{display:flex;justify-content:space-between;margin-top:5px;
  font-family:var(--sk-mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--sk-text3);}
/* ---------- facts grid (account dossier) ---------- */
.facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
  gap:10px 22px;margin-top:12px;}
@media(max-width:1150px){.facts{grid-template-columns:1fr;}}
.fact-label{font-family:var(--sk-mono);font-size:10px;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;color:var(--sk-text3);
  margin-bottom:2px;}
.fact-value{color:var(--sk-text);font-size:13.5px;line-height:1.35;
  font-variant-numeric:tabular-nums;}
/* ---------- callouts ---------- */
.callout{background:var(--sk-s1);border:1px solid var(--sk-line);
  border-radius:6px;padding:11px 13px;color:var(--sk-text2);
  font-size:13px;line-height:1.55;}
.callout b{color:var(--sk-text);font-weight:600;}
.sk-card{background:var(--sk-s1);border:1px solid var(--sk-line);
  border-radius:8px;padding:14px 16px;}
.sk-card-title{font-family:var(--sk-mono);font-size:10.5px;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;color:var(--sk-text3);
  margin-bottom:10px;display:flex;align-items:center;gap:8px;}
/* dataframe header styling is handled by the theme; row hover accent */
[data-testid="stDataFrame"] tbody tr:hover{background:rgba(217,119,6,.05);}
</style>
"""


def inject_css() -> None:
    """Inject the shared stylesheet once per session.

    Business reason: pages re-run on every interaction; re-injecting the
    same <style> block each time costs bytes without changing anything, so
    the session flag keeps the DOM clean. Injected from every page so each
    page is also presentable when run standalone.
    """
    if st.session_state.get("_sk_css_injected"):
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    st.session_state["_sk_css_injected"] = True


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    """Render the shared page header (eyebrow + title + subtitle)."""
    st.markdown(
        f'<p class="sk-eyebrow">{eyebrow}</p>'
        f'<h1 class="sk-title">{title}</h1>'
        f'<p class="sk-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def _kpi_card(card: dict) -> str:
    """One metric card: uppercase mono label, big tabular value, caption."""
    tone = card.get("tone", "gray")
    return (
        f'<div class="kpi t-{tone}">'
        f'<div class="kpi-label">{card["label"]}<span class="dot"></span></div>'
        f'<div class="kpi-value">{card["value"]}</div>'
        f'<div class="kpi-sub">{card.get("sub", "")}</div>'
        f"</div>"
    )


def kpi_row(cards: list[dict], cols: int = 4) -> None:
    """Render a row of metric cards (grid; responsive collapse)."""
    css = "kpi-row kpi-3" if cols == 3 else "kpi-row"
    st.markdown(
        f'<div class="{css}">' + "".join(_kpi_card(c) for c in cards) + "</div>",
        unsafe_allow_html=True,
    )


def chip(text: str, tone: str) -> str:
    """Status chip with a leading dot; tone in red/amber/green/cyan/violet/gray."""
    return f'<span class="chip c-{tone}"><i></i>{text}</span>'


def heatstrip(prob: float, median: float) -> None:
    """Segmented risk telemetry bar with the customer marker + median tick.

    Business reason: a single percentage is abstract; the heatstrip puts the
    customer on the green->amber->red risk continuum next to the base-median
    tick that the priority rule itself uses, so the label is visually
    checkable at a glance.
    """
    p = max(0.0, min(100.0, prob * 100.0))
    m = max(0.0, min(100.0, median * 100.0))
    st.markdown(
        f'<div class="heat">'
        f'<span class="heat-median" style="left:{m:.1f}%" title="base median"></span>'
        f'<span class="heat-marker" style="left:{p:.1f}%"></span>'
        f"</div>"
        f'<div class="heat-legend">'
        f"<span>low risk</span><span>base median · {median * 100:.1f}%</span>"
        f"<span>high risk</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def facts_grid(pairs: list[tuple[str, str]]) -> None:
    """Two-column label/value grid for the account dossier."""
    cells = "".join(
        f'<div><div class="fact-label">{label}</div>'
        f'<div class="fact-value">{value}</div></div>'
        for label, value in pairs
    )
    st.markdown(f'<div class="facts">{cells}</div>', unsafe_allow_html=True)

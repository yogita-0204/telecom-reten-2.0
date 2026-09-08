"""Customer Explorer — page 2 of the Telecom Intelligence dashboard.

What this page shows (TASKS.md #8): search any of the 7,043 customers by
Customer_ID and read that account's full retention dossier — churn
prediction (probability + position vs. the base median), predicted CLV,
segment name, and retention priority label — exactly the four outputs the
PRD promises and the API's /predict returns. The ranked list on top doubles
as the "who to save first" queue: rows are the stage-5 priority matrix in
its own order, and clicking a row opens the dossier beneath.

Why this page exists (business reason): user story 3 ("who to save first,
not just a raw probability") needs a person-level view, not only aggregate
charts — a retention rep searches an account and must see, in one glance,
how urgent the risk is (heatstrip vs. the base median), how much value is at
stake (predicted CLV), what kind of account it is (segment), and the exact
action the matrix assigns (priority label). All values come from
data/processed/priority_matrix.csv joined to features.csv on Customer_ID —
no re-scoring in the UI, so the dossier is byte-identical to the pipeline's
and the API's answer for the same customer (see dashboard/_lib.py docstring).
"""

from __future__ import annotations

import sys
from pathlib import Path

_DASH = Path(__file__).resolve().parents[1]
_ROOT = _DASH.parent
for _p in (str(_DASH), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

import _lib as L

# ---------------------------------------------------------------------------
# Guard: artifacts must exist (fresh repo -> helpful error, not a traceback)
# ---------------------------------------------------------------------------
_missing = L.ensure_artifacts()
if _missing:
    st.error(
        "Pipeline artifacts not found — run `python -m src.run_pipeline` "
        f"from the repo root first. Missing: {', '.join(_missing)}"
    )
    st.stop()

L.inject_css()

features = L.load_features()
matrix = L.load_matrix()

base = len(matrix)
med_churn = float(matrix[L.CHURN_PROB].median())
med_clv = float(matrix[L.PREDICTED_CLV].median())

# One row per customer: priority scores (stage 5) + the human-readable
# account record (stage 1). Joined on Customer_ID — never on row order,
# because the matrix is sorted by urgency while features.csv is not.
dossier_cols = [c for c in L.DOSSIER_COLUMNS if c in features.columns]
joined = matrix.merge(
    features[[L.CUSTOMER_ID, *dossier_cols]],
    on=L.CUSTOMER_ID,
    how="left",
    validate="one_to_one",
)
if joined[L.PRIORITY_LABEL].isna().any():
    st.error("Priority labels missing for some customers in the join.")
    st.stop()

# ---------------------------------------------------------------------------
# Priority-label playbooks (the action behind each quadrant)
# ---------------------------------------------------------------------------
LABEL_STORY = {
    "Immediate Save": (
        "<b>Full save campaign.</b> High churn risk on a high-value account — "
        "the revenue the business must fight for right now: executive or "
        "retention-team outreach, a tailored win-back offer (contract, offer, "
        "service mix), and a follow-up within days, not weeks."
    ),
    "Nurture": (
        "<b>Cheap save offer.</b> High churn risk, but the account's predicted "
        "value does not justify a full campaign — a low-cost offer (discount, "
        "short-term incentive) still beats losing the account, and a batch "
        "nurture flow can carry it."
    ),
    "Monitor": (
        "<b>Protect & grow.</b> Low churn risk on a high-value account — the "
        "base's quiet value engine. No firefighting needed; loyalty perks, "
        "quarterly check-ins and upsell paths keep the relationship healthy."
    ),
    "Low Priority": (
        "<b>No campaign spend.</b> Low churn risk and low predicted value — "
        "standard service only; retention budget is better spent on the "
        "other three quadrants."
    ),
}
LABEL_TONES = {
    "Immediate Save": "red",
    "Nurture": "amber",
    "Monitor": "green",
    "Low Priority": "gray",
}

L.page_header(
    "Strata Intelligence · Customer Explorer",
    "Customer Explorer",
    f"Search the full {L.fmt_int(base)}-account base, ranked as the retention "
    f"queue: every row is one customer's churn probability, predicted CLV, "
    f"segment and priority label from data/processed/priority_matrix.csv — "
    f"click a row (or type a Customer_ID) to open the account dossier.",
)

# ---------------------------------------------------------------------------
# Controls: search box + quadrant / segment filters
# ---------------------------------------------------------------------------
fcol, pcol, scol = st.columns([0.36, 0.32, 0.32], gap="medium")
with fcol:
    query = st.text_input(
        "Customer search",
        placeholder="e.g. 6900-PXRMS or 2845…",
        label_visibility="collapsed",
    ).strip()
with pcol:
    sel_priorities = st.multiselect(
        "Priority", L.PRIORITY_ORDER, default=L.PRIORITY_ORDER
    )
with scol:
    segments = list(joined[L.SEGMENT].unique())
    sel_segments = st.multiselect("Segment", segments, default=segments)

# --- filter the ranked queue -------------------------------------------------
mask = pd.Series(True, index=joined.index)
if query:
    mask &= joined[L.CUSTOMER_ID].astype(str).str.contains(
        query, case=False, regex=False, na=False
    )
if sel_priorities:
    mask &= joined[L.PRIORITY_LABEL].isin(sel_priorities)
if sel_segments:
    mask &= joined[L.SEGMENT].isin(sel_segments)
queue = joined.loc[mask].reset_index(drop=True)

st.caption(
    f"Showing {len(queue):,} of {base:,} accounts · ranked by priority "
    "(most urgent first) · click a row to inspect"
)

event = st.dataframe(
    queue[
        [
            L.CUSTOMER_ID, "Customer_Status", L.PRIORITY_LABEL,
            L.CHURN_PROB, L.PREDICTED_CLV, L.SEGMENT,
            "Tenure_in_Months", "Monthly_Charge",
        ]
    ].rename(
        columns={
            L.CUSTOMER_ID: "Customer ID",
            "Customer_Status": "Status",
            L.PRIORITY_LABEL: "Priority",
            L.CHURN_PROB: "risk_pct",
            L.PREDICTED_CLV: "Predicted CLV",
            L.SEGMENT: "Segment",
            "Tenure_in_Months": "Tenure (mo)",
            "Monthly_Charge": "Monthly bill",
        }
    ),
    on_select="rerun",
    selection_mode="single-row",
    key="explorer_queue",
    hide_index=True,
    width="stretch",
    height=430,
    column_config={
        "Customer ID": st.column_config.TextColumn("Customer ID", width="medium"),
        "Priority": st.column_config.TextColumn("Priority", width="small"),
        "risk_pct": st.column_config.ProgressColumn(
            "Churn risk",
            min_value=0.0,
            max_value=100.0,
            format="%.0f%%",
            help="P(churned) from the deployed churn model",
        ),
        "Predicted CLV": st.column_config.NumberColumn(
            "Predicted CLV", format="$%,.0f"
        ),
        "Monthly bill": st.column_config.NumberColumn(
            "Monthly bill", format="$%,.0f"
        ),
    },
)

if queue.empty:
    st.info("No customers match the current search / filters.")
    st.stop()

# --- which row does the dossier show? ---------------------------------------
# Priority: an exact unique search hit > the clicked row > the top of the
# filtered queue (so the page always shows a real dossier on load).
sel_rows = list(getattr(getattr(event, "selection", None), "rows", []) or [])
selected_index = None
if sel_rows:
    raw = sel_rows[0]
    if isinstance(raw, int):
        selected_index = raw
    elif isinstance(raw, dict):
        selected_index = raw.get("index", raw.get("row"))
    else:  # Row-like object carrying .index
        selected_index = getattr(raw, "index", None)

if query and len(queue) == 1:
    selected_index = 0
if selected_index is None or not 0 <= int(selected_index) < len(queue):
    selected_index = 0
row = queue.iloc[int(selected_index)]

customer_id = str(row[L.CUSTOMER_ID])
prob = float(row[L.CHURN_PROB])
clv = float(row[L.PREDICTED_CLV])
label = str(row[L.PRIORITY_LABEL])
segment = str(row[L.SEGMENT])
status = str(row["Customer_Status"])
high_churn = prob >= med_churn
high_value = clv >= med_clv
tone = LABEL_TONES[label]

# ---------------------------------------------------------------------------
# Dossier
# ---------------------------------------------------------------------------
st.markdown(
    '<div style="height:14px"></div>'
    '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
    f'<span style="font-family:var(--sk-mono);font-size:20px;font-weight:600;'
    f'letter-spacing:-.02em;color:var(--sk-text);">{customer_id}</span>'
    f'{L.chip(label, tone)}'
    f'{L.chip(status, {"Stayed": "green", "Churned": "red", "Joined": "cyan"}.get(status, "gray"))}'
    f'{L.chip(segment, "violet")}'
    '</div>',
    unsafe_allow_html=True,
)

L.kpi_row(
    [
        {
            "label": "Churn Probability",
            "value": L.fmt_pct(prob),
            "sub": f"{'at/above' if high_churn else 'below'} the base median "
            f"({L.fmt_pct(med_churn, 2)}) — the label's risk axis",
            "tone": tone,
        },
        {
            "label": "Predicted CLV",
            "value": L.fmt_money(clv),
            "sub": f"{'at/above' if high_value else 'below'} the base median "
            f"(${med_clv:,.0f}) — the label's value axis",
            "tone": tone,
        },
        {
            "label": "Actual Total Revenue",
            "value": L.fmt_money(float(row["Total_Revenue"])),
            "sub": "revenue to date — the CLV model's target, for reference",
            "tone": "gray",
        },
        {
            "label": "Tenure",
            "value": f'{int(row["Tenure_in_Months"])} mo',
            "sub": "months with the carrier — the strongest CLV driver",
            "tone": "gray",
        },
    ]
)

L.heatstrip(prob, med_churn)

# Rule sentence: reproduce the 2x2 decision from the two numbers on screen.
st.markdown(
    '<div style="height:10px"></div>'
    '<div class="callout">'
    f'<b>Retention rule applied:</b> churn probability '
    f'{">= base median" if high_churn else "< base median"} → '
    f'{"HIGH risk" if high_churn else "low risk"} · predicted CLV '
    f'{">= base median" if high_value else "< base median"} → '
    f'{"HIGH value" if high_value else "low value"} ⇒ '
    f'<b style="color:{L.PRIORITY_COLORS[label]}">{label.upper()}</b>'
    f'<br>{LABEL_STORY[label]}'
    "</div>",
    unsafe_allow_html=True,
)

# --- account dossier grid ------------------------------------------------------
st.markdown(
    '<div style="height:10px"></div>'
    '<div class="sk-card">'
    '<div class="sk-card-title">Account dossier</div>',
    unsafe_allow_html=True,
)

_GENDER = {0: "Female", 1: "Male"}
_YESNO = {0: "No", 1: "Yes"}


def _yn(value) -> str:
    """Decode the pipeline's 0/1 Yes-No flags back to words for display."""
    return _YESNO.get(int(value), str(value))


profile_pairs = [
    ("Contract", str(row["Contract"])),
    ("Internet", str(row["Internet_Type"])),
    ("Payment", str(row["Payment_Method"])),
    ("Offer", str(row["Offer"])),
    (
        "Age / Gender",
        f'{int(row["Age"])} · {_GENDER.get(int(row["Gender"]), "—")}',
    ),
    (
        "Household",
        f"{_yn(row['Married'])} · {int(row['Number_of_Dependents'])} dependents",
    ),
    ("City", str(row["City"])),
    (
        "Services",
        f'{int(row["Total_Services"])} total · '
        f'{int(row["Premium_Services"])} premium · '
        f'{int(row["Streaming_Services"])} streaming',
    ),
    (
        "Billing profile",
        f'{L.fmt_money(float(row["Monthly_Charge"]))}/mo · '
        f'{_yn(row["Paperless_Billing"])} paperless',
    ),
    (
        "Referrals",
        f'{int(row["Number_of_Referrals"])} '
        f'({float(row["Number_of_Referrals"]) / max(float(row["Tenure_in_Months"]) / 12, 1e-9):.1f}/yr tenure)',
    ),
    (
        "Download usage",
        f'{float(row["Avg_Monthly_GB_Download"]):.0f} GB/mo avg',
    ),
    (
        "Status flags",
        f'{"new customer (≤6 mo)" if int(row["Is_New_Customer"]) else "established"} · '
        f'{"senior (≥60)" if int(row["Senior_Citizen"]) else "under 60"}',
    ),
]
L.facts_grid(profile_pairs)

st.markdown(
    '<div style="margin-top:12px;padding-top:10px;border-top:1px solid '
    'rgba(255,255,255,.07);font-family:var(--sk-mono);font-size:10px;'
    'letter-spacing:.08em;text-transform:uppercase;color:var(--sk-text3);">'
    f'Source · features.csv + priority_matrix.csv · model v2 (seed 42)</div>'
    "</div>",
    unsafe_allow_html=True,
)

st.divider()
st.caption(L.artifact_footer())

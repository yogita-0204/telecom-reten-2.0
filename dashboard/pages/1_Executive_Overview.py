"""Executive Overview — page 1 of the Telecom Intelligence dashboard.

What this page shows (TASKS.md #8): real KPIs of the 7,043-account base,
the four-label retention priority distribution, the churn-model comparison
table, and the SHAP global summary plot — the four things a recruiter or
interviewer should absorb in under a minute (PRD user story 1). Every number
is computed at run time from the pipeline artifacts in data/processed/;
nothing is hard-coded, so the page can never display a number the pipeline
did not just write.

Why this page exists (business reason): the overview is the project's front
door. It must show completeness (all four PRD problems solved), real
results (not placeholders), and traceability (which model, which artifact)
in one screenful — the same story the README and the interview answer tell.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DASH = Path(__file__).resolve().parents[1]
_ROOT = _DASH.parent
for _p in (str(_DASH), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plotly.graph_objects as go
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
comparison = L.load_comparison()
clv_metrics = L.load_clv_metrics()
profiles = L.load_segment_profiles()
importance = L.load_clv_importance()

# --- Shared base facts (single source: the artifacts themselves) -----------
status_counts = features["Customer_Status"].value_counts().to_dict()
stayed = int(status_counts.get("Stayed", 0))
churned = int(status_counts.get("Churned", 0))
joined = int(status_counts.get("Joined", 0))
resolved = stayed + churned
base = len(matrix)
churn_rate = churned / resolved

counts = matrix[L.PRIORITY_LABEL].value_counts().to_dict()
prio_counts = {label: int(counts.get(label, 0)) for label in L.PRIORITY_ORDER}
at_risk = prio_counts["Immediate Save"] + prio_counts["Nurture"]
clv_at_risk = float(
    matrix.loc[
        matrix[L.PRIORITY_LABEL].isin(["Immediate Save", "Nurture"]),
        L.PREDICTED_CLV,
    ].sum()
)
deployed_model = str(
    comparison.loc[comparison["roc_auc"].idxmax(), "model"]
)

# ---------------------------------------------------------------------------
# Header + KPI deck
# ---------------------------------------------------------------------------
L.page_header(
    "Strata Intelligence · Telecom Retention",
    "Executive Overview",
    f"Customer intelligence & retention risk across the full {L.fmt_int(base)}-"
    f"account base — churn risk scored by the deployed {deployed_model} model, "
    f"predicted value by the stage-3 CLV regressor, and every account ranked "
    f"into the four-action retention matrix (Immediate Save / Nurture / "
    f"Monitor / Low Priority).",
)

L.kpi_row(
    [
        {
            "label": "Total Customers",
            "value": L.fmt_int(base),
            "sub": f"{stayed:,} stayed · {churned:,} churned · {joined:,} joined "
            "(no churn outcome yet)",
            "tone": "cyan",
        },
        {
            "label": "Resolved Churn Rate",
            "value": L.fmt_pct(churn_rate),
            "sub": f"{churned:,} of {resolved:,} resolved accounts — the actual "
            "leavers the models learn from",
            "tone": "red",
        },
        {
            "label": "Predicted At-Risk",
            "value": L.fmt_int(at_risk),
            "sub": "accounts at/above the base-median churn probability "
            "(Immediate Save + Nurture)",
            "tone": "amber",
        },
        {
            "label": "Predicted CLV at Risk",
            "value": L.fmt_money(clv_at_risk),
            "sub": "predicted lifetime revenue of the at-risk set — the budget "
            "the retention team is defending",
            "tone": "red",
        },
    ]
)

# ---------------------------------------------------------------------------
# Priority distribution + customer-status mix
# ---------------------------------------------------------------------------
st.markdown("#### Retention priority distribution")
st.markdown(
    f"Four-label 2×2 matrix from the stage-5 rule: churn probability ≥ the "
    f"base median **{L.fmt_pct(float(matrix[L.CHURN_PROB].median()), 2)}** "
    f"counts as high-risk, predicted CLV ≥ **${matrix[L.PREDICTED_CLV].median():,.0f}** "
    f"counts as high-value.",
    unsafe_allow_html=True,
)

left, right = st.columns([0.62, 0.38], gap="large")
with left:
    labels = L.PRIORITY_ORDER[::-1]  # plotly draws bottom-up; most urgent on top
    values = [prio_counts[lab] for lab in labels]
    colors = [L.PRIORITY_COLORS[lab] for lab in labels]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:,}  ·  {v / base:.1%}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,} accounts (%{x:.1%} of base)<extra></extra>",
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=4, r=64, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#A9B4C4", size=13),
        xaxis=dict(visible=False),
        yaxis=dict(
            tickfont=dict(family="ui-monospace, Consolas, monospace", size=11,
                          color="#8590A2"),
            title=None,
        ),
        bargap=0.4,
        showlegend=False,
    )
    st.plotly_chart(
        fig, width="stretch",
        config={"displayModeBar": False, "staticPlot": False},
    )
    quadrant_hint = (
        "**Immediate Save** = high churn + high value (full save campaign) · "
        "**Nurture** = high churn + low value (cheap save offer) · "
        "**Monitor** = low churn + high value (protect & grow) · "
        "**Low Priority** = low churn + low value (no campaign spend)."
    )
    st.caption(quadrant_hint)

with right:
    status_labels = ["Stayed", "Churned", "Joined"]
    status_values = [status_counts.get(s, 0) for s in status_labels]
    donut = go.Figure(
        go.Pie(
            labels=status_labels,
            values=status_values,
            hole=0.74,
            marker=dict(colors=[L.STATUS_COLORS[s] for s in status_labels],
                        line=dict(color="#090A0C", width=2)),
            textinfo="none",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        )
    )
    donut.add_annotation(
        text=f"<b>{base:,}</b><br><span style='font-size:10px'>ACCOUNTS</span>",
        showarrow=False,
        font=dict(family="ui-monospace, Consolas, monospace", size=15,
                  color="#F1F3F7"),
    )
    donut.update_layout(
        height=260,
        margin=dict(l=4, r=4, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#A9B4C4"),
        showlegend=False,
    )
    st.plotly_chart(donut, width="stretch",
                    config={"displayModeBar": False})
    st.caption(
        f"**{churned:,}** of the {resolved:,} resolved accounts actually churned "
        f"({L.fmt_pct(churn_rate)}); the **{joined:,}** Joined accounts are recent "
        "sign-ups with no churn outcome yet, so they are scored but never "
        "used to train the models."
    )

# ---------------------------------------------------------------------------
# Model comparison + SHAP global explanation
# ---------------------------------------------------------------------------
st.markdown("#### Churn model comparison & global explanation")

mleft, mright = st.columns([0.5, 0.5], gap="large")
with mleft:
    styled = comparison.copy()
    deployed_idx = int(styled["roc_auc"].idxmax())
    styler = (
        styled.style.format(
            {
                "accuracy": "{:.1%}", "precision": "{:.1%}",
                "recall": "{:.1%}", "f1": "{:.1%}", "roc_auc": "{:.1%}",
            }
        )
        .hide(axis="index")
        .apply(
            lambda row: [
                "background-color: rgba(217,119,6,0.13); color:#F1F3F7;"
                "font-weight:600"
                if row.name == deployed_idx
                else ""
            ]
            * len(row),
            axis=1,
        )
    )
    st.dataframe(styler, width="stretch", height=140)
    st.caption(
        f"Metrics on the stratified held-out 20% test split (churners are the "
        f"minority class, so accuracy alone would flatter the models). "
        f"**{deployed_model}** ships as models/churn_model.pkl because its "
        f"ROC-AUC of {comparison.loc[deployed_idx, 'roc_auc']:.4f} ranks the "
        f"save list best; the Random Forest stays for the SHAP story next "
        f"to it."
    )

with mright:
    st.image(str(L.SHAP_SUMMARY_PATH), width="stretch")
    st.caption(
        "Global SHAP summary (TreeExplainer on the Random Forest): each dot is "
        "one customer; the x-position is how strongly a feature pushed that "
        "customer's churn probability, red = high feature value, blue = low. "
        "Month-to-month contracts, short tenure and fiber-optic bills are the "
        "drivers a retention team can act on — traceability for any "
        "prediction (PRD user story 2)."
    )

# ---------------------------------------------------------------------------
# Segments + CLV model value
# ---------------------------------------------------------------------------
st.markdown("#### Segments & predicted customer value")

sleft, sright = st.columns([0.5, 0.5], gap="large")
with sleft:
    # Per-segment concentration of the save budget, computed from the same
    # priority matrix artifact the KPI cards use, enriched with the stage-4
    # centroid description of what each segment IS (tenure/bill).
    seg_stats = (
        matrix.groupby(L.SEGMENT)
        .agg(
            accounts=(L.CUSTOMER_ID, "size"),
            immediate_save=(L.PRIORITY_LABEL,
                            lambda s: int((s == "Immediate Save").sum())),
            at_risk_clv=(L.PREDICTED_CLV, "sum"),
        )
        .reset_index()
    )
    centroid_map = profiles.set_index("segment")[
        ["Tenure_in_Months_centroid", "Monthly_Charge_centroid"]
    ]
    seg_view = seg_stats.join(
        centroid_map, on=L.SEGMENT
    ).sort_values("accounts", ascending=False).rename(
        columns={
            "accounts": "Accounts",
            "immediate_save": "Immediate Save",
            "at_risk_clv": "At-risk CLV",
            "Tenure_in_Months_centroid": "Centroid tenure (mo)",
            "Monthly_Charge_centroid": "Centroid bill $",
        }
    )
    seg_styler = (
        seg_view[
            [L.SEGMENT, "Accounts", "Immediate Save", "At-risk CLV",
             "Centroid tenure (mo)", "Centroid bill $"]
        ]
        .style.format(
            {
                "Accounts": "{:,}",
                "Immediate Save": "{:,}",
                "At-risk CLV": "${:,.0f}",
                "Centroid tenure (mo)": "{:.0f}",
                "Centroid bill $": "${:,.0f}",
            }
        )
        .hide(axis="index")
    )
    st.dataframe(seg_styler, width="stretch", height=250)
    st.caption(
        "Four K-Means segments named from their centroid terciles on tenure × "
        "monthly charge × services (stage 4); columns 4-6 are the stage-5 "
        "concentration: which segment carries the Immediate Save load."
    )

with sright:
    L.kpi_row(
        [
            {
                "label": "CLV RMSE",
                "value": f"${clv_metrics['rmse']:,.2f}",
                "sub": "root mean squared error, held-out test",
                "tone": "cyan",
            },
            {
                "label": "CLV MAE",
                "value": f"${clv_metrics['mae']:,.2f}",
                "sub": "mean absolute error, held-out test",
                "tone": "cyan",
            },
            {
                "label": "CLV R²",
                "value": f"{clv_metrics['r2']:.4f}",
                "sub": "variance explained by the RF regressor",
                "tone": "green",
            },
        ],
        cols=3,
    )
    top = importance.head(6).iloc[::-1]  # biggest driver on top
    imp_fig = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color="#0EA5E9",
            text=[f"{v:.1%}" for v in top["importance"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    imp_fig.update_layout(
        height=250,
        margin=dict(l=4, r=56, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#A9B4C4"),
        xaxis=dict(visible=False),
        yaxis=dict(
            tickfont=dict(family="ui-monospace, Consolas, monospace", size=10.5,
                          color="#8590A2"),
            title=None,
        ),
        bargap=0.35,
        showlegend=False,
    )
    st.plotly_chart(imp_fig, width="stretch",
                    config={"displayModeBar": False})
    st.caption(
        "Random Forest Regressor on Total_Revenue (stage 3): tenure dominates "
        "predicted lifetime value — the same loyalty signal that protects "
        "against churn."
    )

st.divider()
st.caption(L.artifact_footer())

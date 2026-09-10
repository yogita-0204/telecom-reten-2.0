"""Telecom Customer Intelligence Platform (v2) — Streamlit dashboard entry.

TASKS.md #8 / PRD.md Section 6: exactly two pages — Executive Overview and
Customer Explorer — both pure readers of the pipeline artifacts under
``data/processed/`` and ``models/`` (lowercase artifact names per the
MEMORY.md task-1 gate). This file is the deployment entry point
(Streamlit Community Cloud: dashboard/app.py) and the app router: it
injects the shared "Obsidian Kinetic" stylesheet (the stitch design system
in stitch_stratum_analytics_ui_ux_system/ is the visual reference), brands
the sidebar, and declares the two pages via st.navigation so no third page
can appear by accident.

Why this module exists (business reason): a recruiter or interviewer opens
one URL and must immediately judge the project — churn risk, customer value
and segments in under a minute (PRD user story 1). Routing both views from
one entry keeps the deployed app's page list fixed and auditable: what
app.py declares is exactly what the public dashboard shows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make imports robust no matter where streamlit was launched from: the
# dashboard lives one level below the repo root, and both must be importable
# (dashboard/_lib.py holds the shared artifact layer).
_DASH_DIR = Path(__file__).resolve().parent
_ROOT = _DASH_DIR.parent
for _p in (str(_DASH_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _lib  # noqa: E402  (needs the sys.path bootstrap above)

st.set_page_config(
    page_title="Strata · Telecom Retention Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

_lib.inject_css()


# ---------------------------------------------------------------------------
# First-launch bootstrap (deployment safety net): regenerate artifacts when the
# checkout has none.
# ---------------------------------------------------------------------------
# Pipeline artifacts are committed (see .gitignore), so a deployed clone is
# ready instantly. This bootstrap only fires for a checkout that deliberately
# removed them — the same guarantee PRD.md Section 7 spells out for a clean
# clone ("pipeline runs top-to-bottom with zero errors"). Running the full
# pipeline inside a cold container takes minutes, which is exactly why the
# artifacts ship with the repo. The resource cache keeps it to once per app
# process when it does run.
@st.cache_resource(show_spinner="First launch: running the pipeline to generate artifacts...")
def _bootstrap_artifacts() -> None:
    """Generate every pipeline artifact from the committed raw CSV if missing.

    Business reason: the dashboard is a pure reader of pipeline outputs, so a
    missing artifact otherwise turns the pages into an error screen. Regenerating
    deterministically (seed 42) from the committed raw data keeps a clean
    checkout working even if the generated files were stripped.
    """
    missing = _lib.ensure_artifacts()
    if not missing:
        return
    from src.run_pipeline import main as run_pipeline_main

    run_pipeline_main()
    still_missing = _lib.ensure_artifacts()
    if still_missing:
        raise RuntimeError(
            f"Pipeline completed but artifacts still missing: {still_missing}"
        )


_bootstrap_artifacts()


# ---------------------------------------------------------------------------
# Sidebar brand block (before navigation so it sits above the page radio)
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div style="padding:2px 0 4px;border-bottom:1px solid rgba(255,255,255,.08);'
    'margin-bottom:10px;">'
    '<div style="display:flex;align-items:center;gap:8px;">'
    '<span style="font-family:var(--sk-mono);font-size:13px;font-weight:600;'
    'letter-spacing:.08em;color:var(--sk-text);">STRATA'
    '<span style="color:var(--sk-text3);font-weight:500;">&nbsp;INTELLIGENCE</span></span>'
    '<span class="sk-live" style="display:inline-block;"></span></div>'
    '<div style="font-family:var(--sk-mono);font-size:10px;letter-spacing:.08em;'
    'text-transform:uppercase;color:var(--sk-text3);margin-top:6px;">'
    "Telecom Customer Intelligence · v2</div></div>",
    unsafe_allow_html=True,
)

st.sidebar.caption("Models: Logistic Regression + Random Forest · CLV Random "
                   "Forest Regressor · K-Means k=4")

# ---------------------------------------------------------------------------
# Exactly two pages (PRD non-goal: no 3rd page). First page = default view.
# ---------------------------------------------------------------------------
_executive = st.Page(
    "pages/1_Executive_Overview.py",
    title="Executive Overview",
    url_path="overview",
    default=True,
)
_explorer = st.Page(
    "pages/2_Customer_Explorer.py",
    title="Customer Explorer",
    url_path="explorer",
)

_nav = st.navigation(
    {"Intelligence": [_executive], "Exploration": [_explorer]},
    position="sidebar",
)
_nav.run()

st.sidebar.markdown(
    '<div style="position:fixed;bottom:14px;font-family:var(--sk-mono);'
    'font-size:10px;letter-spacing:.08em;color:#4A5464;">'
    "PIPELINE v2 · DATA: MAVEN TELECOM 7,043 ACCOUNTS</div>",
    unsafe_allow_html=True,
)

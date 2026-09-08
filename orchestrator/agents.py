"""Agent prompt construction + planner output parsing.

Planner (gpt-5.6-luna): reads task state + memory, chooses ONE next task,
writes a precise implementation brief, and names the verification gate.
Executor (deepseek v4 flash): gets the brief + error history and does the work
with full tool access (--yolo). All conventions come from AGENTS.md, which the
executor is told to read first.
"""
from __future__ import annotations

import json
import re

PLANNER_PROMPT = """You are the PLANNER agent (main model) of a multi-agent orchestration
system building the "Telecom v2" project. The project spec lives in PRD.md,
build steps in TASKS.md, and the canonical conventions (stack, docstring rules,
random seed 42, verification command) in AGENTS.md. Read those files first.

You choose the SINGLE next task and write the brief the executor will follow.
Hard constraints to enforce on the executor:
- Stack ONLY: pandas, numpy, scikit-learn (LogisticRegression,
  RandomForestClassifier, RandomForestRegressor, KMeans, StandardScaler,
  train_test_split), shap (TreeExplainer only), joblib, fastapi, uvicorn,
  pydantic, streamlit, plotly, pytest.
- NO lifelines/survival, NO causal inference, NO XGBoost, NO SMOTE (use
  class_weight='balanced'), NO hyperparameter search, NO deep learning.
- Every train/transform function needs a business-reason docstring; every
  model-choice line needs a "why this and not fancier" comment.
- Feature engineering capped at 8-10 derived features. Random seed = 42.
- Artifacts: data/processed/Features.csv etc, models/*.pkl. Verification
  command: `python -m src.run_pipeline && python -m pytest tests/ -v`.

=== TASK STATE ===
{state}

=== MEMORY ===
{memory}

=== YOUR TASK ===
Pick the first PENDING task whose dependencies are all done. Return STRICT JSON:
{{
  "task_id": <int>,
  "task_label": "<short label>",
  "brief": "<3-8 sentence implementation brief: what to build, which files,
             what the artifact names must be, what the executor must verify>",
  "verify": "<the exact check the executor must run and make pass>",
  "risks": ["<1-3 concrete gotchas to warn the executor about>"]
}}
Reply with ONLY the JSON object in a ```json fence. Do not do the work yourself.
"""

EXECUTOR_PROMPT = """You are an EXECUTOR agent (worker) in a multi-agent orchestration
system building the "Telecom v2" project. You have full tool access (file
edits + shell). READ AGENTS.md, PRD.md and TASKS.md first — they are the
source of truth for conventions, stack and verification commands.

=== YOUR TASK (from the planner) ===
{task_section}

=== YOUR JOB ===
1. Implement the brief exactly, respecting the stack and conventions.
2. Run the stated verification yourself. If it fails, debug and fix, then
   re-run. Iterate until the verification passes.
3. Do NOT start other tasks or refactor unrelated code.
4. NEVER invoke cmdc / command-code / claude / other agent CLIs or LLM APIs —
   you have full tool access already; do the work directly.
5. When done, reply with a SHORT summary: what you built, files created,
   verification output (real numbers/artifacts), and any deviations.
"""
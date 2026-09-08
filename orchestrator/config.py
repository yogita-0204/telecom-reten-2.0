"""Orchestration system configuration.

Model routing (Command Code CLI as the provider):
- planner: gpt-5.6-luna   -> strategic brain: picks the next task, writes the brief
- memo:    MiniMax M2.5   -> memory agent: compresses cycle transcripts into memory.md
- executor: deepseek v4 flash -> hands-on worker: writes code, runs checks (--yolo)

All paths are relative to the repo root (this project's working directory).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "orchestrator" / "runtime"

# Model routes (must match the ids `cmdc --list-models` / the model catalog)
MODELS = {
    "planner": "gpt-5.6-luna",
    "memo": "MiniMaxAI/MiniMax-M2.5",
    "executor": "deepseek/deepseek-v4-flash",
}

# Per-role headless run settings
MAX_TURNS = {
    "planner": 40,
    "memo": 30,
    "executor": 150,
}
EXECUTOR_TIMEOUT_S = 60 * 15   # a heavy executor session (pipeline + model training)
PLANNER_TIMEOUT_S = 60 * 8
MEMO_TIMEOUT_S = 60 * 5
PROMPT_TIMEOUT_S = 60 * 2      # stdin write guard

# Loop control
DEFAULT_MAX_CYCLES = 40        # planner->dispatch->verify cycles before giving up
MAX_ATTEMPTS_PER_TASK = 3      # executor retries on the same brief
ESCALATIONS_PER_TASK = 2       # planner re-briefs after executor retries are exhausted
WORKERS = 1                    # parallel executor slots (1 = deterministic order)

VERIFY_TIMEOUT_S = 60 * 15

# Files the system manages
STATE_FILE = RUNTIME / "state.json"
MEMORY_FILE = RUNTIME / "memory.md"
TRANSCRIPT_FILE = RUNTIME / "transcript.jsonl"
LOG_FILE = RUNTIME / "orchestrator.log"

# The canonical gate from AGENTS.md (also the TASKS.md t6/t11 verify)
PIPELINE_VERIFY_CMD = "python -m src.run_pipeline && python -m pytest tests/ -v"
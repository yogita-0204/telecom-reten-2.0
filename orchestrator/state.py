"""Task registry + progress state for the orchestration loop.

The registry mirrors TASKS.md (id, label, section). State lives in
runtime/state.json:
    {"tasks": {<id>: {"status": "pending|in_progress|done|blocked",
                      "attempts": int, "escalations": int, "errors": [str]}}}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT, STATE_FILE


@dataclass
class Task:
    id: int
    label: str
    section: str
    depends_on: list[int] = field(default_factory=list)


# Mirrors TASKS.md sections, in execution order.
TASKS: list[Task] = [
    Task(0, "Setup: copy CSV into data/raw/, tool sign-ins", "0. Setup"),
    Task(1, "Data pipeline foundation: feature engineer + validation", "1. Data pipeline", [0]),
    Task(2, "Churn model: LR vs RF, metrics, SHAP, save pkl", "2. Churn model", [1]),
    Task(3, "CLV model: RandomForestRegressor, metrics, save pkl", "3. CLV model", [1]),
    Task(4, "Segmentation: KMeans k=4, cluster names, save pkl", "4. Segmentation", [1]),
    Task(5, "Priority matrix: 4 labels for all customers", "5. Priority matrix", [2, 3, 4]),
    Task(6, "Orchestration: run_pipeline end-to-end, pytest green", "6. Orchestration", [2, 3, 4, 5]),
    Task(7, "API: FastAPI /health and /predict, verify locally", "7. API", [6]),
    Task(8, "Dashboard: Streamlit 2 pages from stitch designs", "8. Dashboard", [6]),
    Task(9, "Deployment: GitHub push, Streamlit + Render live links", "9. Deployment", [7, 8]),
    Task(10, "Documentation: README with live links, results, rationale", "10. Docs", [9]),
    Task(11, "Final check: PRD Section 7 acceptance criteria", "11. Final check", [10]),
]

STATUS_ORDER = {"pending": 0, "in_progress": 1, "blocked": 2, "done": 3}


def _parse_tasks_md() -> list[Task]:
    """Best-effort: refresh labels from TASKS.md checkboxes (display only)."""
    md = ROOT / "TASKS.md"
    if not md.exists():
        return TASKS
    out: list[Task] = []
    for line in md.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*-\s*\[( |x)\]\s*(.+)", line)
        if m:
            out.append(m.group(2).strip())
    return TASKS


class TaskState:
    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        self.data: dict = {"tasks": {}}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.data = {"tasks": {}}
        self.data.setdefault("tasks", {})
        for t in TASKS:
            self.data["tasks"].setdefault(
                str(t.id),
                {"status": "pending", "attempts": 0, "escalations": 0, "errors": []},
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2), encoding="utf-8")

    # ---- helpers -----------------------------------------------------------
    def task(self, task_id: int) -> dict:
        return self.data["tasks"][str(task_id)]

    def status(self, task_id: int) -> str:
        return self.task(task_id)["status"]

    def set_status(self, task_id: int, status: str) -> None:
        self.task(task_id)["status"] = status
        self.save()

    def add_error(self, task_id: int, error: str) -> None:
        t = self.task(task_id)
        t["attempts"] += 1
        t["errors"].append(error[:2000])
        t["errors"] = t["errors"][-6:]
        self.save()

    def mark_done(self, task_id: int, note: str = "") -> None:
        t = self.task(task_id)
        t["status"] = "done"
        if note:
            t["last_note"] = note
        self.save()

    def pending(self) -> list[Task]:
        ready = []
        for t in TASKS:
            st = self.status(t.id)
            if st in ("done", "in_progress"):
                continue
            deps = [self.status(d) for d in t.depends_on]
            if all(d == "done" for d in deps):
                ready.append(t)
        return ready

    def done_ids(self) -> set[int]:
        return {int(k) for k, v in self.data["tasks"].items()
                if v.get("status") == "done"}

    def summary(self) -> str:
        lines = []
        for t in TASKS:
            st = self.status(t.id)
            mark = {"done": "x", "in_progress": "~", "blocked": "!", "pending": " "}[st]
            lines.append(f"- [{mark}] {t.id}: {t.label} ({st})")
        return "\n".join(lines)


if __name__ == "__main__":
    s = TaskState()
    print(s.summary())
    print("pending:", [t.id for t in s.pending()])
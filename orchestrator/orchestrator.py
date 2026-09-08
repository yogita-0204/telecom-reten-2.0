"""Orchestrator: the control loop of the multi-agent system.

Loop (until all TASKS.md items are done or the budget runs out):
  1. PLANNER (gpt-5.6-luna): read state+memory, choose next task, write brief.
  2. EXECUTOR (deepseek v4 flash): implement the brief with full tool access.
  3. VERIFIER: run the deterministic gate for that task.
  4. On failure: log the error (two-strikes rule in memory), retry the same
     brief, then let the planner re-brief (escalation), then mark blocked.
  5. MEMO (MiniMax M2.5): roll the transcript up into memory.md.
"""
from __future__ import annotations

import json
import re

from . import agents
from .config import (
    DEFAULT_MAX_CYCLES,
    ESCALATIONS_PER_TASK,
    MAX_ATTEMPTS_PER_TASK,
    ROOT,
)
from .memory import MemoryService
from .provider import CommandCodeProvider
from .state import TaskState
from .verifier import verify


def _extract_json(text: str) -> dict:
    """Tolerant extraction of the planner's JSON block."""
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blob = m.group(1) if m else text
    for candidate in (blob, text):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    # last resort: pull task_id out of anything
    m = re.search(r'"task_id"\s*:\s*(\d+)', text)
    if m:
        return {"task_id": int(m.group(1))}
    return {}


class Orchestrator:
    def __init__(self, provider: CommandCodeProvider | None = None,
                 state: TaskState | None = None,
                 max_cycles: int = DEFAULT_MAX_CYCLES,
                 verbose: bool = True):
        self.provider = provider or CommandCodeProvider(cwd=str(ROOT))
        self.state = state or TaskState()
        self.memory = MemoryService(self.provider, self.state)
        self.max_cycles = max_cycles
        self.verbose = verbose

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ---- planner ------------------------------------------------------------
    def _plan(self, task: object | None, failures: str = "") -> dict:
        ctx = self.state.summary()
        mem = self.memory.read()
        prompt = agents.PLANNER_PROMPT.format(state=ctx, memory=mem)
        if failures:
            prompt += (
                "\n\n=== ESCALATION CONTEXT ===\n"
                f"The executor failed on task {task} repeatedly. Errors:\n"
                f"{failures}\n"
                "Reconsider the approach: give a REVISED brief for the SAME "
                "task_id with a different strategy (different file layout, "
                "simpler checks, root-cause note). Keep the JSON shape.")
        self._say(f"  planner (gpt-5.6-luna) ...")
        res = self.provider.run(prompt, "planner")
        plan = _extract_json(res.text)
        self.memory.log({"kind": "planner", "exit": res.exit_code,
                         "duration_s": res.duration_s,
                         "plan": plan.get("task_id"), "msg": plan.get("task_label", "")})
        if not res.ok:
            self._say(f"    planner exited {res.exit_code} ({self.provider.explain(res.exit_code)}); "
                      f"continuing with fallback")
        if not plan or "task_id" not in plan:
            self._say("    planner returned no usable JSON; falling back to first pending task")
        return plan

    # ---- executor -----------------------------------------------------------
    def _execute(self, plan: dict, task_id: int) -> str:
        task_section = (
            f"TASK {task_id}: {plan.get('task_label', '')}\n"
            f"BRIEF: {plan.get('brief', '(no brief supplied - read TASKS.md and do the task)')}\n"
            f"VERIFY: {plan.get('verify', 'run the TASKS.md verification for this task')}\n"
            f"RISKS: {json.dumps(plan.get('risks', []))}"
        )
        self._say(f"  executor (deepseek-v4-flash) on task {task_id} ...")
        res = self.provider.run(agents.EXECUTOR_PROMPT.format(task_section=task_section),
                                "executor")
        self.memory.log({"kind": "executor", "task_id": task_id,
                         "exit": res.exit_code, "duration_s": res.duration_s,
                         "msg": res.text[-600:] or res.error[-400:]})
        if not res.ok:
            self._say(f"    executor exited {res.exit_code} ({self.provider.explain(res.exit_code)})")
        return res.text

    # ---- one task through the retry/escalation ladder ------------------------
    def _dispatch_task(self, task_id: int, plan: dict) -> bool:
        st = self.state
        st.set_status(task_id, "in_progress")
        failures = ""
        for attempt in range(1, MAX_ATTEMPTS_PER_TASK + 1):
            self._say(f"  attempt {attempt}/{MAX_ATTEMPTS_PER_TASK}")
            out = self._execute(plan, task_id)
            ok, detail = verify(task_id)
            self._say(f"  verify task {task_id}: {'PASS' if ok else 'FAIL'} - {detail[:200]}")
            self.memory.log({"kind": "verify", "task_id": task_id,
                             "ok": ok, "msg": detail})
            if ok:
                st.mark_done(task_id, note=detail)
                return True
            failures += f"\n--- attempt {attempt} ---\n{detail}\n{out[-1200:]}"
            st.add_error(task_id, detail)
        # escalation: planner re-briefs
        for _ in range(ESCALATIONS_PER_TASK):
            self._say("  escalating to planner for a revised brief")
            plan = self._plan(task_id, failures=failures[-4000:])
            if plan.get("task_id") not in (task_id, None):
                self._say(f"    planner switched task to {plan.get('task_id')}; ignoring, staying on {task_id}")
                plan = {"task_id": task_id, "task_label": st.task(task_id).get("_label", "retry"),
                        "brief": plan.get("brief", ""), "verify": plan.get("verify", "")}
            out = self._execute(plan, task_id)
            ok, detail = verify(task_id)
            self._say(f"  verify after escalation: {'PASS' if ok else 'FAIL'} - {detail[:200]}")
            self.memory.log({"kind": "verify", "task_id": task_id,
                             "ok": ok, "msg": f"escalated: {detail}"})
            if ok:
                st.mark_done(task_id, note=detail)
                return True
            failures += f"\n--- escalation ---\n{detail}\n{out[-1200:]}"
            st.add_error(task_id, "escalation: " + detail)
        st.set_status(task_id, "blocked")
        return False

    # ---- main loop -----------------------------------------------------------
    def run(self) -> int:
        self._say("=" * 70)
        self._say("ORCHESTRATION START: planner=gpt-5.6-luna "
                  "memo=MiniMax-M2.5 executor=deepseek-v4-flash")
        self._say("=" * 70)
        self.memory.log({"kind": "start", "msg": "orchestration session started"})
        for cycle in range(1, self.max_cycles + 1):
            pending = self.state.pending()
            if not pending:
                self._say("\nAll TASKS.md items done. Exiting loop.")
                self.memory.update_with_memo_agent()
                return 0
            self._say(f"\n--- cycle {cycle}/{self.max_cycles} ---")
            self._say("pending: " + ", ".join(str(t.id) for t in pending))
            plan = self._plan(pending[0])
            task_id = plan.get("task_id")
            if task_id not in {t.id for t in pending}:
                task_id = pending[0].id
                plan = {"task_id": task_id, "task_label": pending[0].label,
                        "brief": plan.get("brief", ""),
                        "verify": plan.get("verify", "")}
            self._say(f"planner chose task {task_id}")
            self._dispatch_task(task_id, plan)
            self.memory.update_with_memo_agent()
        # budget exhausted
        self._say("\nBUDGET EXHAUSTED. Remaining:")
        self._say(self.state.summary())
        self.memory.update_with_memo_agent()
        return 1


if __name__ == "__main__":
    raise SystemExit(Orchestrator().run())
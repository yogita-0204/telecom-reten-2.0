"""CLI entry: python -m orchestrator.main

Examples:
  python -m orchestrator.main            # run the loop until done/budget
  python -m orchestrator.main --check 6  # run verifier gates only
  python -m orchestrator.main --only 1,2 # loop over specific tasks
  python -m orchestrator.main --reset    # wipe runtime state
"""
from __future__ import annotations

import argparse
import shutil

from .config import RUNTIME
from .orchestrator import Orchestrator
from .state import TaskState
from .verifier import verify


def main() -> int:
    ap = argparse.ArgumentParser(description="Telecom v2 multi-agent orchestrator")
    ap.add_argument("--check", nargs="*", type=int, default=None,
                    help="run verifier gates for the given tasks (default: task 6)")
    ap.add_argument("--check-all", action="store_true", help="run every verifier gate")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated task ids the loop may work on")
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument("--reset", action="store_true", help="clear runtime state")
    ap.add_argument("--state", action="store_true", help="print task state and exit")
    args = ap.parse_args()

    if args.reset:
        shutil.rmtree(RUNTIME, ignore_errors=True)
        print("runtime state cleared")
        return 0

    if args.state:
        TaskState().load()
        print(TaskState().summary())
        return 0

    if args.check is not None or args.check_all:
        ids = list(range(12)) if args.check_all else (args.check or [6])
        fails = 0
        for tid in ids:
            ok, detail = verify(tid)
            print(f"task {tid}: {'PASS' if ok else 'FAIL'} — {detail[:400]}")
            fails += 0 if ok else 1
        return 0 if fails == 0 else 1

    only = {int(x) for x in args.only.split(",") if x.strip()}
    state = TaskState()
    if only:
        for tid in only:
            if state.status(tid) != "done":
                state.set_status(tid, "pending")
        # re-open dependents
        from .state import TASKS
        for t in TASKS:
            if t.id in only or any(d in only for d in t.depends_on):
                if state.status(t.id) != "done":
                    state.set_status(t.id, "pending")

    orch = Orchestrator(state=state, max_cycles=args.max_cycles or 40)
    return orch.run()


if __name__ == "__main__":
    raise SystemExit(main())
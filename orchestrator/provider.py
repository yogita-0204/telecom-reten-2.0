"""CommandCodeProvider: talks to the Command Code CLI (`cmdc`) in headless mode.

Command Code IS the provider for every agent in the system:
  - planner runs on gpt-5.6-luna (read-only, no --yolo)
  - memo runs on MiniMax M2.5 (read-only)
  - executor runs on deepseek v4 flash with --yolo so it can edit files
    and run shell commands, exactly like an interactive session would.

Prompts are sent over stdin (the CLI auto-detects piped input when no query
argument is given), which avoids Windows quoting issues with multi-line text.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    EXECUTOR_TIMEOUT_S,
    MEMO_TIMEOUT_S,
    MODELS,
    PLANNER_TIMEOUT_S,
    MAX_TURNS,
)

# cmdc exit codes (from the headless docs)
EXIT_SUCCESS = 0
EXIT_GENERAL = 1
EXIT_AUTH = 3
EXIT_PERMISSION = 4
EXIT_RATE_LIMIT = 5
EXIT_CONNECTION = 6
EXIT_SERVER = 7
EXIT_MAX_TURNS = 8
EXIT_NO_RESPONSE = 9
EXIT_CREDITS = 10

TIMEOUTS = {
    "planner": PLANNER_TIMEOUT_S,
    "memo": MEMO_TIMEOUT_S,
    "executor": EXECUTOR_TIMEOUT_S,
}


@dataclass
class AgentResult:
    text: str = ""
    exit_code: int = -1
    duration_s: float = 0.0
    raw: str = ""
    error: str = ""
    ok: bool = False


@dataclass
class CommandCodeProvider:
    cmd: str = "cmdc"
    cwd: str | None = None
    _bin: str | None = field(default=None, init=False)

    def _resolve(self) -> list[str]:
        """Return an argv prefix that runs the Command Code CLI.

        On Windows the npm shim is a .cmd batch file (not directly runnable
        without cmd.exe quoting pain), so resolve it to `node <cli.js>` and
        call node directly. Elsewhere, use the `cmdc` binary.
        """
        if self._bin:
            return list(self._bin)
        if os.name == "nt":
            import shutil
            node = shutil.which("node.exe") or shutil.which("node")
            shim = shutil.which(self.cmd) or shutil.which("cmdc.cmd") \
                or shutil.which("cmdc")
            if shim and shim.lower().endswith((".cmd", ".bat", ".ps1")):
                entry = Path(shim).parent / "node_modules" / "command-code" / "dist" / "index.mjs"
                if node and entry.exists():
                    self._bin = [node, str(entry)]
                    return list(self._bin)
        self._bin = [shutil_which(self.cmd) or self.cmd]
        return list(self._bin)

    def run(self, prompt: str, role: str, *, turns: int | None = None,
            timeout: float | None = None) -> AgentResult:
        """Run one headless agent call for a role (planner/memo/executor)."""
        model = MODELS[role]
        turns = turns or MAX_TURNS[role]
        timeout = timeout or TIMEOUTS[role]
        argv = list(self._resolve())
        argv += ["-p", "--model", model, "--skip-onboarding", "--no-auto-update",
                 "--max-turns", str(turns), "--output-format", "text"]
        if role == "executor":
            argv.append("--yolo")  # file writes + shell commands for the worker
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                input=prompt.encode("utf-8"),
                capture_output=True,
                cwd=self.cwd,
                timeout=timeout,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            return AgentResult(
                text=stdout.strip(),
                exit_code=proc.returncode,
                duration_s=round(time.time() - started, 1),
                raw=stdout,
                error=stderr[-4000:],
                ok=proc.returncode == EXIT_SUCCESS,
            )
        except subprocess.TimeoutExpired as ex:
            return AgentResult(
                text="",
                exit_code=-2,
                duration_s=round(time.time() - started, 1),
                error=f"timeout after {timeout}s",
                ok=False,
            )
        except Exception as ex:  # noqa: BLE001 - provider must never crash the loop
            return AgentResult(
                text="",
                exit_code=-3,
                duration_s=round(time.time() - started, 1),
                error=f"spawn failed: {ex}",
                ok=False,
            )

    def explain(self, code: int) -> str:
        return {
            EXIT_SUCCESS: "success",
            EXIT_GENERAL: "general error",
            EXIT_AUTH: "not authenticated",
            EXIT_PERMISSION: "permission denied",
            EXIT_RATE_LIMIT: "rate limited",
            EXIT_CONNECTION: "network failure",
            EXIT_SERVER: "api server error",
            EXIT_MAX_TURNS: "max turns reached",
            EXIT_NO_RESPONSE: "no response",
            EXIT_CREDITS: "insufficient credits",
            -2: "timeout",
            -3: "spawn failed",
        }.get(code, f"unknown ({code})")


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


if __name__ == "__main__":  # provider self-test: `python -m orchestrator.provider`
    p = CommandCodeProvider(cwd=os.getcwd())
    print("resolved argv:", p._resolve())
    for role in ("planner", "executor", "memo"):
        r = p.run(
            "Reply with EXACTLY one word: OK",
            role,
            turns=8,
            timeout=120,
        )
        print(f"[{role}] exit={r.exit_code} ({p.explain(r.exit_code)}) "
              f"{r.duration_s}s :: {r.text[:80]!r}")
        if r.error.strip():
            print("   stderr:", r.error.strip()[:300])
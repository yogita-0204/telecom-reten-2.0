"""MemoryService: persistent session memory for the orchestration loop.

Two layers:
1. transcript.jsonl - append-only machine log of every cycle (planner brief,
   executor outcome, verifier result, errors).
2. memory.md - a rolling, human-readable summary maintained by the MEMO
   agent (MiniMax M2.5). The memo agent is a pure summarizer: it READS the
   transcript + current memory.md and RETURNS the updated markdown, which the
   orchestrator writes back. This keeps the memory model stateless and cheap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import LOG_FILE, MEMORY_FILE, TRANSCRIPT_FILE
from .provider import CommandCodeProvider
from .state import TaskState

MEMO_PROMPT = """You are the MEMORY agent of a multi-agent build orchestration system for
the "Telecom v2" project (churn prediction + CLV + segmentation + retention priority,
see AGENTS.md). Your job: maintain the team's long-term memory file.

Below is the CURRENT memory file, followed by the latest cycle transcript
(JSON lines) and the task state.

Rules:
- Output ONLY the updated memory.md content as a single markdown code block (```markdown ... ```).
- Keep the file under ~60 lines. Preserve prior knowledge, fold in new lessons.
- Structure: "## Status" (which tasks done/in progress), "## Decisions"
  (architecture decisions made), "## Lessons" (errors that hit twice = rules,
  one-offs = observations), "## Next" (what the planner should do next).
- If the transcript shows an error that already appears in memory, mark it as a
  confirmed rule; if it's new, add it as an observation.
- Never invent facts. If a transcript mentions "Verify OK", state it as done.
- Reply with the code block only. No commentary.

=== CURRENT MEMORY FILE ===
{memory}

=== LATEST CYCLE TRANSCRIPT ===
{transcript}

=== TASK STATE ===
{state}
"""


class MemoryService:
    def __init__(self, provider: CommandCodeProvider, state: TaskState,
                 root: Path | None = None):
        self.provider = provider
        self.state = state
        self.root = root

    # ---- transcript ---------------------------------------------------------
    def log(self, event: dict) -> None:
        TRANSCRIPT_FILE.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
        with TRANSCRIPT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"[{rec['ts']}] {rec.get('kind', '?')} "
                     f"{str(rec.get('task_id', ''))}: "
                     f"{str(rec.get('msg', ''))[:300]}\n")

    # ---- memory file --------------------------------------------------------
    def read(self) -> str:
        if MEMORY_FILE.exists():
            return MEMORY_FILE.read_text(encoding="utf-8")
        return "(fresh memory — no prior session)"

    def update_with_memo_agent(self, since_ts: str | None = None) -> str:
        """Ask MiniMax M2.5 to summarize recent transcript into memory.md."""
        transcript = self._recent_transcript(since_ts)
        prompt = MEMO_PROMPT.format(
            memory=self.read(), transcript=transcript, state=self.state.summary())
        result = self.provider.run(prompt, "memo", turns=30)
        new_memory = extract_markdown_block(result.text)
        if new_memory and len(new_memory) > 20:
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            MEMORY_FILE.write_text(new_memory, encoding="utf-8")
            self.log({"kind": "memo", "msg": "memory rollup OK",
                      "exit": result.exit_code,
                      "note": f"{len(new_memory)} chars, {result.duration_s}s"})
            return new_memory
        self.log({"kind": "memo", "msg": f"memory rollup failed: {result.error[:200]}"})
        return self.read()

    def _recent_transcript(self, since_ts: str | None) -> str:
        if not TRANSCRIPT_FILE.exists():
            return "(no transcript yet)"
        lines = TRANSCRIPT_FILE.read_text(encoding="utf-8").splitlines()
        if since_ts:
            lines = [ln for ln in lines if json.loads(ln).get("ts", "") >= since_ts]
        # keep it bounded - last 40 events
        return "\n".join(lines[-40:])


def extract_markdown_block(text: str) -> str:
    """Pull the ```markdown ... ``` block out of the memo agent's reply."""
    if not text:
        return ""
    start = text.find("```markdown")
    if start == -1:
        start = text.find("```")
    if start == -1:
        return text.strip()
    start = text.find("\n", start) + 1
    end = text.rfind("```")
    if end == -1 or end <= start:
        return text[start:].strip()
    return text[start:end].strip()
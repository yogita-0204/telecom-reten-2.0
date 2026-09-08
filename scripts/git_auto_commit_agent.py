"""
Git Auto-Commit and Push Loop Agent
-----------------------------------
Continuous background agent that monitors the telecom project workspace for file changes,
generates clean, human-aligned commit messages based on modified features, and pushes
updates directly to GitHub origin/main.
"""

import os
import sys
import time
import subprocess
import datetime
from pathlib import Path

# Working directory root
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_ROOT / "orchestrator" / "git_agent.log"


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [GitAgent] {msg}"
    print(formatted, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


def run_cmd(cmd, cwd=REPO_ROOT):
    """Executes a shell command and returns (returncode, stdout, stderr)."""
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=isinstance(cmd, str),
    )
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()


def get_changed_files():
    """Returns list of changed, added, or untracked files relative to REPO_ROOT."""
    # First stage all tracked changes and check untracked
    code, out, _ = run_cmd(["git", "status", "--porcelain"])
    if code != 0 or not out:
        return []

    lines = out.splitlines()
    files = []
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            files.append(parts[1].strip('"'))
    return files


def generate_commit_message(changed_files):
    """
    Generates an intuitive, professional, human-aligned commit message
    based on the specific files and features modified.
    """
    if not changed_files:
        return "chore: update project files"

    # Specific single-feature matches
    if all("dashboard" in f or ".streamlit" in f for f in changed_files):
        if any("pages" in f for f in changed_files):
            return "feat: refine multi-page executive dashboard and customer views"
        return "feat: update Streamlit dashboard visuals and components"

    if all("api" in f for f in changed_files):
        return "feat: update FastAPI prediction routes and schema models"

    if all("churn" in f for f in changed_files):
        return "feat: enhance churn prediction model logic and metrics"

    if all("clv" in f for f in changed_files):
        return "feat: refine CLV regression model and feature importance"

    if all("segment" in f for f in changed_files):
        return "feat: update customer segmentation and centroid profiles"

    if all("priority" in f for f in changed_files):
        return "feat: update retention priority matrix decision rules"

    if all("feature_engineer" in f for f in changed_files):
        return "feat: update data feature transformations and validations"

    if all("test" in f for f in changed_files):
        return "test: update test suite assertions and coverage"

    if all(f.endswith(".md") for f in changed_files):
        if any("README" in f for f in changed_files):
            return "docs: update README with project deliverables and live links"
        if any("TASKS" in f for f in changed_files):
            return "chore: update tasks completion progress in TASKS.md"
        return "docs: update project documentation specifications"

    if any("README.md" in f for f in changed_files) and len(changed_files) == 1:
        return "docs: update README with project overview and execution results"

    # Multi-area detection
    areas = set()
    for f in changed_files:
        if "api" in f:
            areas.add("API")
        elif "dashboard" in f:
            areas.add("dashboard")
        elif "src" in f:
            areas.add("models")
        elif "tests" in f:
            areas.add("tests")
        elif "data" in f:
            areas.add("data")
        elif f.endswith(".md"):
            areas.add("docs")
        elif "orchestrator" in f:
            areas.add("orchestrator")

    if areas:
        area_str = ", ".join(sorted(areas))
        return f"feat: update {area_str} components and project artifacts"

    return "chore: update workspace files and runtime state"


def commit_and_push(changed_files):
    """Stages files, creates commit, and pushes to origin/main."""
    log(f"Detected {len(changed_files)} changed file(s): {', '.join(changed_files[:5])}{'...' if len(changed_files) > 5 else ''}")

    # Stage changes
    code, _, err = run_cmd(["git", "add", "-A"])
    if code != 0:
        log(f"Error staging files: {err}")
        return False

    msg = generate_commit_message(changed_files)
    log(f"Crafted commit message: '{msg}'")

    # Commit
    code, out, err = run_cmd(["git", "commit", "-m", msg])
    if code != 0:
        log(f"Commit skipped or failed: {err or out}")
        return False

    log(f"Commit successful: {out.splitlines()[0] if out else 'OK'}")

    # Push to origin main
    log("Pushing to GitHub origin/main...")
    push_code, push_out, push_err = run_cmd(["git", "push", "origin", "main"])
    if push_code == 0:
        log("Successfully pushed to GitHub!")
        return True
    else:
        log(f"Push warning/error: {push_err or push_out}. Will retry on next cycle.")
        return False


def run_loop(poll_interval_seconds=15):
    """Main watcher loop that polls for changes until manually terminated."""
    log(f"Starting Git Auto-Commit & Push Agent (polling every {poll_interval_seconds}s)")
    log(f"Monitoring repository at: {REPO_ROOT}")
    log(f"Remote: https://github.com/yogita-0204/telecom-reten-2.0.git (branch: main)")

    try:
        while True:
            changed_files = get_changed_files()
            if changed_files:
                commit_and_push(changed_files)
            time.sleep(poll_interval_seconds)
    except KeyboardInterrupt:
        log("Agent stopped by user interrupt.")
    except Exception as e:
        log(f"Unexpected error in agent loop: {e}")


if __name__ == "__main__":
    poll_sec = 15
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        poll_sec = int(sys.argv[1])
    run_loop(poll_sec)

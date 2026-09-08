"""Verifier: deterministic per-task acceptance gates.

Each gate runs locally (shell/python) and returns (ok: bool, detail: str).
These encode the "Verify:" lines from TASKS.md. The orchestrator trusts these
checks over the executor's self-reported success.
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .config import ROOT, VERIFY_TIMEOUT_S

PY = sys.executable
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def _run(cmd: str, timeout: int = VERIFY_TIMEOUT_S) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=ROOT, timeout=timeout,
                           encoding="utf-8", errors="replace")
        tail = (p.stdout or "")[-2500:] + ("\n" + (p.stderr or "")[-2500:])
        return p.returncode, tail.strip()
    except subprocess.TimeoutExpired:
        return -1, "verification timed out"


def _has(path: str) -> bool:
    return (ROOT / path).exists()


def _read_csv_rows(path: str) -> int | None:
    try:
        import pandas as pd
        return len(pd.read_csv(path))
    except Exception:
        return None


VERIFIERS: dict[int, callable] = {}


def gate(func):
    VERIFIERS[func.__name__.removeprefix("task_")] = func
    return func


@gate
def task_0() -> tuple[bool, str]:
    ok = _has("data/raw/telecom_customer_churn.csv")
    return ok, "data/raw CSV present" if ok else "data/raw/telecom_customer_churn.csv missing"


@gate
def task_1() -> tuple[bool, str]:
    if not _has("data/processed/features.csv"):
        return False, "data/processed/features.csv missing"
    import pandas as pd
    try:
        df = pd.read_csv(ROOT / "data/processed/features.csv")
        nulls = int(df.isnull().sum().sum())
        ok = len(df) == 7043 and nulls == 0
        return ok, f"features.csv: {len(df)} rows, {nulls} nulls"
    except Exception as ex:
        return False, f"features.csv unreadable: {ex}"


@gate
def task_2() -> tuple[bool, str]:
    checks = [
        _has("models/churn_model.pkl"),
        _has("models/segment_encoder_prep.pkl"),  # tolerated if absent
    ]
    # metrics comparison table (name per convention; accept a few)
    cands = ["data/processed/churn_model_comparison.csv",
             "data/processed/churn_comparison.csv",
             "models/churn_model_comparison.csv"]
    table = next((c for c in cands if _has(c)), None)
    has_shap = any(_has(p) for p in (
        "data/processed/shap_summary.png", "models/shap_summary.png",
        "data/processed/shap_global_summary.png"))
    if not table:
        return False, "churn comparison table csv missing"
    import pandas as pd
    try:
        df = pd.read_csv(ROOT / table)
        numeric = df.select_dtypes("number").drop(columns=["Unnamed: 0"], errors="ignore")
        bad = numeric.isnull().any().any() or (numeric.abs().sum().sum() == 0)
    except Exception:
        bad = True
    missing = [c for c in ("models/churn_model.pkl",) if not _has(c)]
    if missing:
        return False, f"missing: {missing}"
    if bad:
        return False, "comparison table has NaN/zero values"
    return True, f"churn done; shap={'yes' if has_shap else 'NO (check)'}"


@gate
def task_3() -> tuple[bool, str]:
    if not _has("models/clv_model.pkl"):
        return False, "models/clv_model.pkl missing"
    cands = ["data/processed/clv_metrics.csv", "models/clv_metrics.csv",
             "data/processed/clv_model_metrics.csv"]
    table = next((c for c in cands if _has(c)), None)
    if not table:
        return False, "clv metrics csv missing"
    import pandas as pd
    try:
        df = pd.read_csv(ROOT / table)
        vals = df.select_dtypes("number").to_numpy().ravel()
        bad = len(vals) == 0 or any(v != v for v in vals)  # NaN check
    except Exception:
        bad = True
    return (not bad), "clv pkl + metrics OK" if not bad else "clv metrics NaN/empty"


@gate
def task_4() -> tuple[bool, str]:
    if not _has("models/segmentation_model.pkl"):
        return False, "models/segmentation_model.pkl missing"
    cands = ["data/processed/segment_profiles.csv", "data/processed/segments.csv",
             "data/processed/cluster_profiles.csv"]
    table = next((c for c in cands if _has(c)), None)
    if not table:
        return False, "segment profile csv missing"
    import pandas as pd
    try:
        df = pd.read_csv(ROOT / table)
        n = len(df)
    except Exception:
        n = 0
    return n == 4, f"segment profiles: {n} clusters (need 4)"


@gate
def task_5() -> tuple[bool, str]:
    cands = ["data/processed/prioritized_customers.csv",
             "data/processed/customers_with_priority.csv",
             "data/processed/priority_matrix.csv"]
    table = next((c for c in cands if _has(c)), None)
    if not table:
        return False, "priority csv missing"
    import pandas as pd
    try:
        df = pd.read_csv(ROOT / table)
        col = next((c for c in df.columns if "priorit" in c.lower()), None)
        if col is None:
            return False, "no priority label column"
        n = len(df)
        nulls = int(df[col].isnull().sum())
        ok = n == 7043 and nulls == 0
        return ok, f"priority: {n} rows, {nulls} null labels"
    except Exception as ex:
        return False, f"priority csv unreadable: {ex}"


@gate
def task_6() -> tuple[bool, str]:
    code, out = _run("python -m src.run_pipeline && python -m pytest tests/ -v")
    return code == 0, "pipeline+pytest OK" if code == 0 else f"FAILED:\n{out[:1500]}"


@gate
def task_7() -> tuple[bool, str]:
    if not _has("api/main.py"):
        return False, "api/main.py missing"
    code, out = _run("python -c \"from api.main import app; print('import ok')\"")
    if code != 0:
        return False, f"api import failed:\n{out[:800]}"
    # live /health smoke test
    import subprocess as sp
    srv = sp.Popen([PY, "-m", "uvicorn", "api.main:app", "--port", "8123",
                    "--log-level", "warning"], cwd=ROOT,
                   stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    try:
        for _ in range(30):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8123/health",
                                            timeout=3) as r:
                    return r.status == 200, f"/health -> {r.status} {r.read()[:80]!r}"
            except Exception:
                continue
        return False, "/health never responded"
    finally:
        srv.terminate()


@gate
def task_8() -> tuple[bool, str]:
    if not _has("dashboard/app.py"):
        return False, "dashboard/app.py missing"
    code, out = _run("python -m py_compile dashboard/app.py")
    if code != 0:
        return False, f"dashboard compile failed:\n{out[:800]}"
    src = (ROOT / "dashboard/app.py").read_text(encoding="utf-8", errors="replace")
    if "st.set_page_config" not in src:
        return False, "missing st.set_page_config"
    # smoke-run streamlit headless for ~25s, then kill; must stay alive
    import subprocess as sp
    srv = sp.Popen([PY, "-m", "streamlit", "run", "dashboard/app.py",
                    "--server.headless", "true", "--server.port", "8124",
                    "--browser.gatherUsageStats", "false"], cwd=ROOT,
                   stdout=sp.PIPE, stderr=sp.STDOUT)
    out_buf = b""
    try:
        for _ in range(25):
            time.sleep(1)
            if srv.poll() is not None:
                return False, f"streamlit exited early:\n{out_buf[-600:].decode(errors='replace')}"
            try:
                with urllib.request.urlopen("http://127.0.0.1:8124", timeout=3) as r:
                    if r.status == 200:
                        return True, "streamlit serves / (HTTP 200)"
            except Exception:
                pass
        return False, "streamlit served no HTTP response"
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:
            srv.kill()


@gate
def task_9() -> tuple[bool, str]:
    code, out = _run("git remote -v", timeout=60)
    has_remote = "github.com" in out
    readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace") \
        if _has("README.md") else ""
    has_links = ("http" in readme and ("streamlit.app" in readme
                                       or "render.com" in readme))
    if not has_remote:
        return False, "no GitHub remote configured (task needs user creds)"
    if not has_links:
        return False, "README has no live deployment links yet"
    return True, "remote + live links present"


@gate
def task_10() -> tuple[bool, str]:
    if not _has("README.md"):
        return False, "README.md missing"
    readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
    return ("http" in readme), "README present with links" if "http" in readme \
        else "README present but no links"


@gate
def task_11() -> tuple[bool, str]:
    code, out = _run("python -m src.run_pipeline && python -m pytest tests/ -v")
    ok = code == 0
    return ok, "final gate: pipeline+pytest OK" if ok else f"FAILED:\n{out[:1500]}"


def verify(task_id: int) -> tuple[bool, str]:
    fn = VERIFIERS.get(str(task_id))
    if fn is None:
        return False, f"no verifier for task {task_id}"
    return fn()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("tasks", nargs="*", type=int)
    args = ap.parse_args()
    ids = args.tasks if args.tasks else (list(VERIFIERS) if args.all else [6])
    for tid in ids:
        ok, detail = verify(tid)
        print(f"task {tid}: {'PASS' if ok else 'FAIL'} — {detail[:400]}")
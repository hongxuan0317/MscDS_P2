"""Prove the deployable app runs from an isolated directory with no P2 parent."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNTIME_ITEMS = [
    "app.py", "config.py", "data_loader.py", "utils.py", "pages", "data",
    "artifacts", "assets", ".streamlit", "requirements.txt",
]

checks = []


def add(name: str, passed: bool, evidence: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})


with tempfile.TemporaryDirectory(prefix="fps_dashboard_standalone_") as temporary:
    target = Path(temporary) / "dashboard"
    target.mkdir()
    for item in RUNTIME_ITEMS:
        source = ROOT / item
        destination = target / item
        if source.is_dir():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, destination)

    add("No parent P2 directory", not (target.parent / "P2").exists(), str(target.parent))
    add("No environment copied", not (target / ".venv").exists(), ".venv absent")
    add("No Python caches copied", not list(target.rglob("__pycache__")), "__pycache__ absent")

    runtime_source = "\n".join(
        path.read_text(encoding="utf-8")
        for pattern in ("*.py", "*.toml") for path in target.rglob(pattern)
    )
    forbidden_paths = ["C:\\Users\\", "P2_OUTPUT_DIR", "P2_DATA_DIR", "P2_DIR /", "PROJECT_DIR / 'P2'"]
    found_paths = [token for token in forbidden_paths if token in runtime_source]
    add("No absolute or parent-P2 runtime paths", not found_paths, f"found={found_paths}")

    incident = pd.read_csv(target / "data" / "incident_modelling_anonymized.csv")
    source_incidents = pd.read_csv(target / "data" / "source_incidents.csv")
    forbidden_person_fields = {
        "id", "name", "first_name", "last_name", "address", "city", "latitude", "longitude",
        "age", "age_group", "gender", "race", "was_mental_illness_related",
    }
    incident_found = sorted(forbidden_person_fields.intersection(incident.columns))
    source_found = sorted(forbidden_person_fields.intersection(source_incidents.columns))
    add("No row-level identifying or sensitive fields", not incident_found and not source_found, f"incident={incident_found}; source={source_found}")

    add(
        "Local model artefacts present",
        (target / "artifacts" / "incident_winner_primary_KMeans.joblib").exists()
        and (target / "artifacts" / "incident_preprocessor_primary.joblib").exists(),
        "model and preprocessor packaged",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    page_test = r'''
from pathlib import Path
from streamlit.testing.v1 import AppTest
root = Path.cwd()
for page in sorted((root / "pages").glob("*.py")):
    test = AppTest.from_file(str(page), default_timeout=60).run()
    if len(test.exception):
        raise RuntimeError(f"{page.name}: {[item.message for item in test.exception]}")
print("five isolated pages passed")
'''
    page_result = subprocess.run(
        [sys.executable, "-c", page_test], cwd=target, env=environment,
        capture_output=True, text=True, timeout=180,
    )
    add("All five isolated pages render", page_result.returncode == 0, page_result.stdout.strip() or page_result.stderr[-500:])

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8503",
            "--server.headless", "true", "--browser.gatherUsageStats", "false",
        ],
        cwd=target, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    healthy = False
    try:
        for _ in range(40):
            time.sleep(0.25)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8503/_stcore/health", timeout=2) as response:
                    healthy = response.status == 200 and response.read().decode().strip() == "ok"
                    if healthy:
                        break
            except Exception:
                pass
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    add("Isolated Streamlit server health", healthy, "HTTP 200 ok" if healthy else "health check failed")

results = pd.DataFrame(checks)
results.to_csv(ROOT / "STANDALONE_DEPLOYMENT_CHECKS.csv", index=False)
summary = {
    "passed": int((results["status"] == "PASS").sum()),
    "failed": int((results["status"] == "FAIL").sum()),
}
(ROOT / "STANDALONE_DEPLOYMENT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(results.to_string(index=False))
print(json.dumps(summary, indent=2))
if summary["failed"]:
    raise SystemExit(1)

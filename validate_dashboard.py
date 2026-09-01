"""Run final, read-only dashboard integrity and acceptance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from config import (
    DASHBOARD_DIR,
    DATA_DIR,
    EXPECTED,
    INCIDENT_MODEL_PATH,
    INCIDENT_PREPROCESSOR_PATH,
)

PROJECT_DIR = DASHBOARD_DIR.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


checks: list[dict[str, str]] = []


def check(name: str, condition: bool, evidence: str) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})


validation = json.loads((DATA_DIR / "validation_summary.json").read_text(encoding="utf-8"))
for key in [
    "source_incidents", "incident_modelling", "incident_clusters", "counties",
    "zero_fps_counties", "county_clusters", "spatial_regions", "fps_positive_counties",
]:
    check(f"Validated {key}", validation[key] == EXPECTED[key], f"observed={validation[key]}; expected={EXPECTED[key]}")

incidents = pd.read_csv(DATA_DIR / "incident_modelling_anonymized.csv")
counties = pd.read_csv(DATA_DIR / "counties.csv", dtype={"county_fips": str})
check("Anonymized incident display IDs unique", incidents["display_record"].is_unique, f"rows={len(incidents):,}")
check("No direct incident identifiers copied", "id" not in incidents.columns, ", ".join(incidents.columns))
row_level_sensitive = {"age", "age_group", "gender", "race", "was_mental_illness_related"}
found_sensitive = sorted(row_level_sensitive.intersection(incidents.columns))
check("No row-level sensitive deployment fields", not found_sensitive, f"found={found_sensitive}")
check("County FIPS unique", counties["county_fips"].is_unique, f"rows={len(counties):,}")
check("Zero-FPS counties retained", int((counties["fps_count_2015_2024"] == 0).sum()) == 1_357, "zero-FPS=1,357")

geojson = json.loads((DATA_DIR / "counties.geojson").read_text(encoding="utf-8"))
geo_ids = [feature["properties"]["county_fips"] for feature in geojson["features"]]
check("County geometry coverage", len(geo_ids) == 3_143 and len(set(geo_ids)) == 3_143, f"features={len(geo_ids):,}; unique={len(set(geo_ids)):,}")

model = joblib.load(INCIDENT_MODEL_PATH)
preprocessor = joblib.load(INCIDENT_PREPROCESSOR_PATH)
check("Saved incident model loaded", model.cluster_centers_.shape == (4, 52), f"centroids={model.cluster_centers_.shape}")
check("Saved preprocessing metadata loaded", {"category_levels", "numeric_scaler", "time_scaler"}.issubset(preprocessor), f"keys={sorted(preprocessor)}")

before_sources = pd.read_csv(DATA_DIR / "locked_p2_source_hashes_before.csv")
before_outputs = pd.read_csv(DATA_DIR / "locked_p2_output_hashes_before.csv")
current_rows = []
for _, row in pd.concat([before_sources, before_outputs], ignore_index=True).iterrows():
    path = PROJECT_DIR / row["relative_path"]
    current_rows.append(path.exists() and sha256(path) == row["sha256"])
check("Locked P2 notebook/script hashes unchanged", all(current_rows[:len(before_sources)]), f"files={len(before_sources)}")
check("Locked P2 output hashes unchanged", all(current_rows[len(before_sources):]), f"files={len(before_outputs)}")

pages = sorted((DASHBOARD_DIR / "pages").glob("*.py"))
application_files = [DASHBOARD_DIR / "app.py", DASHBOARD_DIR / "data_loader.py", DASHBOARD_DIR / "utils.py", *pages]
source = "\n".join(path.read_text(encoding="utf-8") for path in application_files)
prohibited = ["KMeans.fit(", "AgglomerativeClustering.fit(", "HDBSCAN.fit(", ".fit_predict(", "Skater("]
found = [token for token in prohibited if token in source]
check("No model fitting in launch-time application", not found, f"prohibited tokens found={found}")
check("Exactly five dashboard pages", len(pages) == 5, f"pages={len(pages)}")

required_files = [
    "app.py", "data_loader.py", "utils.py", "requirements.txt", "README.md",
    "run_dashboard.bat", "run_dashboard.sh", ".streamlit/config.toml",
    "DASHBOARD_BUILD_REPORT.md",
    "data/incident_pca_coordinates.csv", "data/counties.geojson",
    "data/context_reference_values.csv", "data/incident_feature_scope.csv",
    "artifacts/incident_winner_primary_KMeans.joblib",
    "artifacts/incident_preprocessor_primary.joblib", "DEPLOYMENT_MANIFEST.md",
]
missing = [name for name in required_files if not (DASHBOARD_DIR / name).exists()]
check("Required dashboard files present", not missing, f"missing={missing}")

result = pd.DataFrame(checks)
result.to_csv(DASHBOARD_DIR / "DASHBOARD_ACCEPTANCE_CHECKS.csv", index=False)
summary = {"passed": int((result["status"] == "PASS").sum()), "failed": int((result["status"] == "FAIL").sum())}
(DASHBOARD_DIR / "DASHBOARD_ACCEPTANCE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(result.to_string(index=False))
print(json.dumps(summary, indent=2))
if summary["failed"]:
    raise SystemExit(1)

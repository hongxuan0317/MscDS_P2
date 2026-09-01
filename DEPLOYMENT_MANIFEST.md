# Streamlit Community Cloud deployment manifest

Use the contents of `dashboard/` as the GitHub repository root, or configure Streamlit Community Cloud to launch `dashboard/app.py` if the repository contains the wider project.

## Push these runtime files and folders

- `app.py`
- `config.py`
- `data_loader.py`
- `utils.py`
- `requirements.txt`
- `pages/`
- `data/`
- `artifacts/`
- `assets/`
- `.streamlit/config.toml`

The complete `data/` folder is safe to deploy: row-level incident files exclude names, source IDs, addresses, coordinates, age, age group, gender, race and mental-health status. Race and mental-health results are present only as aggregate counts/proportions. The two files in `artifacts/` are the locked K-Means model and preprocessing metadata; they contain no raw incident rows.

## Push these documentation files (recommended)

- `README.md`
- `DASHBOARD_BUILD_REPORT.md`
- `DASHBOARD_ACCEPTANCE_CHECKS.csv`
- `DASHBOARD_ACCEPTANCE_SUMMARY.json`
- `STANDALONE_DEPLOYMENT_CHECKS.csv`
- `STANDALONE_DEPLOYMENT_SUMMARY.json`
- `DEPLOYMENT_MANIFEST.md`

## Optional local QA/rebuild files

These can be pushed for reproducibility but are not imported by the running app:

- `smoke_test.py`
- `standalone_deployment_audit.py`
- `matcher_responsiveness_audit.py`
- `validate_dashboard.py`
- `build_dashboard_data.py` (offline rebuild utility; requires the original parent research workspace)
- `requirements-build.txt` (extra dependencies for the offline rebuild utility)
- `run_dashboard.bat`
- `run_dashboard.sh`

## Never push

- `.venv/` or `venv/`
- any `__pycache__/` folder
- `*.pyc`, `*.pyo` or `*.pyd`
- `.streamlit/secrets.toml`
- `streamlit_test*.log`
- the parent `P1/` or `P2/` research folders
- original/raw incident CSV files outside `dashboard/data/`
- Word/PDF drafts, local attachments or personal working files

The supplied `.gitignore` excludes the local environment, Python caches, secrets and test logs.

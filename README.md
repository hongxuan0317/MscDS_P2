# Fatal Police Shooting Pattern Explorer

This Streamlit application is the presentation layer for the locked P2 clustering analysis by Yah Hong Xuan (U2103393), Master in Data Science. The deployed app reads only the validated data and saved artefacts packaged inside this folder; it does not need the parent `P2/` folder and does not fit, tune or select clustering models.

## Windows setup

Open PowerShell in the `dashboard` directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

After setup, `run_dashboard.bat` starts the application.

To rerun the final integrity audit:

```powershell
.\.venv\Scripts\python.exe validate_dashboard.py
```

## macOS/Linux setup

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run app.py
```

After setup, make `run_dashboard.sh` executable if required and run it.

## Pages

1. Overview — KPIs, study design, filters and descriptive source-data charts.
2. Incident Profile Explorer — four fixed incident profiles and post-hoc PCA.
3. County Pattern Explorer — interactive K-Means/SCHC county map and profiles.
4. Model Evaluation — separate incident, conventional-county and spatial tabs.
5. Interactive Profile Explorer — existing anonymized assignments and a restricted centroid-similarity demonstration with neutral, existing-county and customized contextual modes.

## Data and interpretation

Dashboard-ready files in `data/` are reproducibly derived from the validated P2 outputs. The existing incident K-Means model and preprocessing metadata are packaged read-only in `artifacts/` for Page 5. No page interaction calls a clustering fit method.

This application presents descriptive patterns. It does not estimate individual probabilities, establish causality, rank people or counties, or support predictive policing.

Implementation sources, transformations and verification evidence are documented in `DASHBOARD_BUILD_REPORT.md`. Machine-readable acceptance results are saved in `DASHBOARD_ACCEPTANCE_CHECKS.csv`.

## Streamlit Community Cloud

The deployment is self-contained. Push the files listed in `DEPLOYMENT_MANIFEST.md`, set the entry point to `app.py` (or `dashboard/app.py` if using a wider repository), and select Python 3.12. Do not push `.venv/`, Python caches, secrets, or any parent research folder.

## Optional offline data rebuild

`build_dashboard_data.py` is a provenance utility, not a launch requirement. It requires the original parent research workspace and the additional packages in `requirements-build.txt`. It should not be run by Streamlit Community Cloud.

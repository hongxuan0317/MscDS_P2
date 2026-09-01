from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = DASHBOARD_DIR / "data"
ASSETS_DIR = DASHBOARD_DIR / "assets"
ARTIFACTS_DIR = DASHBOARD_DIR / "artifacts"

# Runtime paths are entirely local to dashboard/ so the deployed application has
# no dependency on the parent P2 research workspace.
INCIDENT_MODEL_PATH = ARTIFACTS_DIR / "incident_winner_primary_KMeans.joblib"
INCIDENT_PREPROCESSOR_PATH = ARTIFACTS_DIR / "incident_preprocessor_primary.joblib"

EXPECTED = {
    "source_incidents": 10_430,
    "incident_modelling": 10_399,
    "incident_clusters": 4,
    "counties": 3_143,
    "zero_fps_counties": 1_357,
    "county_clusters": 2,
    "spatial_regions": 2,
    "fps_positive_counties": 1_786,
}

INCIDENT_CLUSTER_COLORS = {
    "0": "#2563EB",
    "1": "#D97706",
    "2": "#059669",
    "3": "#7C3AED",
}
COUNTY_CLUSTER_COLORS = {"0": "#2563EB", "1": "#F59E0B"}
SPATIAL_CLUSTER_COLORS = {"0": "#0F766E", "1": "#C2410C"}

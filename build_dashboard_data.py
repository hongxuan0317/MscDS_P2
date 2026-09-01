"""Build compact dashboard-only copies from locked, validated P2 exports."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    EXPECTED,
    INCIDENT_MODEL_PATH,
    INCIDENT_PREPROCESSOR_PATH,
)

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent
P2_DIR = PROJECT_DIR / "P2"
P2_DATA_DIR = P2_DIR / "data"
P2_OUTPUT_DIR = P2_DIR / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

STATE_INFO = {
    "01": ("AL", "Alabama"), "02": ("AK", "Alaska"), "04": ("AZ", "Arizona"),
    "05": ("AR", "Arkansas"), "06": ("CA", "California"), "08": ("CO", "Colorado"),
    "09": ("CT", "Connecticut"), "10": ("DE", "Delaware"),
    "11": ("DC", "District of Columbia"), "12": ("FL", "Florida"),
    "13": ("GA", "Georgia"), "15": ("HI", "Hawaii"), "16": ("ID", "Idaho"),
    "17": ("IL", "Illinois"), "18": ("IN", "Indiana"), "19": ("IA", "Iowa"),
    "20": ("KS", "Kansas"), "21": ("KY", "Kentucky"), "22": ("LA", "Louisiana"),
    "23": ("ME", "Maine"), "24": ("MD", "Maryland"), "25": ("MA", "Massachusetts"),
    "26": ("MI", "Michigan"), "27": ("MN", "Minnesota"), "28": ("MS", "Mississippi"),
    "29": ("MO", "Missouri"), "30": ("MT", "Montana"), "31": ("NE", "Nebraska"),
    "32": ("NV", "Nevada"), "33": ("NH", "New Hampshire"), "34": ("NJ", "New Jersey"),
    "35": ("NM", "New Mexico"), "36": ("NY", "New York"),
    "37": ("NC", "North Carolina"), "38": ("ND", "North Dakota"),
    "39": ("OH", "Ohio"), "40": ("OK", "Oklahoma"), "41": ("OR", "Oregon"),
    "42": ("PA", "Pennsylvania"), "44": ("RI", "Rhode Island"),
    "45": ("SC", "South Carolina"), "46": ("SD", "South Dakota"),
    "47": ("TN", "Tennessee"), "48": ("TX", "Texas"), "49": ("UT", "Utah"),
    "50": ("VT", "Vermont"), "51": ("VA", "Virginia"), "53": ("WA", "Washington"),
    "54": ("WV", "West Virginia"), "55": ("WI", "Wisconsin"), "56": ("WY", "Wyoming"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def manifest(paths: list[Path], root: Path) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths) if path.is_file()
    ])


locked_sources = [
    P2_DIR / "P2_Full_Research_Analysis.py",
    P2_DIR / "P2_Full_Research_Analysis.ipynb",
    P2_DIR / "P2_Full_Research_Analysis.html",
]
locked_outputs = [path for path in P2_OUTPUT_DIR.rglob("*") if path.is_file()]
source_hashes_before = manifest(locked_sources, PROJECT_DIR)
output_hashes_before = manifest(locked_outputs, PROJECT_DIR)
source_hashes_before.to_csv(DATA_DIR / "locked_p2_source_hashes_before.csv", index=False)
output_hashes_before.to_csv(DATA_DIR / "locked_p2_output_hashes_before.csv", index=False)

# ---------------------------------------------------------------------------
# Incident-level dashboard data. Personally identifying fields are never copied.
# ---------------------------------------------------------------------------
full = pd.read_csv(PROJECT_DIR / "df_final_dataset_full_10430.csv", low_memory=False)
model_source = pd.read_csv(P2_DATA_DIR / "incident_model_source_10399.csv", low_memory=False)
labels = pd.read_csv(P2_OUTPUT_DIR / "incident" / "incident_selected_cluster_labels.csv")
cluster_names = pd.read_csv(P2_OUTPUT_DIR / "incident" / "incident_cluster_descriptive_names.csv")
name_map = cluster_names.set_index("cluster")["descriptive_name"].to_dict()

assert len(full) == EXPECTED["source_incidents"]
assert len(model_source) == EXPECTED["incident_modelling"] and model_source["id"].is_unique
assert labels["id"].is_unique

source_fields = ["year", "state", "armed_with", "threat_type", "flee_status", "body_camera"]
source_incidents = full[source_fields].copy()
source_incidents.to_csv(DATA_DIR / "source_incidents.csv", index=False)

# Sensitive characteristics required for report-level description are exported
# only as overall aggregates, never as public row-level records.
sensitive_aggregates = []
for variable in ["race", "was_mental_illness_related"]:
    counts = full[variable].fillna("Unknown").astype(str).value_counts(dropna=False)
    for category, count in counts.items():
        sensitive_aggregates.append({
            "variable": variable,
            "category": category,
            "count": int(count),
            "share": float(count / len(full)),
        })
pd.DataFrame(sensitive_aggregates).to_csv(DATA_DIR / "source_sensitive_aggregates.csv", index=False)

safe_fields = [
    "id", "year", "state", "armed_with", "threat_type", "flee_status", "body_camera",
    "median_income", "poverty_rate", "unemployment_rate", "gini_index",
    "bachelor_plus_rate", "less_than_highschool_rate", "population_density",
    "land_area_sqmi",
]
incident_dashboard = model_source[safe_fields].merge(
    labels[["id", "primary_cluster"]], on="id", how="left", validate="one_to_one"
)
incident_dashboard["profile_name"] = incident_dashboard["primary_cluster"].map(name_map)
incident_dashboard.insert(
    0, "display_record",
    [f"Incident record {i:05d}" for i in range(1, len(incident_dashboard) + 1)],
)
incident_dashboard = incident_dashboard.drop(columns="id")
incident_dashboard.to_csv(DATA_DIR / "incident_modelling_anonymized.csv", index=False)

# Derive complete cluster/overall categorical composition from validated modelling
# rows and fixed assignments. This is descriptive aggregation only.
composition_rows = []
composition_variables = [
    "age_group", "gender", "race", "armed_with", "threat_type", "flee_status",
    "was_mental_illness_related", "body_camera",
]
composition_source = model_source[["id", *composition_variables]].merge(
    labels[["id", "primary_cluster"]], on="id", how="left", validate="one_to_one"
)
for variable in composition_variables:
    for scope, frame in [("Overall", composition_source)]:
        proportions = frame[variable].astype(str).value_counts(normalize=True, dropna=False)
        for category, proportion in proportions.items():
            composition_rows.append({
                "variable": variable, "scope": scope, "cluster": "Overall",
                "category": category, "proportion": proportion,
            })
    for cluster, frame in composition_source.groupby("primary_cluster"):
        proportions = frame[variable].astype(str).value_counts(normalize=True, dropna=False)
        for category, proportion in proportions.items():
            composition_rows.append({
                "variable": variable, "scope": f"Cluster {cluster}", "cluster": int(cluster),
                "category": category, "proportion": proportion,
            })
pd.DataFrame(composition_rows).to_csv(DATA_DIR / "incident_composition_long.csv", index=False)

# Exact post-hoc PCA reproduction: same stored 52-feature matrix, same algorithm,
# same fixed seed, and never used for model fitting or selection.
incident_npz = np.load(P2_DATA_DIR / "incident_matrix_primary.npz", allow_pickle=True)
X_incident = np.asarray(incident_npz["X"], dtype=np.float64)
matrix_ids = pd.Series(incident_npz["ids"]).astype(str)
assert X_incident.shape == (EXPECTED["incident_modelling"], 52)
assert np.array_equal(model_source["id"].astype(str).to_numpy(), matrix_ids.to_numpy())
pca = PCA(n_components=2, random_state=42)
coordinates = pca.fit_transform(X_incident)
pca_table = pd.DataFrame({
    "display_record": incident_dashboard["display_record"],
    "PC1": coordinates[:, 0], "PC2": coordinates[:, 1],
    "primary_cluster": incident_dashboard["primary_cluster"].astype(int),
    "profile_name": incident_dashboard["profile_name"],
})
pca_table.to_csv(DATA_DIR / "incident_pca_coordinates.csv", index=False)
pca_variance = pd.DataFrame([
    {"component": "PC1", "explained_variance_ratio": pca.explained_variance_ratio_[0]},
    {"component": "PC2", "explained_variance_ratio": pca.explained_variance_ratio_[1]},
    {"component": "PC1+PC2 cumulative", "explained_variance_ratio": pca.explained_variance_ratio_[:2].sum()},
])
pca_variance["explained_variance_pct"] = 100 * pca_variance["explained_variance_ratio"]
pca_variance.to_csv(DATA_DIR / "incident_pca_explained_variance.csv", index=False)

# Reference modelling-space vector for the ethical hypothetical matcher. Unspecified
# features use the population mean in the exact 52-feature modelling matrix.
np.save(DATA_DIR / "incident_modelling_space_mean.npy", X_incident.mean(axis=0))
pd.read_csv(P2_DATA_DIR / "incident_feature_dictionary_primary.csv").to_csv(
    DATA_DIR / "incident_feature_dictionary.csv", index=False
)

feature_dictionary = pd.read_csv(P2_DATA_DIR / "incident_feature_dictionary_primary.csv")
scope_map = {
    "armed_with": "Incident/situational", "threat_type": "Incident/situational",
    "flee_status": "Incident/situational", "body_camera": "Incident/situational",
    "age_group": "Personal/demographic", "gender": "Personal/demographic",
    "race": "Personal/demographic", "was_mental_illness_related": "Personal/demographic",
}
feature_scope = (
    feature_dictionary[["source_variable", "group"]]
    .drop_duplicates()
    .assign(
        conceptual_scope=lambda frame: frame["source_variable"].map(scope_map).fillna("County contextual"),
        exposed_in_hypothetical=lambda frame: frame["source_variable"].isin([
            "armed_with", "threat_type", "flee_status", "body_camera", "median_income",
            "poverty_rate", "unemployment_rate", "gini_index", "bachelor_plus_rate",
            "less_than_highschool_rate", "population_density", "land_area_sqmi",
        ]),
    )
)
feature_scope.to_csv(DATA_DIR / "incident_feature_scope.csv", index=False)

context_variables = [
    "median_income", "poverty_rate", "unemployment_rate", "gini_index",
    "bachelor_plus_rate", "less_than_highschool_rate", "population_density", "land_area_sqmi",
]
context_rows = []
for variable in context_variables:
    if variable == "population_density":
        levels = [("Rural", 0.10), ("Mixed", 0.35), ("Urban", 0.65), ("Highly urban", 0.90)]
    else:
        levels = [("Lower (25th percentile)", 0.25), ("Typical (median)", 0.50), ("Higher (75th percentile)", 0.75)]
    for label, quantile in levels:
        context_rows.append({
            "variable": variable,
            "selection_label": label,
            "percentile": quantile,
            "raw_value": float(model_source[variable].quantile(quantile)),
        })
pd.DataFrame(context_rows).to_csv(DATA_DIR / "context_reference_values.csv", index=False)

# Copy/derive small incident summaries needed by the app.
incident_copy_map = {
    P2_OUTPUT_DIR / "incident" / "incident_cluster_report_interpretation_table.csv": "incident_profiles.csv",
    P2_OUTPUT_DIR / "incident" / "incident_cluster_descriptive_names.csv": "incident_cluster_names.csv",
    P2_OUTPUT_DIR / "incident" / "selected_incident_model_metrics.csv": "incident_selected_metrics.csv",
    P2_OUTPUT_DIR / "incident" / "incident_all_candidate_metrics.csv": "incident_candidate_metrics.csv",
    P2_OUTPUT_DIR / "incident" / "primary_vs_time_sensitivity.csv": "incident_time_sensitivity.csv",
}
for source, target_name in incident_copy_map.items():
    pd.read_csv(source).to_csv(DATA_DIR / target_name, index=False)

# ---------------------------------------------------------------------------
# County dashboard data and compact display geometry.
# ---------------------------------------------------------------------------
county = pd.read_csv(
    P2_OUTPUT_DIR / "comparison" / "all_county_primary_assignments.csv",
    dtype={"county_fips": str, "STATEFP": str},
)
county["county_fips"] = county["county_fips"].str.zfill(5)
county["STATEFP"] = county["STATEFP"].str.zfill(2)
county["state_abbr"] = county["STATEFP"].map(lambda value: STATE_INFO[value][0])
county["state_name"] = county["STATEFP"].map(lambda value: STATE_INFO[value][1])
county["fps_status"] = np.where(county["fps_count_2015_2024"] > 0, "FPS-positive", "Zero FPS")
county["conventional_cluster_label"] = "County K-Means cluster " + county["conventional_county_cluster"].astype(str)
county["spatial_cluster_label"] = "SCHC region " + county["spatial_cluster"].astype(str)

county_fields = [
    "county_fips", "county_name_2020", "acs_name", "STATEFP", "state_abbr", "state_name",
    "fps_count_2015_2024", "avg_annual_fps_rate_per_100k", "fps_status",
    "conventional_county_cluster", "conventional_cluster_label",
    "spatial_cluster", "spatial_cluster_label", "median_income", "poverty_rate",
    "unemployment_rate", "bachelor_plus_rate", "less_than_highschool_rate",
    "population_density", "land_area_sqmi", "gini_index", "black_pct", "hispanic_pct",
    "profile_armed_with_gun", "profile_armed_with_knife",
    "profile_threat_type_shoot", "profile_flee_status_not_fleeing",
]
county_dashboard = county[county_fields].copy()
county_dashboard.to_csv(DATA_DIR / "counties.csv", index=False)

geometries = gpd.read_file(
    P2_DATA_DIR / "county_model_source_prior_5.gpkg", layer="counties"
)[["county_fips", "geometry"]]
geometries["county_fips"] = geometries["county_fips"].astype(str).str.zfill(5)
county_geo = geometries.merge(county_dashboard, on="county_fips", how="left", validate="one_to_one")
assert len(county_geo) == EXPECTED["counties"] and county_geo["county_fips"].is_unique
# Display-only simplification in a projected CRS; no analytical geometry is changed.
county_geo = county_geo.to_crs("EPSG:5070")
county_geo["geometry"] = county_geo.geometry.simplify(1_500, preserve_topology=True)
county_geo = county_geo.to_crs("EPSG:4326")
(DATA_DIR / "counties.geojson").write_text(county_geo.to_json(drop_id=True), encoding="utf-8")

county_copy_map = {
    P2_OUTPUT_DIR / "county_conventional" / "selected_county_conventional_profiles.csv": "county_conventional_profiles.csv",
    P2_OUTPUT_DIR / "county_spatial" / "selected_county_spatial_profiles.csv": "county_spatial_profiles.csv",
    P2_OUTPUT_DIR / "spatial" / "spatial_cluster_descriptive_names.csv": "county_spatial_names.csv",
    P2_OUTPUT_DIR / "county_conventional" / "county_conventional_all_candidate_metrics.csv": "county_conventional_metrics.csv",
    P2_OUTPUT_DIR / "county_spatial" / "county_spatial_all_candidate_metrics.csv": "county_spatial_metrics.csv",
    P2_OUTPUT_DIR / "diagnostics" / "all_county_fps_burden_morans_i.csv": "county_morans_i.csv",
    P2_OUTPUT_DIR / "diagnostics" / "model_selection_rules.csv": "model_selection_rules.csv",
    P2_OUTPUT_DIR / "comparison" / "three_representative_model_summary.csv": "selected_model_summary.csv",
    P2_OUTPUT_DIR / "sensitivity" / "spatial_smoothing_sensitivity_agreement.csv": "spatial_smoothing_sensitivity.csv",
    P2_OUTPUT_DIR / "sensitivity" / "lower48_queen_only_sensitivity.csv": "lower48_sensitivity.csv",
    P2_OUTPUT_DIR / "sensitivity" / "fps_positive_representative_model_comparison.csv": "fps_positive_sensitivity.csv",
}
for source, target_name in county_copy_map.items():
    pd.read_csv(source).to_csv(DATA_DIR / target_name, index=False)

# Dashboard terminology: the source label's "burden" dimension is the
# annualized FPS rate, so use that direct term in the presentation copy.
spatial_names_path = DATA_DIR / "county_spatial_names.csv"
dashboard_spatial_names = pd.read_csv(spatial_names_path)
dashboard_spatial_names["descriptive_name"] = (
    dashboard_spatial_names["descriptive_name"]
    .str.replace("lower-burden", "lower-FPS-rate", regex=False)
    .str.replace("Higher-burden", "Higher-FPS-rate", regex=False)
    .str.replace("higher-burden", "higher-FPS-rate", regex=False)
)
dashboard_spatial_names.to_csv(spatial_names_path, index=False)

# Package only the two non-identifying, locked artefacts required at runtime.
source_model_path = P2_OUTPUT_DIR / "models" / "incident_winner_primary_KMeans.joblib"
source_preprocessor_path = P2_OUTPUT_DIR / "models" / "incident_preprocessor_primary.joblib"
shutil.copy2(source_model_path, INCIDENT_MODEL_PATH)
shutil.copy2(source_preprocessor_path, INCIDENT_PREPROCESSOR_PATH)
assert sha256_file(source_model_path) == sha256_file(INCIDENT_MODEL_PATH)
assert sha256_file(source_preprocessor_path) == sha256_file(INCIDENT_PREPROCESSOR_PATH)

# Runtime validation evidence.
model = joblib.load(INCIDENT_MODEL_PATH)
preprocessor = joblib.load(INCIDENT_PREPROCESSOR_PATH)
validation = {
    "source_incidents": len(full),
    "incident_modelling": len(incident_dashboard),
    "incident_ids_unique_before_anonymization": bool(model_source["id"].is_unique),
    "incident_clusters": int(incident_dashboard["primary_cluster"].nunique()),
    "counties": len(county_dashboard),
    "county_fips_unique": bool(county_dashboard["county_fips"].is_unique),
    "zero_fps_counties": int((county_dashboard["fps_count_2015_2024"] == 0).sum()),
    "county_clusters": int(county_dashboard["conventional_county_cluster"].nunique()),
    "spatial_regions": int(county_dashboard["spatial_cluster"].nunique()),
    "fps_positive_counties": int((county_dashboard["fps_count_2015_2024"] > 0).sum()),
    "county_geometries": int(county_geo.geometry.notna().sum()),
    "incident_model_centroids": list(model.cluster_centers_.shape),
    "preprocessor_keys": sorted(preprocessor.keys()),
    "pca_variance": pca_variance.set_index("component")["explained_variance_pct"].to_dict(),
}
assert validation["source_incidents"] == EXPECTED["source_incidents"]
assert validation["incident_modelling"] == EXPECTED["incident_modelling"]
assert validation["incident_clusters"] == EXPECTED["incident_clusters"]
assert validation["counties"] == EXPECTED["counties"]
assert validation["zero_fps_counties"] == EXPECTED["zero_fps_counties"]
assert validation["county_clusters"] == EXPECTED["county_clusters"]
assert validation["spatial_regions"] == EXPECTED["spatial_regions"]
assert validation["fps_positive_counties"] == EXPECTED["fps_positive_counties"]
(DATA_DIR / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

# Prove locked P2 sources and outputs were not touched during derivation.
source_hashes_after = manifest(locked_sources, PROJECT_DIR)
output_hashes_after = manifest(locked_outputs, PROJECT_DIR)
source_hashes_after.to_csv(DATA_DIR / "locked_p2_source_hashes_after.csv", index=False)
output_hashes_after.to_csv(DATA_DIR / "locked_p2_output_hashes_after.csv", index=False)
assert source_hashes_before.equals(source_hashes_after)
assert output_hashes_before.equals(output_hashes_after)

dashboard_files = [path for path in DATA_DIR.rglob("*") if path.is_file()]
dashboard_files.extend(path for path in ARTIFACTS_DIR.rglob("*") if path.is_file())
manifest(dashboard_files, PROJECT_DIR).to_csv(DATA_DIR / "dashboard_data_manifest.csv", index=False)
print(json.dumps(validation, indent=2))
print("Dashboard-ready data built without modifying locked P2 files.")

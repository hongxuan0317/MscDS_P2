# Dashboard Build Report

Build completed and verified on 31 August 2026 using Python 3.12.0. The dashboard is a presentation and exploration layer only. The locked P2 analysis was not refitted, retuned, reselected, overwritten, or otherwise modified.

## 1. Source P2 notebook/script used

The validated analysis was traced to:

- `P2/P2_Full_Research_Analysis.ipynb`
- `P2/P2_Full_Research_Analysis.py`
- `P2/P2_Full_Research_Analysis.html`

The primary export directory is `P2/outputs/`. SHA-256 snapshots were taken for the three analysis files and all 131 existing P2 output files before and after dashboard-data construction. Every hash remained identical.

## 2. Source output files used

Incident data and artefacts:

- `df_final_dataset_full_10430.csv`
- `P2/data/incident_model_source_10399.csv`
- `P2/data/incident_matrix_primary.npz`
- `P2/data/incident_feature_dictionary_primary.csv`
- `P2/outputs/incident/incident_selected_cluster_labels.csv`
- `P2/outputs/incident/incident_cluster_descriptive_names.csv`
- `P2/outputs/incident/incident_cluster_report_interpretation_table.csv`
- `P2/outputs/incident/selected_incident_model_metrics.csv`
- `P2/outputs/incident/incident_all_candidate_metrics.csv`
- `P2/outputs/incident/primary_vs_time_sensitivity.csv`
- `P2/outputs/models/incident_winner_primary_KMeans.joblib`
- `P2/outputs/models/incident_preprocessor_primary.joblib`

County, spatial and sensitivity data:

- `P2/outputs/comparison/all_county_primary_assignments.csv`
- `P2/data/county_model_source_prior_5.gpkg`, layer `counties`
- `P2/outputs/county_conventional/selected_county_conventional_profiles.csv`
- `P2/outputs/county_conventional/county_conventional_all_candidate_metrics.csv`
- `P2/outputs/county_spatial/selected_county_spatial_profiles.csv`
- `P2/outputs/county_spatial/county_spatial_all_candidate_metrics.csv`
- `P2/outputs/spatial/spatial_cluster_descriptive_names.csv`
- `P2/outputs/comparison/three_representative_model_summary.csv`
- `P2/outputs/diagnostics/all_county_fps_burden_morans_i.csv`
- `P2/outputs/diagnostics/model_selection_rules.csv`
- `P2/outputs/sensitivity/spatial_smoothing_sensitivity_agreement.csv`
- `P2/outputs/sensitivity/lower48_queen_only_sensitivity.csv`
- `P2/outputs/sensitivity/fps_positive_representative_model_comparison.csv`

No result was imported from a separate diagnostic/sensitivity notebook.

## 3. Dashboard-ready derived files created

All derived copies are under `dashboard/data/` and are reproducible with `build_dashboard_data.py`.

| Purpose | Dashboard files |
|---|---|
| Anonymized descriptive records | `source_incidents.csv`, `incident_modelling_anonymized.csv`, `source_sensitive_aggregates.csv` |
| Incident labels and profiles | `incident_profiles.csv`, `incident_cluster_names.csv`, `incident_composition_long.csv` |
| Incident diagnostics | `incident_selected_metrics.csv`, `incident_candidate_metrics.csv`, `incident_time_sensitivity.csv` |
| PCA display | `incident_pca_coordinates.csv`, `incident_pca_explained_variance.csv` |
| Matcher reference | `incident_modelling_space_mean.npy`, `incident_feature_dictionary.csv`, `incident_feature_scope.csv`, `context_reference_values.csv` |
| County display | `counties.csv`, `counties.geojson` |
| County profiles/metrics | `county_conventional_profiles.csv`, `county_spatial_profiles.csv`, `county_spatial_names.csv`, `county_conventional_metrics.csv`, `county_spatial_metrics.csv` |
| Selection and sensitivity | `selected_model_summary.csv`, `model_selection_rules.csv`, `county_morans_i.csv`, `spatial_smoothing_sensitivity.csv`, `lower48_sensitivity.csv`, `fps_positive_sensitivity.csv` |
| Integrity evidence | `validation_summary.json`, `dashboard_data_manifest.csv`, four locked-P2 hash manifests |

Names, original incident IDs, addresses, cities, coordinates, age, age group, gender, race and mental-health status were not copied into row-level deployment tables. Race and mental-health results are retained only as overall or cluster-level aggregates. Stable labels such as `Incident record 00001` are dashboard-only row labels.

## 4. Data mappings and joins used

- Incident modelling rows were joined one-to-one to their fixed K-Means assignments using the validated incident `id`. The source ID was removed after the join.
- Incident profile names were mapped from the existing cluster-name export. No new cluster name was invented.
- The stored incident matrix IDs were asserted to be in exactly the same order as `incident_model_source_10399.csv` before PCA coordinates or reference means were created.
- Complete selected-cluster and overall categorical compositions were aggregated from the fixed modelling rows and labels. This is descriptive aggregation, not model fitting.
- The county table came from the existing all-county primary assignment export, which already contains the conventional and spatial labels. State names/abbreviations were mapped from two-digit state FIPS.
- County geometry was joined one-to-one using five-digit `county_fips`. Geometry was simplified by 1,500 metres in EPSG:5070 only for browser display, then converted to EPSG:4326. Analytical geometry and P2 files were unchanged.

## 5. Pages implemented

1. **Overview** — dynamic study KPIs, two-phase design, display-only filters, and interactive descriptive charts.
2. **Incident Profile Explorer** — the four existing incident profiles, cluster-versus-overall composition, original-unit context and an interactive post-hoc PCA plot.
3. **County Pattern Explorer** — toggle between County K-Means and SCHC, an interactive 3,143-county map, display-only state/label/FPS filters, and original-unit profile comparisons.
4. **Model Evaluation** — separate incident, conventional-county and spatial tabs; no cross-unit universal leaderboard.
5. **Interactive Profile Explorer** — existing anonymized incident assignments plus a restricted, non-sensitive centroid-similarity demonstration.

Every page shows the persistent non-causal, non-predictive disclaimer.

## 6. Metrics displayed

The app reads the exported values dynamically. The selected representatives are:

| Analytical family | Representative | Silhouette | DBI | CHI | Resampling ARI | Resampling NMI | Boundary cut |
|---|---:|---:|---:|---:|---:|---:|---:|
| Incident | K-Means k=4 | 0.160 | 1.749 | 1,752.4 | 0.985 | 0.976 | — |
| Conventional county | County K-Means k=2 | 0.127 | 2.638 | 360.5 | 0.981 | 0.961 | 0.233 |
| Spatial county | SCHC k=2 | 0.077 | 3.417 | 203.7 | — | — | 0.037 |

Candidate tables additionally show cluster-size usability, connected components, spatial connectedness, smoothing sensitivity and relevant HDBSCAN diagnostics. HDBSCAN's noise-excluded Silhouette is always paired with its 30.2% noise share; persistence is not presented as equivalent to ARI/NMI. Incident and county metrics are never combined into a universal ranking.

## 7. County map data source

The map uses the 3,143 validated 2020 county-equivalent geometries from `P2/data/county_model_source_prior_5.gpkg`. All 1,357 zero-FPS counties remain visible by default. Existing County K-Means k=2 and SCHC k=2 assignments are switched only at display time. The map contains 3,143 unique FIPS geometries and exposes contextual values through hover information. Dashboard labels now use the direct term **FPS rate** instead of higher/lower “burden”; this is a presentation wording change only.

## 8. Page 5 matching methodology

Page 5 loads dashboard-local, hash-identical copies of `incident_winner_primary_KMeans.joblib` and `incident_preprocessor_primary.joblib` through `@st.cache_resource`. It never calls `fit`, `fit_predict`, model selection, or hyperparameter tuning.

For a hypothetical scenario, the page:

1. starts with a neutral reference vector in the exact 52-feature primary modelling space;
2. replaces only the original one-hot blocks for armed status, threat type and fleeing status using the original block scaling of `1/sqrt(2)`;
3. replaces only the existing body-camera binary feature;
4. optionally replaces eight genuine non-sensitive county-context features: median income, poverty, unemployment, Gini, bachelor's attainment, less-than-high-school attainment, population density and land area;
5. applies the locked StandardScaler conventions, including `log1p` before scaling density and land area;
6. computes Euclidean distance to the four saved K-Means centroids; and
7. converts inverse distances into a normalized **Profile Similarity** display.

Three context modes are available: neutral average context, an existing county from the 3,143-county primary table, or customized observed quantiles. The custom controls use 25th percentile/median/75th percentile values; density uses the 10th, 35th, 65th and 90th percentiles labelled Rural, Mixed, Urban and Highly urban. Every selected control displays the original-unit value supplied to the transformation.

The normalized display is explicitly not labelled or interpreted as probability, confidence, risk, likelihood, dangerousness or an outcome estimate.

## 9. Representation of unspecified hypothetical variables

The neutral reference is the element-wise mean of all 10,399 rows in the exact stored 52-feature modelling matrix. Consequently:

- standardized continuous/contextual features use their overall modelling-space mean;
- unselected categorical blocks retain the empirical mean of their scaled one-hot indicators, representing the modelling population distribution rather than an invented person;
- selected situational categorical blocks use the original feature dictionary and scaling;
- age, gender, race, mental-illness status, county racial composition and county youth composition remain unavailable as inputs and retain their neutral reference values;
- the eight approved context controls describe county conditions, not individual characteristics.

This representation is saved as `incident_modelling_space_mean.npy` and checked against the 52-row feature dictionary. The feature audit separates 23 incident/situational dimensions, 18 personal/demographic dimensions and 11 county-context dimensions.

The responsiveness audit evaluated all 640 combinations of the original four situational inputs. The original matcher reached only Clusters 1 and 3, with Cluster 3 receiving 92.8% of scenarios. The expanded audit evaluated those combinations across seven neutral/low/typical/high/rural/urban contexts (4,480 scenarios): Cluster 0 received 14.3%, Cluster 1 43.8%, Cluster 2 14.3% and Cluster 3 27.6%. It reached all four profiles, no cluster exceeded 80%, and mean first-versus-second distance margin rose from 0.036 to 0.197. Representative interpretable combinations for each profile are exported in `matcher_responsiveness_examples.csv`; the expanded matcher is therefore retained.

## 10. Ethical restrictions implemented

- No hypothetical inputs for race, gender, age, mental-health status or other sensitive personal characteristics.
- No individual-risk, event-probability, dangerousness, police-behaviour or intervention output.
- No recommendation for police action, resource allocation or county ranking.
- Race and mental-health results are deployed only as aggregate counts/proportions; row-level age, gender, race and mental-health fields are absent.
- County socioeconomic characteristics are labelled as contextual and are not attributed to individuals.
- Cluster IDs and colours are described as non-ordinal and non-causal.
- Personally identifying incident fields are excluded.
- The app repeatedly distinguishes the FPS dataset from a predictive or causal population model.

## 11. Validation checks

`validate_dashboard.py` produced `DASHBOARD_ACCEPTANCE_CHECKS.csv` and `DASHBOARD_ACCEPTANCE_SUMMARY.json`: **21 passed, 0 failed**.

Confirmed:

- 10,430 raw/source incidents;
- 10,399 unique anonymized modelling records and four fixed incident labels;
- 3,143 unique county FIPS rows and geometries;
- 1,357 zero-FPS counties and 1,786 FPS-positive sensitivity counties;
- two County K-Means labels and two SCHC region labels;
- saved model centroid shape `(4, 52)`;
- saved preprocessing metadata available;
- no row-level sensitive or identifying deployment columns;
- dashboard-local model and preprocessing artefacts available;
- all required plotting and application files available;
- three P2 analysis-file hashes and 131 P2 output-file hashes unchanged; and
- no prohibited clustering-training call in launch-time application code.

The post-hoc PCA coordinates were the only requested display field not already exported. They were deterministically reproduced from the exact stored 52-feature matrix with the original fixed seed. PC1 explains 27.599%, PC2 17.556%, and the cumulative share is 45.155%. This PCA is display-only and does not affect clustering.

## 12. Runtime checks performed

- Created a clean dashboard-specific Python 3.12.0 environment and installed `requirements.txt`.
- Rebuilt dashboard data successfully in that environment.
- Compiled every dashboard Python file without syntax errors.
- Rendered all five pages with Streamlit's application test framework: all passed without application exceptions.
- Exercised the County K-Means/SCHC toggle: passed.
- Exercised a county state display filter: passed.
- Exercised neutral, existing-county and customized hypothetical context modes: passed.
- Audited 640 original and 4,480 expanded matcher scenarios: passed, with all four profiles reached by the expanded design.
- Audited launch-time source for prohibited model-fitting tokens: passed.
- Launched the real local Streamlit server on port 8502.
- Requested `/_stcore/health`: HTTP 200, `ok`.
- Requested the app root: HTTP 200.
- Shut the test server down cleanly.
- Copied only runtime files into an isolated temporary directory with no parent `P2/` folder, rendered all five pages, and launched its server: **8 standalone checks passed, 0 failed**.

## 13. Unresolved issues

No blocking issue remains. The app now loads all runtime data, model and preprocessing artefacts from paths inside `dashboard/`; the parent `P2/` folder is unnecessary for launch or Community Cloud. The original P2 exports did not contain reusable incident PCA coordinates, so the exact post-hoc PCA display was reconstructed as documented above. The optional web basemap requires internet access; the 3,143 local county polygons and analytical assignments do not.

## 14. Exact command to run the application

From the project root in Windows PowerShell:

```powershell
cd dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Alternatively, double-click or run `dashboard/run_dashboard.bat`. On macOS/Linux, after creating the documented environment, run `dashboard/run_dashboard.sh`.

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DATA_DIR,
    EXPECTED,
    INCIDENT_MODEL_PATH,
    INCIDENT_PREPROCESSOR_PATH,
)


@st.cache_data(show_spinner=False)
def _csv(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Dashboard data file is missing: {path}")
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def load_validation() -> dict:
    with (DATA_DIR / "validation_summary.json").open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    expected_pairs = {
        "source_incidents": EXPECTED["source_incidents"],
        "incident_modelling": EXPECTED["incident_modelling"],
        "incident_clusters": EXPECTED["incident_clusters"],
        "counties": EXPECTED["counties"],
        "zero_fps_counties": EXPECTED["zero_fps_counties"],
        "county_clusters": EXPECTED["county_clusters"],
        "spatial_regions": EXPECTED["spatial_regions"],
        "fps_positive_counties": EXPECTED["fps_positive_counties"],
    }
    failures = {
        key: {"expected": expected, "observed": validation.get(key)}
        for key, expected in expected_pairs.items()
        if validation.get(key) != expected
    }
    if failures:
        raise ValueError(f"Dashboard validation failed: {failures}")
    return validation


@st.cache_data(show_spinner=False)
def load_source_incidents() -> pd.DataFrame:
    frame = _csv("source_incidents.csv")
    if len(frame) != EXPECTED["source_incidents"]:
        raise ValueError("Source incident count is not 10,430")
    return frame


@st.cache_data(show_spinner=False)
def load_source_sensitive_aggregates() -> pd.DataFrame:
    """Load overall summaries; no row-level sensitive data are deployed."""
    return _csv("source_sensitive_aggregates.csv")


@st.cache_data(show_spinner=False)
def load_incident_modelling_data() -> pd.DataFrame:
    frame = _csv("incident_modelling_anonymized.csv")
    if len(frame) != EXPECTED["incident_modelling"]:
        raise ValueError("Incident modelling count is not 10,399")
    if frame["display_record"].duplicated().any():
        raise ValueError("Anonymized incident display records are not unique")
    return frame


@st.cache_data(show_spinner=False)
def load_incident_assignments() -> pd.DataFrame:
    return load_incident_modelling_data()[
        ["display_record", "primary_cluster", "profile_name"]
    ].copy()


@st.cache_data(show_spinner=False)
def load_incident_profiles() -> pd.DataFrame:
    return _csv("incident_profiles.csv")


@st.cache_data(show_spinner=False)
def load_incident_composition() -> pd.DataFrame:
    return _csv("incident_composition_long.csv")


@st.cache_data(show_spinner=False)
def load_incident_model_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _csv("incident_selected_metrics.csv"), _csv("incident_candidate_metrics.csv")


@st.cache_data(show_spinner=False)
def load_incident_pca() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _csv("incident_pca_coordinates.csv"), _csv("incident_pca_explained_variance.csv")


@st.cache_data(show_spinner=False)
def load_county_data() -> pd.DataFrame:
    frame = _csv("counties.csv", dtype={"county_fips": str, "STATEFP": str})
    frame["county_fips"] = frame["county_fips"].str.zfill(5)
    if len(frame) != EXPECTED["counties"] or not frame["county_fips"].is_unique:
        raise ValueError("County data must contain 3,143 unique FIPS rows")
    if int((frame["fps_count_2015_2024"] == 0).sum()) != EXPECTED["zero_fps_counties"]:
        raise ValueError("Zero-FPS county count is not 1,357")
    return frame


@st.cache_data(show_spinner=False)
def load_county_assignments() -> pd.DataFrame:
    return load_county_data()[[
        "county_fips", "conventional_county_cluster", "spatial_cluster"
    ]].copy()


@st.cache_data(show_spinner=False)
def load_county_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _csv("county_conventional_profiles.csv"), _csv("county_spatial_profiles.csv")


@st.cache_data(show_spinner=False)
def load_county_model_metrics() -> pd.DataFrame:
    return _csv("county_conventional_metrics.csv")


@st.cache_data(show_spinner=False)
def load_spatial_model_metrics() -> pd.DataFrame:
    return _csv("county_spatial_metrics.csv")


@st.cache_data(show_spinner=False)
def load_spatial_names() -> pd.DataFrame:
    return _csv("county_spatial_names.csv")


@st.cache_data(show_spinner=False)
def load_selected_model_summary() -> pd.DataFrame:
    return _csv("selected_model_summary.csv")


@st.cache_data(show_spinner=False)
def load_sensitivity_tables() -> dict[str, pd.DataFrame]:
    return {
        "Time features": _csv("incident_time_sensitivity.csv"),
        "Smoothing prior": _csv("spatial_smoothing_sensitivity.csv"),
        "Lower 48 + DC graph": _csv("lower48_sensitivity.csv"),
        "FPS-positive counties": _csv("fps_positive_sensitivity.csv"),
        "Moran's I": _csv("county_morans_i.csv"),
    }


@st.cache_data(show_spinner=False)
def load_geodata() -> dict:
    with (DATA_DIR / "counties.geojson").open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_feature_reference() -> tuple[pd.DataFrame, np.ndarray]:
    dictionary = _csv("incident_feature_dictionary.csv")
    reference = np.load(DATA_DIR / "incident_modelling_space_mean.npy")
    if len(dictionary) != 52 or reference.shape != (52,):
        raise ValueError("Incident feature reference must contain exactly 52 features")
    return dictionary, reference


@st.cache_data(show_spinner=False)
def load_context_reference_values() -> pd.DataFrame:
    return _csv("context_reference_values.csv")


@st.cache_data(show_spinner=False)
def load_feature_scope() -> pd.DataFrame:
    return _csv("incident_feature_scope.csv")


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> tuple[object, dict]:
    if not INCIDENT_MODEL_PATH.exists() or not INCIDENT_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Saved primary incident model/preprocessor is unavailable")
    model = joblib.load(INCIDENT_MODEL_PATH)
    preprocessor = joblib.load(INCIDENT_PREPROCESSOR_PATH)
    if model.cluster_centers_.shape != (4, 52):
        raise ValueError("Expected four fixed centroids in the 52-feature space")
    return model, preprocessor

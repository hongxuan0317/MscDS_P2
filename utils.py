from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PLOTLY_TEMPLATE = "plotly_white"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
        [data-testid="stSidebar"] {background: #F8FAFC; border-right: 1px solid #E2E8F0;}
        .hero {padding: 1.35rem 1.55rem; border-radius: 16px; color: white;
               background: linear-gradient(120deg, #0F172A 0%, #1E3A5F 55%, #0F766E 100%);
               margin-bottom: 1rem;}
        .hero h1 {margin: 0; font-size: 2.15rem; color: white;}
        .hero p {margin: .4rem 0 0; color: #DCEAF5; font-size: 1.02rem;}
        .disclaimer {padding: .75rem 1rem; border-radius: 10px; background: #FFF7ED;
                     border-left: 4px solid #EA580C; color: #7C2D12; margin: .7rem 0 1.1rem;}
        .study-card {padding: 1rem; border: 1px solid #DCE3EB; border-radius: 12px;
                     background: #FFFFFF; min-height: 145px; box-shadow: 0 2px 8px rgba(15,23,42,.04);}
        .study-card h4 {color: #0F3D56; margin-top: 0;}
        .small-note {font-size: .88rem; color: #475569;}
        div[data-testid="stMetric"] {background: white; border: 1px solid #E2E8F0;
                                     border-radius: 12px; padding: .75rem 1rem;}
        div[data-testid="stMetricValue"] {font-size: clamp(1.25rem, 1.75vw, 1.8rem) !important;
                                           line-height: 1.2; white-space: normal !important;
                                           overflow: visible !important; overflow-wrap: anywhere;
                                           text-overflow: clip !important; min-height: 2.35em;}
        div[data-testid="stMetricValue"] * {white-space: normal !important;
                                             overflow: visible !important;
                                             text-overflow: clip !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def tidy_plot(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        font=dict(family="Inter, Segoe UI, Arial", color="#1E293B"),
    )
    return fig


def percentage_label(value: float) -> str:
    return f"{100 * value:.1f}%"


def create_scenario_vector(
    dictionary: pd.DataFrame,
    reference: np.ndarray,
    armed_with: str,
    threat_type: str,
    flee_status: str,
    body_camera: bool,
    context_values: dict[str, float] | None = None,
    preprocessor: dict | None = None,
) -> np.ndarray:
    """Create a neutral vector with approved situational/contextual overrides."""
    vector = np.asarray(reference, dtype=float).copy()
    features = dictionary["feature"].astype(str).tolist()
    feature_to_index = {feature: index for index, feature in enumerate(features)}
    for variable, category in {
        "armed_with": armed_with,
        "threat_type": threat_type,
        "flee_status": flee_status,
    }.items():
        block_indices = dictionary.index[dictionary["source_variable"].eq(variable)].tolist()
        selected_feature = f"{variable}_{category}"
        if selected_feature not in feature_to_index:
            raise ValueError(f"Unsupported scenario category: {selected_feature}")
        vector[block_indices] = 0.0
        vector[feature_to_index[selected_feature]] = 1 / math.sqrt(2)
    vector[feature_to_index["body_camera"]] = float(body_camera)
    if context_values:
        if preprocessor is None:
            raise ValueError("Preprocessing metadata is required for contextual overrides")
        approved = {
            "median_income", "poverty_rate", "unemployment_rate", "gini_index",
            "bachelor_plus_rate", "less_than_highschool_rate", "population_density",
            "land_area_sqmi",
        }
        unsupported = set(context_values) - approved
        if unsupported:
            raise ValueError(f"Unsupported contextual override(s): {sorted(unsupported)}")
        scaler = preprocessor["numeric_scaler"]
        numeric_names = list(scaler.feature_names_in_)
        for variable, raw_value in context_values.items():
            if variable not in feature_to_index or variable not in numeric_names:
                raise ValueError(f"Context feature is absent from the primary model: {variable}")
            scaler_index = numeric_names.index(variable)
            transformed = np.log1p(float(raw_value)) if variable in {
                "population_density", "land_area_sqmi"
            } else float(raw_value)
            vector[feature_to_index[variable]] = (
                transformed - scaler.mean_[scaler_index]
            ) / scaler.scale_[scaler_index]
    return vector


def neutral_context_values(preprocessor: dict, variables: list[str]) -> dict[str, float]:
    """Return original-unit values corresponding to the modelling-space mean."""
    scaler = preprocessor["numeric_scaler"]
    numeric_names = list(scaler.feature_names_in_)
    result = {}
    for variable in variables:
        index = numeric_names.index(variable)
        value = float(scaler.mean_[index])
        if variable in {"population_density", "land_area_sqmi"}:
            value = float(np.expm1(value))
        result[variable] = value
    return result


def centroid_similarity(vector: np.ndarray, centers: np.ndarray) -> pd.DataFrame:
    distances = np.linalg.norm(centers - vector.reshape(1, -1), axis=1)
    inverse = 1.0 / np.maximum(distances, 1e-12)
    similarity = 100 * inverse / inverse.sum()
    return pd.DataFrame({
        "cluster": np.arange(len(centers), dtype=int),
        "distance": distances,
        "profile_similarity": similarity,
    }).sort_values("distance")


def profile_comparison_chart(
    frame: pd.DataFrame,
    cluster_column: str,
    selected_metrics: list[str],
    title: str,
) -> go.Figure:
    long = frame[[cluster_column, *selected_metrics]].melt(
        id_vars=cluster_column, var_name="metric", value_name="value"
    )
    long[cluster_column] = long[cluster_column].astype(str)
    fig = px.bar(
        long, x="metric", y="value", color=cluster_column, barmode="group",
        title=title, labels={"metric": "Characteristic", "value": "Original-unit mean"},
    )
    return tidy_plot(fig, 440)

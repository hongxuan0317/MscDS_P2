import json

import pandas as pd
import plotly.express as px
import streamlit as st

from config import COUNTY_CLUSTER_COLORS, SPATIAL_CLUSTER_COLORS
from data_loader import (
    load_county_data,
    load_county_profiles,
    load_geodata,
    load_spatial_names,
)
from utils import page_header, tidy_plot

page_header(
    "County Pattern Explorer",
    "Compare attribute-based county grouping with connected spatial regionalisation on the same 3,143 counties.",
)

counties = load_county_data()
geojson = load_geodata()
conventional_profiles, spatial_profiles = load_county_profiles()
spatial_names = load_spatial_names()
spatial_name_map = spatial_names.set_index("spatial_cluster")["descriptive_name"].to_dict()

model_view = st.radio(
    "County model view", ["Spatial SCHC", "Conventional K-Means"], horizontal=True
)
if model_view == "Conventional K-Means":
    cluster_column = "conventional_county_cluster"
    label_column = "conventional_cluster_label"
    color_map = {f"County K-Means cluster {key}": value for key, value in COUNTY_CLUSTER_COLORS.items()}
    model_description = (
        "Conventional county clustering groups counties according to similarity in FPS rate, "
        "incident composition and contextual characteristics without requiring geographical proximity. "
        "These clusters are not geographic regions."
    )
else:
    cluster_column = "spatial_cluster"
    label_column = "spatial_cluster_label"
    color_map = {f"SCHC region {key}": value for key, value in SPATIAL_CLUSTER_COLORS.items()}
    model_description = (
        "Spatially constrained hierarchical clustering uses the same analytical characteristics "
        "while requiring each resulting region to remain geographically connected. Connectedness "
        "does not by itself make SCHC objectively superior."
    )
st.info(model_description)

st.subheader("Display filters")
filter_cols = st.columns(3)
with filter_cols[0]:
    selected_states = st.multiselect("State", sorted(counties["state_name"].unique()))
with filter_cols[1]:
    selected_clusters = st.multiselect(
        "Displayed cluster/region", sorted(counties[label_column].unique())
    )
with filter_cols[2]:
    fps_status = st.multiselect("FPS status", ["FPS-positive", "Zero FPS"])

display_counties = counties.copy()
if selected_states:
    display_counties = display_counties[display_counties["state_name"].isin(selected_states)]
if selected_clusters:
    display_counties = display_counties[display_counties[label_column].isin(selected_clusters)]
if fps_status:
    display_counties = display_counties[display_counties["fps_status"].isin(fps_status)]

st.caption(
    f"Displaying {len(display_counties):,} of 3,143 counties. Zero-FPS counties remain visible by default. "
    "Filters affect display only."
)

if display_counties.empty:
    st.warning("No counties match the selected display filters.")
else:
    map_frame = display_counties.copy()
    map_frame["poverty_display"] = 100 * map_frame["poverty_rate"]
    map_frame["unemployment_display"] = 100 * map_frame["unemployment_rate"]
    map_frame["bachelor_display"] = 100 * map_frame["bachelor_plus_rate"]
    map_frame["less_hs_display"] = 100 * map_frame["less_than_highschool_rate"]
    fig = px.choropleth_map(
        map_frame,
        geojson=geojson,
        locations="county_fips",
        featureidkey="properties.county_fips",
        color=label_column,
        color_discrete_map=color_map,
        map_style="carto-positron",
        center={"lat": 38.5, "lon": -96},
        zoom=2.45,
        opacity=0.72,
        hover_name="acs_name",
        hover_data={
            "county_fips": True,
            "state_name": True,
            "fps_count_2015_2024": ":,.0f",
            "avg_annual_fps_rate_per_100k": ":.3f",
            "conventional_cluster_label": True,
            "spatial_cluster_label": True,
            "median_income": ":$,.0f",
            "poverty_display": ":.1f",
            "unemployment_display": ":.1f",
            "bachelor_display": ":.1f",
            "less_hs_display": ":.1f",
            "population_density": ":,.1f",
            label_column: False,
        },
        labels={
            label_column: "Assignment",
            "county_fips": "County FIPS",
            "state_name": "State",
            "fps_count_2015_2024": "FPS count, 2015–2024",
            "avg_annual_fps_rate_per_100k": "Average annual FPS rate / 100k",
            "median_income": "Median income",
            "poverty_display": "Poverty (%)",
            "unemployment_display": "Unemployment (%)",
            "bachelor_display": "Bachelor+ (%)",
            "less_hs_display": "Less than high school (%)",
            "population_density": "Population density",
        },
        title=f"{model_view} assignments",
    )
    fig.update_layout(map_bounds={"west": -180, "east": -60, "south": 15, "north": 75})
    st.plotly_chart(tidy_plot(fig, 700), width="stretch")

st.subheader("County/region profile comparison")
profile_metrics = {
    "avg_annual_fps_rate_per_100k": "Average annual FPS rate per 100,000",
    "fps_count_2015_2024": "Raw FPS count, 2015–2024",
    "median_income": "Median household income ($)",
    "poverty_rate": "Poverty rate",
    "unemployment_rate": "Unemployment rate",
    "bachelor_plus_rate": "Bachelor+ attainment",
    "less_than_highschool_rate": "Less-than-high-school attainment",
    "population_density": "Population density",
    "profile_armed_with_gun": "Smoothed firearm composition",
    "profile_armed_with_knife": "Smoothed knife composition",
}
selected_metric = st.selectbox(
    "Profile characteristic", list(profile_metrics), format_func=lambda key: profile_metrics[key]
)
if model_view == "Conventional K-Means":
    profile_frame = conventional_profiles.copy()
    profile_frame["group"] = "County K-Means cluster " + profile_frame["conventional_county_cluster"].astype(int).astype(str)
else:
    profile_frame = spatial_profiles.copy()
    profile_frame["group"] = profile_frame["spatial_cluster"].astype(int).map(
        lambda cluster: f"SCHC region {cluster} — {spatial_name_map.get(cluster, 'descriptive region')}"
    )

is_rate = selected_metric in {
    "poverty_rate", "unemployment_rate", "bachelor_plus_rate",
    "less_than_highschool_rate", "profile_armed_with_gun", "profile_armed_with_knife",
}
profile_frame["display_value"] = profile_frame[selected_metric] * (100 if is_rate else 1)
unit = "%" if is_rate else "Original-unit mean"
fig = px.bar(
    profile_frame, x="group", y="display_value", color="group", text_auto=".3s",
    title=profile_metrics[selected_metric],
    labels={"group": "", "display_value": unit},
)
st.plotly_chart(tidy_plot(fig), width="stretch")

with st.expander("View the complete original-unit profile tables"):
    st.dataframe(profile_frame.drop(columns="display_value"), hide_index=True, width="stretch")

explain_left, explain_right = st.columns(2)
with explain_left:
    st.markdown(
        """
        <div class="study-card"><h4>K-Means asks</h4>
        <p>Which counties look similar based on the analytical characteristics?</p>
        <span class="small-note">Geographic proximity is not required.</span></div>
        """,
        unsafe_allow_html=True,
    )
with explain_right:
    st.markdown(
        """
        <div class="study-card"><h4>SCHC asks</h4>
        <p>Which geographically connected counties can be grouped while retaining similarity?</p>
        <span class="small-note">Attribute compactness may be traded for contiguity.</span></div>
        """,
        unsafe_allow_html=True,
    )

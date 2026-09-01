import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import (
    load_selected_model_summary,
    load_source_incidents,
    load_source_sensitive_aggregates,
    load_validation,
)
from utils import page_header, tidy_plot

page_header(
    "Fatal Police Shooting Pattern Explorer",
    "Interactive exploration of incident profiles and county-level spatial patterns in U.S. fatal police shootings, 2015–2024.",
)

validation = load_validation()
selected = load_selected_model_summary()
incidents = load_source_incidents()
sensitive_aggregates = load_source_sensitive_aggregates()

incident_model = selected.query("representative_role == 'incident_profile'").iloc[0]
county_model = selected.query("representative_role == 'conventional_county'").iloc[0]
spatial_model = selected.query("representative_role == 'spatial_county'").iloc[0]

st.subheader("Study at a glance")
kpi_columns = st.columns(4)
kpis = [
    ("Source incidents", f"{validation['source_incidents']:,}"),
    ("Modelling incidents", f"{validation['incident_modelling']:,}"),
    ("County equivalents", f"{validation['counties']:,}"),
    ("Zero-FPS counties", f"{validation['zero_fps_counties']:,}"),
    ("Study period", "2015–2024"),
    ("Incident representative", f"{incident_model['selected_algorithm']} {incident_model['configuration']}"),
    ("Conventional county", f"{county_model['selected_algorithm']} {county_model['configuration']}"),
    ("Spatial county", f"{spatial_model['selected_algorithm']} {spatial_model['configuration']}"),
]
for index, (label, value) in enumerate(kpis):
    kpi_columns[index % 4].metric(label, value)

st.subheader("Two-phase analytical design")
left, arrow, right = st.columns([5, 1, 5])
with left:
    st.markdown(
        """
        <div class="study-card">
        <h4>Phase 1 · Incident-level profiling</h4>
        <p>Individual incident characteristics and county context</p>
        <p><b>10,399 incidents → K-Means k=4 → four recurring profiles</b></p>
        <span class="small-note">Distant incidents may be grouped together when their attributes are similar.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with arrow:
    st.markdown("<div style='text-align:center;font-size:2rem;padding-top:2.5rem'>⇄</div>", unsafe_allow_html=True)
with right:
    st.markdown(
        """
        <div class="study-card">
        <h4>Phase 2 · County-level analysis</h4>
        <p>FPS rate, smoothed incident composition and county context</p>
        <p><b>3,143 counties → conventional grouping + connected regionalisation</b></p>
        <span class="small-note">Incident and county metrics are not placed in one universal ranking.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Descriptive incident explorer")
st.caption("Filters change the displayed descriptive records only. No model is refitted.")
filter_columns = st.columns(3)
with filter_columns[0]:
    years = st.multiselect("Year", sorted(incidents["year"].dropna().astype(int).unique()))
    states = st.multiselect("State", sorted(incidents["state"].dropna().astype(str).unique()))
with filter_columns[1]:
    armed = st.multiselect("Armed status", sorted(incidents["armed_with"].dropna().astype(str).unique()))
with filter_columns[2]:
    threats = st.multiselect("Threat type", sorted(incidents["threat_type"].dropna().astype(str).unique()))
    fleeing = st.multiselect("Fleeing status", sorted(incidents["flee_status"].dropna().astype(str).unique()))

filtered = incidents.copy()
for column, choices in [
    ("year", years), ("state", states), ("armed_with", armed),
    ("threat_type", threats), ("flee_status", fleeing),
]:
    if choices:
        filtered = filtered[filtered[column].isin(choices)]
st.info(f"Displaying {len(filtered):,} of {len(incidents):,} source incidents.")

def count_chart(column: str, title: str, horizontal: bool = False):
    counts = filtered[column].fillna("Unknown").astype(str).value_counts().rename_axis(column).reset_index(name="count")
    if horizontal:
        fig = px.bar(counts.sort_values("count"), x="count", y=column, orientation="h", title=title, text="count")
    else:
        fig = px.bar(counts, x=column, y="count", title=title, text="count")
    fig.update_traces(marker_color="#2563EB")
    return tidy_plot(fig)

row1 = st.columns(3)
with row1[0]:
    st.plotly_chart(count_chart("year", "Incidents by year"), width="stretch")
with row1[1]:
    race_counts = sensitive_aggregates.query("variable == 'race'").sort_values("count")
    fig = px.bar(
        race_counts, x="count", y="category", orientation="h", text="count",
        title="Incidents by race (overall aggregate)",
        labels={"count": "Incidents", "category": "Race"},
    )
    fig.update_traces(marker_color="#2563EB")
    st.plotly_chart(tidy_plot(fig), width="stretch")
with row1[2]:
    st.plotly_chart(count_chart("armed_with", "Armed-status composition", horizontal=True), width="stretch")

row2 = st.columns(3)
with row2[0]:
    st.plotly_chart(count_chart("threat_type", "Threat-type composition", horizontal=True), width="stretch")
with row2[1]:
    st.plotly_chart(count_chart("flee_status", "Fleeing-status composition", horizontal=True), width="stretch")
with row2[2]:
    body_counts = (
        filtered["body_camera"].astype(str).value_counts().rename_axis("recorded")
        .reset_index(name="count").assign(indicator="Body camera (filtered)")
    )
    mental_counts = (
        sensitive_aggregates.query("variable == 'was_mental_illness_related'")
        .rename(columns={"category": "recorded"})[["recorded", "count"]]
        .assign(indicator="Mental illness (overall aggregate)")
    )
    binary_counts = pd.concat([body_counts, mental_counts], ignore_index=True)
    fig = px.bar(
        binary_counts, x="indicator", y="count", color="recorded", barmode="stack",
        title="Recorded body-camera and mental-illness indicators",
        labels={"count": "Incidents", "indicator": "Indicator", "recorded": "Recorded value"},
    )
    st.plotly_chart(tidy_plot(fig), width="stretch")

st.caption(
    "Race and mental-illness values are deployed only as overall aggregates; other charts respond to the display filters. "
    "These records cover fatal police shootings rather than all police encounters."
)

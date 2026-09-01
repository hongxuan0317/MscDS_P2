import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import INCIDENT_CLUSTER_COLORS
from data_loader import (
    load_incident_composition,
    load_incident_modelling_data,
    load_incident_pca,
    load_incident_profiles,
)
from utils import page_header, tidy_plot

page_header(
    "Incident Profile Explorer",
    "Explore the four existing K-Means incident profiles fitted to 10,399 modelling records.",
)

profiles = load_incident_profiles()
composition = load_incident_composition()
incident_data = load_incident_modelling_data()
pca_coordinates, pca_variance = load_incident_pca()

profile_options = {
    int(row["cluster"]): f"Cluster {int(row['cluster'])} — {row['descriptive_profile']}"
    for _, row in profiles.iterrows()
}
selected_cluster = st.selectbox(
    "Select an existing profile", options=list(profile_options),
    format_func=lambda value: profile_options[value],
)
profile = profiles.set_index("cluster").loc[selected_cluster]

st.subheader(profile_options[selected_cluster])
age_rows = composition.query(
    "variable == 'age_group'"
)
age_rows = age_rows[
    age_rows["cluster"].astype(str) == str(selected_cluster)
].sort_values("proportion", ascending=False)
age_characteristic = age_rows.iloc[0]["category"] if len(age_rows) else "Not available"

kpi_cols = st.columns(5)
kpis = [
    ("Incidents", f"{int(profile['incidents']):,}"),
    ("Sample share", f"{profile['share_pct']:.1f}%"),
    ("Largest age group", age_characteristic),
    ("Firearm", f"{profile['gun_pct']:.1f}%"),
    ("Knife", f"{profile['knife_pct']:.1f}%"),
    ("Mental-illness indicator", f"{profile['mental_illness_pct']:.1f}%"),
    ("Body camera", f"{profile['body_camera_pct']:.1f}%"),
    ("Mean county income", f"${profile['median_income']:,.0f}"),
    ("Mean county poverty", f"{profile['poverty_pct']:.1f}%"),
    ("Mean population density", f"{profile['population_density']:,.0f}"),
]
for index, (label, value) in enumerate(kpis):
    kpi_cols[index % 5].metric(label, value)

left, right = st.columns([1, 1])
with left:
    st.subheader("Incident composition")
    variable_labels = {
        "age_group": "Age group", "gender": "Gender", "race": "Race",
        "armed_with": "Armed status", "threat_type": "Threat type",
        "flee_status": "Fleeing status",
        "was_mental_illness_related": "Mental-illness indicator",
        "body_camera": "Body-camera indicator",
    }
    variable = st.selectbox(
        "Characteristic", list(variable_labels), format_func=lambda item: variable_labels[item]
    )
    comparison = composition[
        (composition["variable"] == variable)
        & (composition["cluster"].astype(str).isin([str(selected_cluster), "Overall"]))
    ].copy()
    comparison["group"] = np.where(
        comparison["cluster"].astype(str) == "Overall", "Overall sample", f"Cluster {selected_cluster}"
    )
    comparison["percent"] = 100 * comparison["proportion"]
    fig = px.bar(
        comparison, x="category", y="percent", color="group", barmode="group",
        title=f"{variable_labels[variable]}: selected cluster vs overall",
        labels={"category": "Category", "percent": "Within-group share (%)", "group": ""},
        color_discrete_map={f"Cluster {selected_cluster}": INCIDENT_CLUSTER_COLORS[str(selected_cluster)], "Overall sample": "#94A3B8"},
    )
    st.plotly_chart(tidy_plot(fig), width="stretch")

with right:
    st.subheader("Context associated with profile incidents")
    context_metrics = {
        "median_income": "Median income ($)",
        "poverty_pct": "Poverty (%)",
        "unemployment_pct": "Unemployment (%)",
        "bachelor_plus_pct": "Bachelor+ (%)",
        "less_than_highschool_pct": "Less than high school (%)",
        "population_density": "Population density",
    }
    context = profiles[["cluster", *context_metrics]].melt(
        id_vars="cluster", var_name="metric", value_name="value"
    )
    context["metric_label"] = context["metric"].map(context_metrics)
    context["cluster"] = "Cluster " + context["cluster"].astype(int).astype(str)
    selected_context = context[context["cluster"] == f"Cluster {selected_cluster}"]
    fig = px.bar(
        selected_context, x="value", y="metric_label", orientation="h",
        title="Original-unit profile means", text_auto=".3s",
        labels={"value": "Mean", "metric_label": ""},
    )
    fig.update_traces(marker_color=INCIDENT_CLUSTER_COLORS[str(selected_cluster)])
    st.plotly_chart(tidy_plot(fig), width="stretch")

st.info(
    "County characteristics describe the contexts associated with incidents in a profile. "
    "They must not be assigned to individuals and do not demonstrate causal effects."
)

st.subheader("Post-hoc PCA visualisation")
pc1 = pca_variance.query("component == 'PC1'").iloc[0]["explained_variance_pct"]
pc2 = pca_variance.query("component == 'PC2'").iloc[0]["explained_variance_pct"]
cumulative = pca_variance.query("component == 'PC1+PC2 cumulative'").iloc[0]["explained_variance_pct"]
st.caption(
    f"PC1 explains {pc1:.1f}%, PC2 {pc2:.1f}%, cumulative {cumulative:.1f}%. "
    "PCA is used only for two-dimensional visualisation; K-Means was fitted using all 52 features."
)
display_mode = st.radio("PCA display", ["All profiles", "Highlight selected profile"], horizontal=True)
pca_plot = pca_coordinates.copy()
pca_plot["cluster_label"] = "Cluster " + pca_plot["primary_cluster"].astype(int).astype(str)
if display_mode == "Highlight selected profile":
    pca_plot["display_group"] = np.where(
        pca_plot["primary_cluster"] == selected_cluster,
        f"Cluster {selected_cluster}", "Other profiles",
    )
    color = "display_group"
    color_map = {f"Cluster {selected_cluster}": INCIDENT_CLUSTER_COLORS[str(selected_cluster)], "Other profiles": "#CBD5E1"}
else:
    color = "cluster_label"
    color_map = {f"Cluster {key}": value for key, value in INCIDENT_CLUSTER_COLORS.items()}
fig = px.scatter(
    pca_plot, x="PC1", y="PC2", color=color, hover_data=["profile_name"],
    render_mode="webgl", opacity=0.55, color_discrete_map=color_map,
    title="Incident profiles projected onto the first two principal components",
)
st.plotly_chart(tidy_plot(fig, 650), width="stretch")

with st.expander("Compare all four original-unit profile summaries"):
    display_columns = [
        "cluster", "descriptive_profile", "incidents", "share_pct", "gun_pct", "knife_pct",
        "mental_illness_pct", "body_camera_pct", "median_income", "poverty_pct",
        "unemployment_pct", "population_density",
    ]
    st.dataframe(profiles[display_columns], hide_index=True, width="stretch")

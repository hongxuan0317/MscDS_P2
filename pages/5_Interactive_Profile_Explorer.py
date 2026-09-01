import pandas as pd
import plotly.express as px
import streamlit as st

from config import INCIDENT_CLUSTER_COLORS
from data_loader import (
    load_context_reference_values,
    load_county_data,
    load_feature_reference,
    load_feature_scope,
    load_incident_composition,
    load_incident_modelling_data,
    load_incident_profiles,
    load_model_artifacts,
)
from utils import (
    centroid_similarity,
    create_scenario_vector,
    neutral_context_values,
    page_header,
    tidy_plot,
)

CONTEXT_VARIABLES = [
    "median_income", "poverty_rate", "unemployment_rate", "gini_index",
    "bachelor_plus_rate", "less_than_highschool_rate", "population_density", "land_area_sqmi",
]
CONTEXT_LABELS = {
    "median_income": "Median household income",
    "poverty_rate": "Poverty rate",
    "unemployment_rate": "Unemployment rate",
    "gini_index": "Income inequality (Gini)",
    "bachelor_plus_rate": "Bachelor's attainment",
    "less_than_highschool_rate": "Less-than-high-school attainment",
    "population_density": "Population density",
    "land_area_sqmi": "Land area",
}


def format_context(variable: str, value: float) -> str:
    if variable == "median_income":
        return f"${value:,.0f}"
    if variable in {
        "poverty_rate", "unemployment_rate", "bachelor_plus_rate", "less_than_highschool_rate",
    }:
        return f"{100 * value:.1f}%"
    if variable == "gini_index":
        return f"{value:.3f}"
    if variable == "population_density":
        return f"{value:,.1f} people/sq mi"
    return f"{value:,.1f} sq mi"


page_header(
    "Interactive Incident Profile Explorer",
    "Explore how incident characteristics relate to the four descriptive profiles identified by the locked K-Means model.",
)
st.error(
    "This tool demonstrates similarity to descriptive clusters only. It does not estimate individual "
    "risk, probability, causation, future outcomes, or recommended action."
)

incidents = load_incident_modelling_data()
profiles = load_incident_profiles()
composition = load_incident_composition()
profile_names = profiles.set_index("cluster")["descriptive_profile"].to_dict()

mode = st.radio(
    "Explorer mode", ["Existing anonymized incident", "Hypothetical situational scenario"], horizontal=True,
)

if mode == "Existing anonymized incident":
    st.subheader("Existing incident explorer")
    st.caption(
        "Assignments are the existing K-Means k=4 labels. Names, source identifiers, addresses, "
        "coordinates and row-level sensitive characteristics are excluded from the deployed data."
    )
    filter_cols = st.columns(3)
    with filter_cols[0]:
        year = st.multiselect("Year", sorted(incidents["year"].dropna().astype(int).unique()))
        state = st.multiselect("State", sorted(incidents["state"].dropna().unique()))
    with filter_cols[1]:
        armed = st.multiselect("Armed status", sorted(incidents["armed_with"].dropna().unique()))
        threat = st.multiselect("Threat type", sorted(incidents["threat_type"].dropna().unique()))
    with filter_cols[2]:
        flee = st.multiselect("Fleeing status", sorted(incidents["flee_status"].dropna().unique()))
        body = st.multiselect("Body camera", sorted(incidents["body_camera"].astype(str).unique()))

    filtered = incidents.copy()
    for column, choices in [
        ("year", year), ("state", state), ("armed_with", armed),
        ("threat_type", threat), ("flee_status", flee),
    ]:
        if choices:
            filtered = filtered[filtered[column].isin(choices)]
    if body:
        filtered = filtered[filtered["body_camera"].astype(str).isin(body)]

    st.caption(f"{len(filtered):,} existing anonymized records match the display filters.")
    if filtered.empty:
        st.warning("No existing incident matches these filters.")
    else:
        selected_record = st.selectbox("Select anonymized record", filtered["display_record"].tolist())
        record = filtered.set_index("display_record").loc[selected_record]
        cluster = int(record["primary_cluster"])
        st.success(f"Existing assignment: Cluster {cluster} — {profile_names[cluster]}")
        cols = st.columns(4)
        existing_values = [
            ("Year / state", f"{int(record['year'])} · {record['state']}"),
            ("Armed / threat", f"{record['armed_with']} · {record['threat_type']}"),
            ("Fleeing / body camera", f"{record['flee_status']} · {record['body_camera']}"),
            ("County income context", f"${record['median_income']:,.0f}"),
            ("County poverty context", f"{100 * record['poverty_rate']:.1f}%"),
            ("County unemployment", f"{100 * record['unemployment_rate']:.1f}%"),
            ("Population density", f"{record['population_density']:,.1f}"),
            ("Land area", f"{record['land_area_sqmi']:,.1f} sq mi"),
        ]
        for index, (label, value) in enumerate(existing_values):
            cols[index % 4].metric(label, value)

        cluster_profile = profiles.set_index("cluster").loc[cluster]
        comparison = pd.DataFrame([
            {"Characteristic": "Median income ($)", "Selected incident context": record["median_income"], "Cluster mean": cluster_profile["median_income"]},
            {"Characteristic": "Poverty (%)", "Selected incident context": 100 * record["poverty_rate"], "Cluster mean": cluster_profile["poverty_pct"]},
            {"Characteristic": "Unemployment (%)", "Selected incident context": 100 * record["unemployment_rate"], "Cluster mean": cluster_profile["unemployment_pct"]},
            {"Characteristic": "Population density", "Selected incident context": record["population_density"], "Cluster mean": cluster_profile["population_density"]},
        ])
        st.markdown("#### Selected incident context compared with assigned profile")
        st.dataframe(comparison, hide_index=True, width="stretch")

else:
    st.subheader("Hypothetical situational scenario matcher")
    st.caption(
        "Only non-sensitive situational and county-context fields can be changed. The context describes "
        "a place, not an individual. Personal and demographic dimensions remain neutral."
    )
    model, preprocessor = load_model_artifacts()
    dictionary, reference = load_feature_reference()
    context_references = load_context_reference_values()
    counties = load_county_data()
    levels = preprocessor["category_levels"]

    st.markdown("#### 1. Situational information")
    input_cols = st.columns(4)
    with input_cols[0]:
        armed_value = st.selectbox("Armed status", levels["armed_with"])
    with input_cols[1]:
        threat_value = st.selectbox("Threat type", levels["threat_type"])
    with input_cols[2]:
        flee_value = st.selectbox("Fleeing status", levels["flee_status"])
    with input_cols[3]:
        body_value = st.selectbox("Body camera", [False, True])

    st.markdown("#### 2. County context")
    context_mode = st.radio(
        "Context mode", ["Neutral average context", "Select an existing county", "Customize context"],
        horizontal=True,
    )
    context_values = neutral_context_values(preprocessor, CONTEXT_VARIABLES)
    context_description = "Overall modelling-population reference"

    if context_mode == "Neutral average context":
        st.info(
            "All county-context dimensions use the overall modelling-space reference. For log-transformed "
            "density and land area, the displayed value is the inverse-transformed reference."
        )
    elif context_mode == "Select an existing county":
        county_cols = st.columns([1, 2])
        with county_cols[0]:
            county_state = st.selectbox("State", sorted(counties["state_name"].unique()), key="context_state")
        county_options = counties[counties["state_name"] == county_state].copy()
        county_options["display_name"] = county_options["acs_name"] + " · FIPS " + county_options["county_fips"]
        with county_cols[1]:
            selected_county_name = st.selectbox("County", county_options["display_name"].tolist())
        county_row = county_options.set_index("display_name").loc[selected_county_name]
        context_values = {variable: float(county_row[variable]) for variable in CONTEXT_VARIABLES}
        context_description = selected_county_name
        st.caption(
            "The selected county supplies only compatible contextual fields. It does not supply or infer "
            "personal characteristics."
        )
    else:
        custom_cols = st.columns(2)
        context_values = {}
        for index, variable in enumerate(CONTEXT_VARIABLES):
            choices = context_references[context_references["variable"] == variable].copy()
            labels = choices["selection_label"].tolist()
            with custom_cols[index % 2]:
                selected_level = st.selectbox(
                    CONTEXT_LABELS[variable], labels, index=1, key=f"context_{variable}",
                )
                raw_value = float(
                    choices.loc[choices["selection_label"] == selected_level, "raw_value"].iloc[0]
                )
                context_values[variable] = raw_value
                st.caption(f"Value used: {format_context(variable, raw_value)}")
        context_description = "Customized observed-percentile context"

    scenario = create_scenario_vector(
        dictionary, reference, armed_value, threat_value, flee_value, body_value,
        context_values=context_values, preprocessor=preprocessor,
    )
    result = centroid_similarity(scenario, model.cluster_centers_)
    result["profile_name"] = result["cluster"].map(profile_names)
    result["cluster_label"] = "Profile " + result["cluster"].astype(str)
    closest = result.iloc[0]
    closest_cluster = int(closest["cluster"])
    closest_profile = profiles.set_index("cluster").loc[closest_cluster]

    st.markdown("#### Scenario information used for matching")
    scenario_rows = [
        {"Section": "Situational", "Characteristic": "Armed status", "Value used": str(armed_value)},
        {"Section": "Situational", "Characteristic": "Threat type", "Value used": str(threat_value)},
        {"Section": "Situational", "Characteristic": "Fleeing status", "Value used": str(flee_value)},
        {"Section": "Situational", "Characteristic": "Body camera", "Value used": str(body_value)},
    ]
    scenario_rows.extend(
        {"Section": "County context", "Characteristic": CONTEXT_LABELS[variable], "Value used": format_context(variable, value)}
        for variable, value in context_values.items()
    )
    st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, width="stretch")
    st.caption(
        f"Context source: {context_description}. Other modelling dimensions remain at the overall "
        "modelling-population reference value."
    )

    st.success(f"Closest descriptive profile: Cluster {closest_cluster} — {profile_names[closest_cluster]}")
    result_sorted = result.sort_values("cluster")
    fig = px.bar(
        result_sorted, x="cluster_label", y="profile_similarity", color="cluster_label",
        color_discrete_map={f"Profile {key}": value for key, value in INCIDENT_CLUSTER_COLORS.items()},
        text_auto=".1f", hover_data={"distance": ":.3f", "profile_name": True},
        labels={"cluster_label": "Existing profile", "profile_similarity": "Profile Similarity"},
        title="Relative similarity to the four existing centroids",
    )
    st.plotly_chart(tidy_plot(fig), width="stretch")
    st.caption(
        "Profile Similarity is a normalized inverse-distance display derived from distances to the existing "
        "centroids. It is not a statistical probability, confidence measure, or individual score."
    )

    kpis = st.columns(4)
    for col, (label, value) in zip(kpis, [
        ("Closest cluster", str(closest_cluster)),
        ("Existing incidents", f"{int(closest_profile['incidents']):,}"),
        ("Modelling-sample share", f"{closest_profile['share_pct']:.1f}%"),
        ("Centroid distance", f"{closest['distance']:.3f}"),
    ]):
        col.metric(label, value)

    st.markdown("#### Why this profile?")
    selected_characteristics = {
        "armed_with": armed_value, "threat_type": threat_value,
        "flee_status": flee_value, "body_camera": str(body_value),
    }
    explanations = []
    for variable, category in selected_characteristics.items():
        rows = composition[
            (composition["variable"] == variable)
            & (composition["category"].astype(str) == str(category))
        ]
        cluster_row = rows[rows["cluster"].astype(str) == str(closest_cluster)]
        overall_row = rows[rows["cluster"].astype(str) == "Overall"]
        if len(cluster_row) and len(overall_row):
            cluster_share = float(cluster_row.iloc[0]["proportion"])
            overall_share = float(overall_row.iloc[0]["proportion"])
            explanations.append({
                "Characteristic": f"{variable.replace('_', ' ').title()}: {category}",
                "Closest-profile share": cluster_share,
                "Overall share": overall_share,
                "Difference": cluster_share - overall_share,
            })
    explanation_frame = pd.DataFrame(explanations)
    if len(explanation_frame):
        explanation_frame = explanation_frame.sort_values("Difference", ascending=False)
        st.dataframe(
            explanation_frame.style.format({
                "Closest-profile share": "{:.1%}", "Overall share": "{:.1%}", "Difference": "{:+.1%}",
            }),
            hide_index=True, width="stretch",
        )
        positive = explanation_frame.iloc[0]
        st.write(
            f"Among the selected situational fields, **{positive['Characteristic']}** is most positively "
            f"aligned with Profile {closest_cluster} relative to the overall modelling sample."
        )

    context_comparison = pd.DataFrame([
        {
            "County-context characteristic": CONTEXT_LABELS[variable],
            "Scenario value": format_context(variable, context_values[variable]),
            "Closest-profile mean": format_context(variable, float(closest_profile[variable])),
        }
        for variable in CONTEXT_VARIABLES
    ])
    st.markdown("##### Scenario context and closest-profile mean")
    st.dataframe(context_comparison, hide_index=True, width="stretch")
    st.caption(
        "The closest profile is determined by the complete 52-dimensional centroid distance, not by any "
        "single row in these comparison tables."
    )

    with st.expander("Exact matching methodology and feature-scope restrictions"):
        st.markdown(
            """
            1. The saved K-Means k=4 centroids and preprocessing metadata are loaded from local dashboard artefacts; no model is trained or updated.
            2. The neutral starting vector is the mean of all 10,399 records in the exact 52-feature modelling space.
            3. Armed status, threat type and fleeing status replace only their original one-hot blocks using the original `1/sqrt(2)` encoding; body camera replaces its existing binary feature.
            4. Selected context values use the original StandardScaler. Population density and land area receive the original `log1p` transformation first.
            5. All other dimensions remain at the modelling-population reference, after which Euclidean distances are calculated to the four saved centroids.

            Hypothetical race, gender, age and mental-health inputs are unavailable. County Black, Hispanic and youth composition also remain neutral. The context controls describe county conditions only.
            """
        )
        scope = load_feature_scope()
        dimensions = dictionary.groupby("source_variable").size().rename("Model dimensions")
        scope_display = scope.merge(dimensions, on="source_variable")
        scope_display = scope_display.groupby("conceptual_scope").agg(
            **{
                "Conceptual variables": ("source_variable", lambda values: ", ".join(values)),
                "Model dimensions": ("Model dimensions", "sum"),
            }
        ).reset_index().rename(columns={"conceptual_scope": "Feature group"})
        st.dataframe(scope_display, hide_index=True, width="stretch")

st.subheader("Four-profile comparison")
comparison_columns = [
    "cluster", "descriptive_profile", "incidents", "share_pct", "gun_pct", "knife_pct",
    "shoot_threat_pct", "not_fleeing_pct", "mental_illness_pct", "body_camera_pct",
    "median_income", "poverty_pct", "unemployment_pct", "population_density",
]
st.dataframe(profiles[comparison_columns], hide_index=True, width="stretch")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import (
    load_county_model_metrics,
    load_incident_model_metrics,
    load_sensitivity_tables,
    load_spatial_model_metrics,
)
from utils import page_header, tidy_plot

page_header(
    "Model Evaluation",
    "Review representative selection within each analytical family—never as one universal leaderboard.",
)

incident_selected, incident_candidates = load_incident_model_metrics()
county_metrics = load_county_model_metrics()
spatial_metrics = load_spatial_model_metrics()
sensitivity_tables = load_sensitivity_tables()

st.warning(
    "Incident and county models operate on different observations and feature spaces. "
    "Their metric values should not be ranked against each other."
)

incident_tab, county_tab, spatial_tab = st.tabs([
    "A · Incident clustering", "B · Conventional county", "C · Spatial regionalisation"
])

with incident_tab:
    st.subheader("Incident K-Means, AHC and HDBSCAN")
    primary_representatives = incident_selected.query("variant == 'primary'").copy()
    selected_kmeans = primary_representatives.query("algorithm == 'KMeans'").iloc[0]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, [
        ("Representative", f"KMeans {selected_kmeans['configuration']}"),
        ("Silhouette", f"{selected_kmeans['silhouette']:.3f}"),
        ("Davies–Bouldin", f"{selected_kmeans['davies_bouldin']:.3f}"),
        ("Calinski–Harabasz", f"{selected_kmeans['calinski_harabasz']:.1f}"),
        ("Resampling ARI / NMI", f"{selected_kmeans['resample_ari_mean']:.3f} / {selected_kmeans['resample_nmi_mean']:.3f}"),
    ]):
        col.metric(label, value)

    metric = st.selectbox(
        "Incident candidate metric",
        ["silhouette", "davies_bouldin", "calinski_harabasz"],
        format_func=lambda key: {
            "silhouette": "Silhouette (higher)", "davies_bouldin": "Davies–Bouldin (lower)",
            "calinski_harabasz": "Calinski–Harabasz (higher)",
        }[key],
        key="incident_metric",
    )
    k_candidates = incident_candidates.query(
        "variant == 'primary' and algorithm in ['KMeans', 'AHC']"
    )
    fig = px.line(
        k_candidates, x="k", y=metric, color="algorithm", markers=True,
        title="Incident k-grid candidates", labels={"k": "Number of clusters", metric: metric.replace("_", " ").title()},
    )
    st.plotly_chart(tidy_plot(fig), width="stretch")

    st.markdown("#### Representative algorithm comparison")
    display_columns = [
        "algorithm", "configuration", "silhouette", "davies_bouldin",
        "calinski_harabasz", "resample_ari_mean", "resample_nmi_mean",
        "noise_share", "min_cluster_share", "cluster_persistence", "eligible",
    ]
    st.dataframe(primary_representatives[display_columns], hide_index=True, width="stretch")
    hdbscan_row = primary_representatives.query("algorithm == 'HDBSCAN'").iloc[0]
    st.info(
        f"The diagnostic HDBSCAN representative reports silhouette {hdbscan_row['silhouette']:.3f} "
        f"on non-noise observations while classifying {100*hdbscan_row['noise_share']:.1f}% as noise. "
        "Its noise-excluded internal metric is therefore shown together with coverage and does not establish superiority."
    )

with county_tab:
    st.subheader("County K-Means and County AHC")
    representative = county_metrics.query("algorithm == 'County_KMeans' and k == 2").iloc[0]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, [
        ("Representative", "County K-Means k=2"),
        ("Silhouette", f"{representative['silhouette']:.3f}"),
        ("Davies–Bouldin", f"{representative['davies_bouldin']:.3f}"),
        ("Resampling ARI / NMI", f"{representative['resample_ari_mean']:.3f} / {representative['resample_nmi_mean']:.3f}"),
        ("Boundary / components", f"{representative['boundary_cut_ratio']:.3f} / {int(representative['connected_components_across_labels'])}"),
    ]):
        col.metric(label, value)
    metric = st.selectbox(
        "Conventional county metric",
        ["silhouette", "davies_bouldin", "calinski_harabasz", "resample_ari_mean", "cluster_size_cv"],
        key="county_metric",
    )
    fig = px.line(
        county_metrics, x="k", y=metric, color="algorithm", markers=True,
        title="Conventional county candidates on identical X_county",
    )
    st.plotly_chart(tidy_plot(fig), width="stretch")
    st.dataframe(
        county_metrics[[
            "algorithm", "k", "silhouette", "davies_bouldin", "calinski_harabasz",
            "resample_ari_mean", "resample_nmi_mean", "min_cluster_count",
            "cluster_size_cv", "boundary_cut_ratio", "connected_components_across_labels", "eligible",
        ]],
        hide_index=True, width="stretch",
    )
    st.caption(
        "Connectedness and boundary metrics are descriptive for conventional algorithms because "
        "these algorithms were not required to produce contiguous regions."
    )

with spatial_tab:
    st.subheader("SCHC and SKATER")
    representative = spatial_metrics.query("algorithm == 'SCHC' and k == 2").iloc[0]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, [
        ("Representative", "SCHC k=2"),
        ("Silhouette", f"{representative['silhouette']:.3f}"),
        ("Davies–Bouldin", f"{representative['davies_bouldin']:.3f}"),
        ("Calinski–Harabasz", f"{representative['calinski_harabasz']:.1f}"),
        ("Boundary-cut ratio", f"{representative['boundary_cut_ratio']:.3f}"),
    ]):
        col.metric(label, value)
    metric = st.selectbox(
        "Spatial candidate metric",
        ["silhouette", "davies_bouldin", "calinski_harabasz", "cluster_size_cv", "boundary_cut_ratio"],
        key="spatial_metric",
    )
    eligible_spatial = spatial_metrics.query("eligible == True")
    fig = px.line(
        eligible_spatial, x="k", y=metric, color="algorithm", markers=True,
        title="Eligible spatial candidates",
    )
    st.plotly_chart(tidy_plot(fig), width="stretch")
    st.dataframe(
        spatial_metrics[[
            "algorithm", "k", "silhouette", "davies_bouldin", "calinski_harabasz",
            "min_cluster_count", "cluster_size_cv", "boundary_cut_ratio",
            "labels_connected", "smoothing_agreement_ari_mean", "eligible",
        ]],
        hide_index=True, width="stretch",
    )
    st.caption(
        "Connectedness is an explicit algorithmic constraint. Evaluation must also consider attribute "
        "separation, region balance, boundary coherence, interpretation and sensitivity."
    )

st.subheader("Metric explanations")
metric_explanations = {
    "Silhouette": "Whether observations are closer to their own cluster than neighbouring clusters; higher is generally better.",
    "Davies–Bouldin Index": "Within-cluster scatter relative to separation; lower is generally better.",
    "Calinski–Harabasz Index": "Between-cluster dispersion relative to within-cluster dispersion; higher is generally better.",
    "ARI": "Chance-adjusted agreement between assignments; higher means stronger reproducibility.",
    "NMI": "Shared information between assignments; higher means stronger agreement.",
    "Boundary-cut ratio": "Share of neighbouring-county edges crossing labels; lower means smoother geographic grouping.",
    "Connected components": "Number of geographically disconnected pieces across cluster labels.",
}
metric_cols = st.columns(2)
for index, (name, explanation) in enumerate(metric_explanations.items()):
    with metric_cols[index % 2]:
        st.markdown(f"**{name}**  \n{explanation}")

with st.expander("Primary sensitivity results"):
    for name, frame in sensitivity_tables.items():
        st.markdown(f"**{name}**")
        st.dataframe(frame, hide_index=True, width="stretch")

import streamlit as st

from data_loader import load_validation
from utils import inject_css

st.set_page_config(
    page_title="Fatal Police Shooting Pattern Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

try:
    validation = load_validation()
except Exception as exc:
    st.error(f"Validated dashboard data could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("### Fatal Police Shooting\nPattern Explorer")
    st.caption("Yah Hong Xuan U2103393 Master in Data Science")
    with st.expander("About and methodology", expanded=False):
        st.markdown(
            """
            **Phase 1 — incident profiling**  
            10,399 incidents · K-Means k=4 representative

            **Phase 2 — county analysis**  
            3,143 counties · County K-Means k=2 conventional representative ·
            SCHC k=2 spatial representative

            - **K-Means:** groups observations by similarity to centroids.
            - **AHC:** builds hierarchical groups using Ward linkage.
            - **HDBSCAN:** finds density-based groups and may label noise.
            - **SCHC:** hierarchical clustering with spatial connectivity.
            - **SKATER:** connected regions from a minimum spanning tree.

            Cluster numbers and colours have no ordinal meaning. County variables
            are contextual, clustering is descriptive, and the results do not
            demonstrate causal effects.
            """
        )
    st.caption(
        f"Validated: {validation['incident_modelling']:,} modelling incidents · "
        f"{validation['counties']:,} counties"
    )

pages = [
    st.Page("pages/1_Overview.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/2_Incident_Profile_Explorer.py", title="Incident Profile Explorer", icon="🧭"),
    st.Page("pages/3_County_Pattern_Explorer.py", title="County Pattern Explorer", icon="🗺️"),
    st.Page("pages/4_Model_Evaluation.py", title="Model Evaluation", icon="📈"),
    st.Page("pages/5_Interactive_Profile_Explorer.py", title="Interactive Profile Explorer", icon="🔎"),
]
navigation = st.navigation(pages)
navigation.run()

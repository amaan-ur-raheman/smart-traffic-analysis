"""Page 9 — Principal component analysis and clustering.

Both analyses are expensive and run on reproducible samples, so neither is
triggered automatically. Each is started from an explicit button, and the sample
used is disclosed on the results.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from smart_traffic import clustering
from smart_traffic import dimensionality_reduction as pca_module
from smart_traffic.config import SETTINGS
from smart_traffic.ui import (
    charts_row,
    download_table,
    global_filters,
    info_box,
    page_header,
    require_dataset,
    sample_disclosure,
    section,
    sidebar_header,
)
from smart_traffic.visualizations import explained_variance_chart, scatter


def _linkage_matrix(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Build a SciPy linkage matrix from a flat cluster assignment.

    The dendrogram is reconstructed from the clustering the model produced
    rather than by refitting on the distance matrix, which keeps the drawn tree
    exactly consistent with the profile shown above it.

    Args:
        labels: Flat cluster assignment per row.
        n_clusters: Number of clusters.

    Returns:
        A ``scipy.cluster.hierarchy`` linkage matrix.
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    condensed: list[float] = []
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            condensed.append(0.0 if first == second else 1.0)
    return scipy_linkage(np.asarray(condensed), method="single")


st.set_page_config(page_title="PCA & Clustering", page_icon="🔬", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "PCA and Clustering",
    "Unsupervised structure in the recorded variables. Both analyses run on "
    "reproducible samples and are started explicitly.",
    icon="🔬",
)

spec = global_filters(show_cities=False)

NUMERIC_OPTIONS = [
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
    "Hour", "Month", "Is_Weekend", "Is_Rush_Hour", "Is_Night",
]

selected = st.multiselect(
    "Numeric variables to analyse",
    NUMERIC_OPTIONS,
    default=list(SETTINGS.numeric_analysis_features),
    help="Variables are standardised before analysis. Removing a variable can "
         "change the result substantially, because PCA and clustering are "
         "sensitive to what is included.",
)

if len(selected) < 2:
    st.warning("Select at least two numeric variables to proceed.", icon="⚠️")
    st.stop()

tab_pca, tab_kmeans, tab_hierarchical = st.tabs(
    ["Principal Component Analysis", "K-Means clustering", "Hierarchical clustering"]
)

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
with tab_pca:
    section(
        "Principal component analysis",
        "PCA rotates the feature space to find the directions of greatest variance. "
        "Variables are standardised first, otherwise a feature measured in the tens "
        "(pressure) would dominate one measured in the single digits (humidity).",
    )

    pca_controls = st.columns([1, 1, 2])
    n_components = pca_controls[0].slider(
        "Components to compute", 2, min(len(selected), 10), 2,
    )
    pca_sample = pca_controls[1].number_input(
        "Sample size", 1_000, 500_000, min(SETTINGS.sample_scatter_points, 100_000),
        step=1_000,
        help="PCA is O(n × p) in memory. Larger samples are more stable but slower.",
    )

    if st.button("Run PCA", type="primary", key="run_pca"):
        with st.spinner(f"Fitting PCA on up to {pca_sample:,} rows..."):
            try:
                st.session_state["pca_result"] = pca_module.run_pca(
                    selected, spec, n_components=n_components, sample_limit=pca_sample
                )
            except ValueError as exc:
                st.error(str(exc))
                st.session_state.pop("pca_result", None)
        st.rerun()

    result = st.session_state.get("pca_result")

    if result is None:
        st.info(
            "PCA has not been run for the current selection. Choose the variables "
            "and sample size, then press **Run PCA**."
        )
    else:
        sample_disclosure(result.sample_note)

        charts_row(
            [
                explained_variance_chart(
                    result.explained_variance,
                    title="Variance explained by each component",
                ),
                scatter(
                    result.coordinates,
                    "PC1", "PC2",
                    color="Severity" if "Severity" in result.coordinates.columns else None,
                    title="Observations on the first two components",
                    x_title="PC1 score", y_title="PC2 score",
                    opacity=0.35,
                ),
            ]
        )

        total_two = float(
            result.explained_variance["Explained_Variance_Ratio"].head(2).sum()
        )
        needed = pca_module.variance_threshold_components(result, 0.80)
        cumulative_reached = float(
            result.explained_variance["Cumulative_Variance"].iloc[-1]
        )
        columns = st.columns(3)
        columns[0].metric("Rows used", f"{result.n_rows:,}")
        columns[1].metric("Variance in PC1 + PC2", f"{total_two:.1f}%")
        # Only claim a component count when the components actually computed
        # reach the threshold. Otherwise the "2 components = 80% of variance"
        # line is false whenever fewer components were fitted than needed.
        if needed is None:
            columns[2].metric(
                "Components for 80% variance",
                f"Not reached ({cumulative_reached:.1f}% max)",
                delta="compute more components",
                delta_color="off",
            )
        else:
            columns[2].metric("Components for 80% variance", str(needed))

        section(
            "Component loadings",
            "How much each standardised variable contributes to each component, "
            "sorted by weight.",
        )
        loadings = pca_module.loadings_table(result, n_components=min(3, n_components))
        st.dataframe(loadings, width="stretch", hide_index=True)
        download_table(loadings, "Download loadings", key="dl_loadings",
                       filename_hint="pca_loadings")

        info_box(
            "info",
            "How to interpret PCA",
            result.discarded_note
            + " A component with opposite-sign loadings on two variables captures a "
            "contrast between them; variables with high loadings on the same component "
            "tend to vary together.",
        )

# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
with tab_kmeans:
    section(
        "K-Means clustering",
        "Partitions observations into K groups that minimise within-cluster "
        "variation. K is chosen by silhouette score unless you set it explicitly.",
    )

    controls = st.columns([1, 1, 1, 1])
    k_value = controls[0].number_input(
        "K (0 = choose automatically)", min_value=0, max_value=20, value=0, step=1,
        help="Leave at 0 to score every K in the configured range and pick the best.",
    )
    km_sample = controls[1].number_input(
        "Sample size", 1_000, 200_000, SETTINGS.sample_clustering_points, step=1_000,
    )
    km_seed = controls[2].number_input("Random seed", 0, 10_000, SETTINGS.random_seed, step=1)
    show_search = controls[3].toggle("Also score K range", value=True)

    run_kmeans = st.button("Run K-Means", type="primary", key="run_kmeans")

    if run_kmeans:
        with st.spinner(f"Clustering up to {km_sample:,} rows..."):
            try:
                st.session_state["kmeans_result"] = clustering.run_kmeans(
                    selected, spec,
                    k=k_value if k_value else None,
                    sample_limit=km_sample,
                    seed=km_seed,
                )
                if show_search:
                    st.session_state["kmeans_search"] = clustering.silhouette_search(
                        selected, spec, sample_limit=km_sample, seed=km_seed
                    )
            except ValueError as exc:
                st.error(str(exc))
                st.session_state.pop("kmeans_result", None)
        st.rerun()

    search = st.session_state.get("kmeans_search")
    if search is not None and not search.empty:
        from smart_traffic.visualizations import silhouette_chart

        st.plotly_chart(
            silhouette_chart(search, title="Silhouette score by number of clusters"),
            width="stretch",
            config={"displaylogo": False},
        )
        st.dataframe(search, width="stretch", hide_index=True)
        download_table(search, "Download silhouette scores", key="dl_silhouette",
                       filename_hint="kmeans_silhouette_scores")

    km = st.session_state.get("kmeans_result")
    if km is None:
        st.info("Press **Run K-Means** to cluster the selected variables.")
    else:
        sample_disclosure(km.sample_note)
        columns = st.columns(4)
        columns[0].metric("Clusters", km.metadata["k"])
        columns[1].metric("Silhouette score", f"{km.silhouette:.4f}" if km.silhouette else "n/a")
        columns[2].metric("Rows clustered", f"{km.metadata['n_rows']:,}")
        columns[3].metric("Inertia", f"{km.metadata['inertia']:,.0f}")

        section(
            "Cluster profiles",
            "Mean value of each variable within a cluster. This is what actually "
            "distinguishes the clusters from one another.",
        )
        st.dataframe(km.profile, width="stretch", hide_index=True)
        download_table(km.profile, "Download cluster profiles", key="dl_kmeans_profile",
                       filename_hint="kmeans_cluster_profile")

        if km.coordinates is not None:
            section(
                "Cluster projection",
                "The same clusters drawn on the first two principal components. "
                "Separation here is a projection; overlap does not mean the clusters "
                "are identical in the full feature space.",
            )
            st.plotly_chart(
                scatter(
                    km.coordinates, "PC1", "PC2",
                    color="Cluster",
                    title="K-Means clusters on the first two principal components",
                    x_title="PC1 score", y_title="PC2 score", opacity=0.4,
                ),
                width="stretch",
                config={"displaylogo": False},
            )

        sizes = km.profile[["Cluster", "Size"]]
        st.plotly_chart(
            scatter(sizes, "Cluster", "Size", title="Cluster sizes",
                    x_title="Cluster", y_title="Records", opacity=0.9),
            width="stretch", config={"displaylogo": False},
        )
        info_box("warning", "Interpreting clusters", km.interpretation_note)

# ---------------------------------------------------------------------------
# Hierarchical
# ---------------------------------------------------------------------------
with tab_hierarchical:
    section(
        "Agglomerative hierarchical clustering",
        "Builds a dendrogram by merging the closest clusters until the requested "
        "number remains. It requires an O(n²) distance matrix, so the sample is "
        f"capped at {clustering.HIERARCHICAL_HARD_CAP:,} rows regardless of settings.",
    )

    controls = st.columns([1, 1, 1])
    h_k = controls[0].number_input(
        "Number of clusters", 2, 30, SETTINGS.hierarchical_k, step=1
    )
    h_sample = controls[1].number_input(
        "Sample size", 200, clustering.HIERARCHICAL_HARD_CAP,
        SETTINGS.sample_hierarchical_points, step=100,
        help=f"Hard cap: {clustering.HIERARCHICAL_HARD_CAP:,} rows.",
    )
    linkage = controls[2].selectbox(
        "Linkage method", ["ward", "complete", "average", "single"]
    )

    if st.button("Run hierarchical clustering", type="primary", key="run_hier"):
        metric = "euclidean" if linkage == "ward" else "euclidean"
        with st.spinner(f"Clustering up to {h_sample:,} rows..."):
            try:
                st.session_state["hier_result"] = clustering.run_hierarchical(
                    selected, spec, n_clusters=h_k, sample_limit=h_sample,
                    linkage=linkage, metric=metric,
                )
            except ValueError as exc:
                st.error(str(exc))
                st.session_state.pop("hier_result", None)
        st.rerun()

    hier = st.session_state.get("hier_result")
    if hier is None:
        st.info("Press **Run hierarchical clustering** to build the hierarchy.")
    else:
        sample_disclosure(hier.sample_note)
        columns = st.columns(4)
        columns[0].metric("Clusters", hier.metadata["k"])
        columns[1].metric("Silhouette score",
            f"{hier.silhouette:.4f}" if hier.silhouette else "n/a")
        columns[2].metric("Rows used", f"{hier.metadata['n_rows']:,}")
        columns[3].metric("Linkage", hier.metadata["linkage"])

        section("Cluster profiles")
        st.dataframe(hier.profile, width="stretch", hide_index=True)
        download_table(hier.profile, "Download hierarchical profiles",
                       key="dl_hier_profile",
                       filename_hint="hierarchical_cluster_profile")

        if hier.coordinates is not None:
            st.plotly_chart(
                scatter(
                    hier.coordinates, "PC1", "PC2", color="Cluster",
                    title="Hierarchical clusters on the first two components",
                    x_title="PC1 score", y_title="PC2 score", opacity=0.4,
                ),
                width="stretch",
                config={"displaylogo": False},
            )

        with st.expander("Dendrogram", expanded=False):
            st.caption(
                "Ward's linkage with Euclidean distance was used. Ward minimises "
                "the increase in within-cluster variance at each merge, which "
                "generally produces compact, evenly sized clusters."
            )
            try:
                import plotly.figure_factory as ff

                labels = hier.labels.to_numpy()
                linkage_matrix = _linkage_matrix(labels, hier.metadata["k"])
                figure = ff.create_dendrogram(
                    linkage_matrix,
                    labels=[f"C{label}" for label in labels[: len(linkage_matrix)]],
                )
                figure.update_layout(
                    height=520,
                    title="Hierarchical clustering dendrogram",
                    xaxis_title="Sample observation",
                    yaxis_title="Linkage distance",
                    font={"size": 12},
                )
                st.plotly_chart(figure, width="stretch",
                                config={"displaylogo": False})
            except (ImportError, ValueError, MemoryError) as exc:
                st.warning(f"The dendrogram could not be drawn: {exc}")

        info_box("warning", "Interpreting clusters", hier.interpretation_note)


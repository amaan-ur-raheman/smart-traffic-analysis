"""Clustering analysis: K-Means and agglomerative hierarchical.

Both algorithms operate on a bounded, documented sample. Hierarchical clustering
builds an ``O(n^2)`` distance matrix, so its sample cap is deliberately small --
running it on hundreds of thousands of rows is not slow, it is impossible.

Neither result is a ground-truth label. K-Means ``K`` is chosen by silhouette
score, which is a heuristic, not a test of whether ``K`` groups *exist*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from smart_traffic.aggregation import FilterSpec
from smart_traffic.config import SETTINGS
from smart_traffic.logging_config import get_logger
from smart_traffic.sampling import draw_sample

logger = get_logger("clustering")

#: Hard ceiling on rows for hierarchical clustering, independent of settings.
HIERARCHICAL_HARD_CAP: int = 5_000


@dataclass(slots=True)
class ClusteringResult:
    """A fitted clustering run and everything needed to present it.

    Attributes:
        labels: Cluster assignment per sampled row.
        profile: Mean of each feature per cluster, plus cluster size.
        sample_note: Provenance statement for the fit.
        features: Feature names in matrix order.
        coordinates: Optional 2D projection for plotting.
        silhouette: Silhouette score, or ``None`` when not computable.
        metadata: Algorithm-specific extra fields.
        interpretation_note: Caveats shown alongside the result.
    """

    labels: pd.Series
    profile: pd.DataFrame
    sample_note: str
    features: list[str]
    coordinates: pd.DataFrame | None = None
    silhouette: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    interpretation_note: str = (
        "Clusters are a mathematical partition of the selected variables, not a "
        "discovery of real-world accident types. Cluster membership depends on "
        "the feature set, the scaling and the sample: changing any of these "
        "produces a different, equally valid partition. Treat cluster labels as "
        "a descriptive lens, not a ground truth."
    )


def _load_matrix(
    features: Sequence[str],
    spec: FilterSpec | None,
    limit: int,
    seed: int | None,
) -> tuple[np.ndarray, pd.DataFrame, list[str], str]:
    """Fetch, impute and standardise a clustering matrix.

    Args:
        features: Columns to cluster on.
        spec: Active filters.
        limit: Row cap.
        seed: Random seed.

    Returns:
        Tuple of ``(scaled_matrix, raw_frame, feature_names, sample_note)``.

    Raises:
        ValueError: If fewer than two usable features are available.
    """
    available = [f for f in features if _column_exists(f)]
    if len(available) < 2:
        raise ValueError(
            f"Clustering needs at least two usable numeric features; got {len(available)}."
        )

    draw = draw_sample(available, spec=spec, limit=limit, seed=seed)
    frame = draw.frame
    if len(frame) < 3:
        raise ValueError("Not enough rows in the sample to cluster.")

    imputed = SimpleImputer(strategy="median").fit_transform(
        frame[available].to_numpy(dtype=float)
    )
    scaled = StandardScaler().fit_transform(imputed)
    return scaled, frame, available, draw.note()


def _build_profile(
    raw: pd.DataFrame, features: list[str], labels: np.ndarray, n_clusters: int
) -> pd.DataFrame:
    """Compute per-cluster means of the raw (unscaled) features.

    Args:
        raw: Sampled rows, unstandardised.
        features: Feature names.
        labels: Cluster assignment.
        n_clusters: Number of clusters.

    Returns:
        DataFrame indexed by ``Cluster`` with means and a ``Size`` column.
    """
    working = raw[features].apply(pd.to_numeric, errors="coerce").copy()
    working["_cluster"] = labels

    grouped = working.groupby("_cluster").mean().round(4)
    sizes = working["_cluster"].value_counts().sort_index()
    grouped.insert(0, "Size", sizes.to_numpy())
    grouped.index.name = "Cluster"
    return grouped.reset_index()


def run_kmeans(
    features: Sequence[str],
    spec: FilterSpec | None = None,
    *,
    k: int | None = None,
    sample_limit: int | None = None,
    seed: int | None = None,
    project_pca: bool = True,
) -> ClusteringResult:
    """Run K-Means, choosing ``K`` by silhouette score when not specified.

    Args:
        features: Numeric columns to cluster on.
        spec: Active filters.
        k: Number of clusters. When ``None``, every value in the configured
            search range is scored and the best silhouette wins.
        sample_limit: Rows used. Defaults to
            ``SETTINGS.sample_clustering_points``.
        seed: Random seed.
        project_pca: Whether to compute a 2D PCA projection for plotting.

    Returns:
        A :class:`ClusteringResult`.

    Raises:
        ValueError: If too few features or rows are available.
    """
    limit = sample_limit or SETTINGS.sample_clustering_points
    scaled, raw, available, note = _load_matrix(features, spec, limit, seed)
    effective_seed = SETTINGS.random_seed if seed is None else seed

    silhouette_table = pd.DataFrame(columns=["K", "Silhouette_Score", "Inertia"])
    best_k: int
    scores: dict[int, float] = {}

    if k is None:
        for candidate in range(SETTINGS.kmeans_k_min, SETTINGS.kmeans_k_max + 1):
            if candidate >= len(scaled):
                break
            model = KMeans(
                n_clusters=candidate, random_state=effective_seed, n_init=10
            ).fit(scaled)
            score = float(silhouette_score(scaled, model.labels_))
            scores[candidate] = score
            silhouette_table.loc[len(silhouette_table)] = {
                "K": candidate,
                "Silhouette_Score": round(score, 4),
                "Inertia": round(float(model.inertia_), 2),
            }
        if not scores:
            raise ValueError("Could not score any K; the sample is too small.")
        best_k = max(scores, key=lambda c: scores[c])
    else:
        best_k = int(k)
        if best_k >= len(scaled):
            raise ValueError("K must be smaller than the number of sampled rows.")

    model = KMeans(n_clusters=best_k, random_state=effective_seed, n_init=10).fit(scaled)
    silhouette = (
        float(silhouette_score(scaled, model.labels_)) if 1 < best_k < len(scaled) else None
    )

    coordinates = None
    if project_pca:
        coordinates = _pca_projection(scaled, model.labels_, raw)

    logger.info("KMeans k=%s silhouette=%.4f on %s rows", best_k, silhouette or float("nan"),
        f"{len(raw):,}")

    return ClusteringResult(
        labels=pd.Series(model.labels_, name="Cluster"),
        profile=_build_profile(raw, available, model.labels_, best_k),
        sample_note=note,
        features=available,
        coordinates=coordinates,
        silhouette=silhouette,
        metadata={
            "k": best_k,
            "inertia": round(float(model.inertia_), 2),
            "silhouette_search": silhouette_table,
            "n_rows": len(raw),
        },
    )


def silhouette_search(
    features: Sequence[str],
    spec: FilterSpec | None = None,
    *,
    k_min: int | None = None,
    k_max: int | None = None,
    sample_limit: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Score a range of ``K`` values without committing to one.

    Separating the search from :func:`run_kmeans` lets the UI show the elbow
    before the user commits, and keeps the expensive sweep out of the default
    page render.

    Args:
        features: Numeric columns.
        spec: Active filters.
        k_min: Lowest ``K`` to score. Defaults to the configured minimum.
        k_max: Highest ``K``. Defaults to the configured maximum.
        sample_limit: Rows used.
        seed: Random seed.

    Returns:
        DataFrame with ``K``, ``Silhouette_Score``, ``Inertia`` and
        ``Cluster_Size_Min`` / ``Cluster_Size_Max``.
    """
    low = k_min or SETTINGS.kmeans_k_min
    high = k_max or SETTINGS.kmeans_k_max
    limit = sample_limit or SETTINGS.sample_clustering_points

    scaled, _raw, _available, _note = _load_matrix(features, spec, limit, seed)
    effective_seed = SETTINGS.random_seed if seed is None else seed

    rows = []
    for candidate in range(low, high + 1):
        if candidate >= len(scaled):
            break
        model = KMeans(
            n_clusters=candidate, random_state=effective_seed, n_init=10
        ).fit(scaled)
        counts = np.bincount(model.labels_)
        rows.append(
            {
                "K": candidate,
                "Silhouette_Score": round(float(silhouette_score(scaled, model.labels_)), 4),
                "Inertia": round(float(model.inertia_), 2),
                "Cluster_Size_Min": int(counts.min()),
                "Cluster_Size_Max": int(counts.max()),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["Is_Recommended"] = table["K"] == table.loc[table["Silhouette_Score"].idxmax(), "K"]
    return table


def run_hierarchical(
    features: Sequence[str],
    spec: FilterSpec | None = None,
    *,
    n_clusters: int | None = None,
    sample_limit: int | None = None,
    seed: int | None = None,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> ClusteringResult:
    """Run agglomerative hierarchical clustering on a small sample.

    Args:
        features: Numeric columns.
        spec: Active filters.
        n_clusters: Number of clusters to cut the dendrogram into.
        sample_limit: Rows used, capped at :data:`HIERARCHICAL_HARD_CAP`.
        seed: Random seed for sample selection.
        linkage: Linkage criterion. ``ward`` requires Euclidean distance.
        metric: Distance metric.

    Returns:
        A :class:`ClusteringResult`.

    Raises:
        ValueError: If too few features or rows are available.
        ValueError: If ``ward`` linkage is paired with a non-Euclidean metric.
    """
    if linkage == "ward" and metric != "euclidean":
        raise ValueError("Ward's linkage requires euclidean distance.")

    requested = sample_limit or SETTINGS.sample_hierarchical_points
    limit = min(requested, HIERARCHICAL_HARD_CAP)

    scaled, raw, available, note = _load_matrix(features, spec, limit, seed)
    effective_seed = SETTINGS.random_seed if seed is None else seed

    k = n_clusters or SETTINGS.hierarchical_k
    k = max(2, min(int(k), len(scaled) - 1))

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
    labels = model.fit_predict(scaled)

    silhouette = (
        float(silhouette_score(scaled, labels)) if 1 < k < len(scaled) else None
    )

    coordinates = _pca_projection(scaled, labels, raw)

    logger.info(
        "Hierarchical clustering: %s clusters, linkage=%s, %s rows",
        k, linkage, f"{len(raw):,}",
    )

    return ClusteringResult(
        labels=pd.Series(labels, name="Cluster"),
        profile=_build_profile(raw, available, labels, k),
        sample_note=note,
        features=available,
        coordinates=coordinates,
        silhouette=silhouette,
        metadata={
            "k": k,
            "linkage": linkage,
            "metric": metric,
            "n_rows": len(raw),
            "seed": effective_seed,
            "hard_row_cap": HIERARCHICAL_HARD_CAP,
        },
    )


def _pca_projection(
    scaled: np.ndarray, labels: np.ndarray, raw: pd.DataFrame
) -> pd.DataFrame:
    """Project the scaled matrix onto its first two principal components.

    Args:
        scaled: Standardised clustering matrix.
        labels: Cluster assignment.
        raw: Sampled rows, for carrying along Severity.

    Returns:
        DataFrame with ``PC1``, ``PC2`` and ``Cluster``.
    """
    from sklearn.decomposition import PCA

    n_components = min(2, scaled.shape[1], scaled.shape[0])
    scores = PCA(n_components=n_components, random_state=SETTINGS.random_seed).fit_transform(scaled)

    frame = pd.DataFrame(
        {
            "PC1": np.round(scores[:, 0], 4),
            "PC2": np.round(scores[:, min(1, n_components - 1)], 4),
            "Cluster": labels,
        }
    )
    if "Severity" in raw.columns:
        frame["Severity"] = raw["Severity"].to_numpy()
    return frame


def _column_exists(name: str) -> bool:
    """Whether ``name`` is present in the processed table."""
    from smart_traffic.db import DB

    try:
        return DB.has_column(name)
    except Exception:  # noqa: BLE001 - dataset may be absent
        return False

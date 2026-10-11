"""Principal component analysis.

PCA is run on a documented, reproducible sample. Standardisation is mandatory
here: without it, ``Pressure(in)`` (values near 30) would dominate
``Humidity(%)`` purely because of its scale, and the components would describe
units rather than variance.

What PCA can and cannot tell you is stated in the returned metadata so the UI
can present it honestly rather than as a black box.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from smart_traffic.aggregation import FilterSpec
from smart_traffic.config import SETTINGS
from smart_traffic.logging_config import get_logger
from smart_traffic.sampling import draw_sample

logger = get_logger("pca")

#: Cap on rows used for the 2D scatter. PC1/PC2 coordinates are cheap to plot;
#: a million markers is not.
SCATTER_CAP: int = 30_000


@dataclass(slots=True)
class PCAResult:
    """Everything the UI needs to present a PCA run.

    Attributes:
        coordinates: PC1/PC2 scores per sampled observation.
        explained_variance: Per-component explained variance ratio.
        cumulative_variance: Running total of explained variance.
        loadings: Component loadings per original feature.
        components: The fitted transformer.
        scaler: The fitted standardiser.
        imputer: The fitted imputer.
        features: Feature names in matrix order.
        n_rows: Rows used in the fit.
        sample_note: Provenance statement for the fit.
        discarded_note: What PCA cannot recover.
    """

    coordinates: pd.DataFrame
    explained_variance: pd.DataFrame
    cumulative_variance: pd.DataFrame
    loadings: pd.DataFrame
    components: PCA
    scaler: StandardScaler
    imputer: SimpleImputer
    features: list[str]
    n_rows: int
    sample_note: str
    discarded_note: str = field(
        default=(
            "PCA compresses to as many components as there are features, so no "
            "information is discarded from the full decomposition -- but the two "
            "components plotted below capture only the leading directions of "
            "variance. Any structure living in the remaining components is not "
            "visible in the scatter plot. Components are linear combinations and "
            "should not be assumed to have a direct real-world meaning."
        )
    )


def run_pca(
    features: Sequence[str],
    spec: FilterSpec | None = None,
    *,
    n_components: int = 2,
    sample_limit: int | None = None,
    seed: int | None = None,
) -> PCAResult | None:
    """Fit PCA on a standardised, imputed sample of the dataset.

    Args:
        features: Numeric columns to analyse. At least two are required.
        spec: Active filters.
        n_components: Number of components to compute. ``min(n_components,
            len(features))`` are produced.
        sample_limit: Rows used for the fit. Defaults to
            ``SETTINGS.sample_scatter_points``.
        seed: Random seed for sample selection.

    Returns:
        A :class:`PCAResult`, or ``None`` when no dataset is available.

    Raises:
        ValueError: If fewer than two usable features are supplied.
    """
    available = [f for f in features if _column_exists(f)]
    if len(available) < 2:
        raise ValueError(
            f"PCA needs at least two usable numeric features; got {len(available)}."
        )

    limit = sample_limit or SETTINGS.sample_scatter_points
    draw = draw_sample(available, spec=spec, limit=limit, seed=seed)
    frame = draw.frame
    if frame.empty:
        return None

    # Median imputation keeps every sampled row, and the imputer is fitted only
    # on this matrix -- there is no separate evaluation set here, so the
    # distinction is about honesty rather than leakage.
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(frame[available].to_numpy(dtype=float))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    effective_components = min(n_components, len(available))
    pca = PCA(n_components=effective_components, random_state=SETTINGS.random_seed)
    scores = pca.fit_transform(scaled)

    explained = pd.DataFrame(
        {
            "Component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "Explained_Variance_Ratio": np.round(pca.explained_variance_ratio_ * 100, 4),
            "Cumulative_Variance": np.round(
                np.cumsum(pca.explained_variance_ratio_) * 100, 4
            ),
        }
    )

    loadings = pd.DataFrame(pca.components_.T, index=available)
    loadings.columns = explained["Component"].tolist()
    loadings = loadings.round(4).reset_index().rename(columns={"index": "Feature"})

    coordinates = frame[available].copy()
    for index in range(effective_components):
        coordinates[f"PC{index + 1}"] = np.round(scores[:, index], 4)
    if "Severity" in frame.columns:
        coordinates["Severity"] = frame["Severity"].to_numpy()

    logger.info(
        "PCA fit on %s rows x %s features: PC1+PC2 explain %.1f%% of variance",
        f"{len(frame):,}", len(available),
        float(pca.explained_variance_ratio_[:2].sum() * 100),
    )

    return PCAResult(
        coordinates=coordinates,
        explained_variance=explained,
        cumulative_variance=explained,
        loadings=loadings,
        components=pca,
        scaler=scaler,
        imputer=imputer,
        features=available,
        n_rows=len(frame),
        sample_note=draw.note(),
    )


def loadings_table(result: PCAResult, n_components: int = 2) -> pd.DataFrame:
    """Return the loadings for the leading components, sorted by magnitude.

    Sorting by absolute weight makes the dominant contributors to each component
    readable, which a raw loading table in column order does not.

    Args:
        result: A fitted :class:`PCAResult`.
        n_components: How many leading components to include.

    Returns:
        Long DataFrame with ``Feature``, ``Component``, ``Loading``,
        ``Abs_Loading`` and ``Direction``.
    """
    components = [c for c in result.explained_variance["Component"] if c in result.loadings.columns]
    components = components[:n_components]
    subset = result.loadings[["Feature", *components]].copy()

    rows = []
    for component in components:
        block = subset[["Feature", component]].copy()
        block.columns = ["Feature", "Loading"]
        block["Component"] = component
        rows.append(block)

    long = pd.concat(rows, ignore_index=True)
    long["Abs_Loading"] = long["Loading"].abs().round(4)
    long["Direction"] = np.where(long["Loading"] >= 0, "Positive", "Negative")
    return (
        long.sort_values(["Component", "Abs_Loading"], ascending=[True, False])
        .reset_index(drop=True)
        .drop(columns=["Loading"])
    )


def variance_threshold_components(result: PCAResult, threshold: float = 0.80) -> int | None:
    """Return how many components are needed to reach a variance threshold.

    Args:
        result: A fitted :class:`PCAResult`.
        threshold: Cumulative variance fraction to reach, in 0-1.

    Returns:
        Number of components required, or ``None`` when the components that were
        actually computed do not reach the threshold. Returning ``None`` rather
        than clamping to the component count matters: clamping makes an
        under-computed analysis report a confident-looking number that is simply
        false (two components holding 40% of the variance are not "2 components
        for 80%"). Callers must render the ``None`` case explicitly.
    """
    cumulative = result.explained_variance["Cumulative_Variance"].to_numpy(dtype=float)
    target = threshold * 100
    if cumulative.size == 0 or cumulative[-1] < target:
        return None
    needed = int(np.searchsorted(cumulative, target, side="left") + 1)
    return min(max(needed, 1), len(cumulative))


def _column_exists(name: str) -> bool:
    """Whether ``name`` is present in the processed table."""
    from smart_traffic.db import DB

    try:
        return DB.has_column(name)
    except Exception:  # noqa: BLE001 - dataset may be absent
        return False

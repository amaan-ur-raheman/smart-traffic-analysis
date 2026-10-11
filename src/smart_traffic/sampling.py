"""Reproducible sampling for analysis that cannot run over the full table.

PCA scatter plots, clustering and model training all need the data in memory.
This module is the single place that decides *how much* data comes into memory
and *which* rows, so the choice is consistent, documented and reproducible.

Selection is deterministic: rows are ordered by ``hash(ID, seed)`` rather than
``RANDOM()``. Given the same dataset and seed, the same rows are drawn every
time -- which is what makes the reported metrics meaningful across reruns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smart_traffic.aggregation import FilterSpec
from smart_traffic.config import SETTINGS
from smart_traffic.db import ACCIDENTS_VIEW, Database
from smart_traffic.logging_config import get_logger

logger = get_logger("sampling")


@dataclass(slots=True)
class SampleDraw:
    """A materialised sample plus its provenance.

    Attributes:
        frame: The sampled rows.
        seed: Seed used for selection.
        population: Number of rows matching the filter before sampling.
        requested: Row cap that was requested.
        source_table: Which DuckDB view the rows came from.
    """

    frame: pd.DataFrame
    seed: int
    population: int
    requested: int
    source_table: str

    @property
    def rows(self) -> int:
        """Number of rows actually returned."""
        return len(self.frame)

    @property
    def coverage_percent(self) -> float:
        """Share of the filtered population captured."""
        return round(self.rows / self.population * 100, 3) if self.population else 0.0

    def provenance(self) -> dict[str, float | int | str]:
        """Return sample metadata for display and for report footers."""
        return {
            "rows": self.rows,
            "population": self.population,
            "coverage_%": self.coverage_percent,
            "seed": self.seed,
            "source_table": self.source_table,
        }

    def note(self) -> str:
        """Return a one-line statement suitable for a chart caption."""
        return (
            f"Computed on a reproducible sample of {self.rows:,} rows "
            f"({self.coverage_percent}% of {self.population:,} matching records, "
            f"seed={self.seed}). Not a full-dataset figure."
        )


def draw_sample(
    columns: list[str],
    *,
    spec: FilterSpec | None = None,
    limit: int | None = None,
    seed: int | None = None,
    db: Database | None = None,
    source_table: str = ACCIDENTS_VIEW,
) -> SampleDraw:
    """Draw a deterministic sample of rows from the processed table.

    Args:
        columns: Columns to select.
        spec: Active filters. The population is defined by these.
        limit: Maximum rows to return. Defaults to 50,000, or
            ``SETTINGS.sample_scatter_points`` when ``columns`` looks like a
            scatter-plot selection. Callers should pass an explicit limit.
        seed: Random seed. Defaults to ``SETTINGS.random_seed``.
        db: Optional database handle.
        source_table: Which view to sample from.

    Returns:
        A :class:`SampleDraw` with the rows and their provenance.

    Raises:
        ValueError: If ``columns`` is empty.
    """
    if not columns:
        raise ValueError("draw_sample requires at least one column.")

    from smart_traffic.db import DB

    handle = db or DB
    spec = spec or FilterSpec()
    effective_seed = SETTINGS.random_seed if seed is None else seed
    row_cap = limit if limit is not None else SETTINGS.sample_scatter_points

    population = int(
        spec.select(
            f"SELECT COUNT(*) AS n FROM {source_table} WHERE {{where}}", handle
        )["n"][0]
    )

    projection = ", ".join(f'"{c}"' for c in columns)
    sql = f"""
        SELECT {projection}
        FROM {source_table}
        WHERE {{where}}
        ORDER BY hash(ID, ?)
        LIMIT ?
    """
    frame = spec.select(sql, handle, params=[effective_seed, int(row_cap)])

    logger.debug(
        "Sampled %s of %s rows (seed %s) from %s",
        f"{len(frame):,}", f"{population:,}", effective_seed, source_table,
    )
    return SampleDraw(
        frame=frame.reset_index(drop=True),
        seed=effective_seed,
        population=population,
        requested=int(row_cap),
        source_table=source_table,
    )


def prepare_numeric_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    drop_all_missing: bool = True,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Extract a numeric matrix suitable for scikit-learn.

    Rows containing a missing value are dropped rather than imputed here.
    Imputation belongs inside the fitted pipeline (see
    :mod:`smart_traffic.modeling`) so that no statistic leaks from the
    evaluation set into training.

    Args:
        frame: Source frame.
        columns: Columns to extract, in order.
        drop_all_missing: When ``True``, remove rows where every selected
            column is missing. Partial gaps are left for the caller to impute.

    Returns:
        Tuple of ``(matrix, used_columns, rows_without_any_value)``.

    Raises:
        ValueError: If ``columns`` is empty.
    """
    if not columns:
        raise ValueError("prepare_numeric_matrix requires at least one column.")

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Columns not present in the frame: {', '.join(missing)}")

    subset = frame[columns].apply(pd.to_numeric, errors="coerce")

    if drop_all_missing:
        keep = ~subset.isna().all(axis=1)
        subset = subset[keep]

    # Rows still holding a gap would break KMeans and PCA; they are removed and
    # reported rather than silently imputed.
    usable = subset.dropna()
    return (
        usable.to_numpy(dtype=float),
        list(columns),
        subset.loc[subset.index.difference(usable.index)],
    )

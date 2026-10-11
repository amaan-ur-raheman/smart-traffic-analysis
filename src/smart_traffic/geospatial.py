"""Geographic analysis.

Plotting 7.7.7M individual coordinates is not possible in a browser and is not
necessary. Everything here aggregates first: the density grid bins coordinates
into cells in SQL and returns one row per cell, which for a realistic cell size
is a few thousand rows rather than a few million.

The distinction this module is careful about: a cell with many recorded
accidents is a *hotspot of recorded accidents*. It is not the most dangerous
place to drive, because that ranking needs traffic exposure data the dataset
does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from smart_traffic.aggregation import Aggregator, FilterSpec
from smart_traffic.config import EXPOSURE_DISCLAIMER, SETTINGS
from smart_traffic.db import ACCIDENTS_VIEW, DB
from smart_traffic.logging_config import get_logger

logger = get_logger("geospatial")

#: Default aggregation cell size in degrees. Roughly 50-100 km at US latitudes,
#: which keeps metropolitan areas distinct while collapsing empty ocean and
#: rural space.
DEFAULT_CELL_DEGREES: Final[float] = 0.25

#: Hard cap on grid cells returned to the browser.
MAX_GRID_CELLS: Final[int] = 20_000


@dataclass(slots=True)
class DensityGrid:
    """A binned accident-density surface.

    Attributes:
        cells: One row per cell, with the cell centre and its count.
        cell_degrees: Cell size used.
        total: Records covered by the grid.
        note: Provenance statement for display.
    """

    cells: pd.DataFrame
    cell_degrees: float
    total: int
    note: str = EXPOSURE_DISCLAIMER

    @property
    def peak(self) -> dict[str, float] | None:
        """The highest-count cell, or ``None`` when the grid is empty."""
        if self.cells.empty:
            return None
        row = self.cells.loc[self.cells["Accidents"].idxmax()]
        return {
            "Latitude": round(float(row["Latitude"]), 3),
            "Longitude": round(float(row["Longitude"]), 3),
            "Accidents": int(row["Accidents"]),
        }

    def top_cells(self, n: int = SETTINGS.top_n) -> pd.DataFrame:
        """Return the ``n`` busiest cells.

        Args:
            n: Number of cells.

        Returns:
            DataFrame sorted by count descending.
        """
        return self.cells.head(n).reset_index(drop=True) if not self.cells.empty else self.cells


def density_grid(
    spec: FilterSpec | None = None,
    *,
    cell_degrees: float = DEFAULT_CELL_DEGREES,
    max_cells: int = MAX_GRID_CELLS,
) -> DensityGrid:
    """Bin accident coordinates into a grid and count each cell.

    Args:
        spec: Active filters.
        cell_degrees: Cell size in degrees.
        max_cells: Cap on returned cells; the busiest are kept.

    Returns:
        A :class:`DensityGrid`.
    """
    spec = spec or FilterSpec()
    cell = max(float(cell_degrees), 0.01)

    clause, params = spec.where_clause()
    sql = f"""
        SELECT
            FLOOR(Start_Lat / ?) * ? + ? / 2.0 AS Latitude,
            FLOOR(Start_Lng / ?) * ? + ? / 2.0 AS Longitude,
            COUNT(*)::BIGINT AS Accidents
        FROM {ACCIDENTS_VIEW}
        WHERE {clause or "TRUE"}
          AND Is_Valid_Record = TRUE
          AND Start_Lat IS NOT NULL AND Start_Lng IS NOT NULL
        GROUP BY 1, 2
        ORDER BY Accidents DESC
        LIMIT {int(max_cells)}
    """
    cells = DB.sql(sql, [cell, cell, cell, cell, cell, cell] + params)

    total = int(cells["Accidents"].sum()) if not cells.empty else 0
    logger.info(
        "Density grid: %s cells at %.2f degrees covering %s records",
        f"{len(cells):,}", cell, f"{total:,}",
    )
    return DensityGrid(cells=cells, cell_degrees=cell, total=total)


def state_hotspots(
    spec: FilterSpec | None = None,
    *,
    limit: int = SETTINGS.top_n,
) -> pd.DataFrame:
    """Return the busiest states with coordinates for map plotting.

    Args:
        spec: Active filters.
        limit: Number of states.

    Returns:
        DataFrame with ``State``, ``Accidents`` and approximate centre
        ``Latitude`` / ``Longitude`` (computed as the mean of valid records,
        which is more accurate than a hardcoded capital lookup for states with
        uneven reporting).
    """
    spec = spec or FilterSpec()
    sql = f"""
        SELECT State,
               COUNT(*)::BIGINT AS Accidents,
               ROUND(AVG(Start_Lat), 4) AS Latitude,
               ROUND(AVG(Start_Lng), 4) AS Longitude
        FROM {ACCIDENTS_VIEW}
        WHERE {{where}} AND State IS NOT NULL AND Is_Valid_Record = TRUE
        GROUP BY State
        ORDER BY Accidents DESC
        LIMIT {int(limit)}
    """
    frame = spec.select(sql, DB)
    return frame.rename(columns={"State": "State"})


def top_cities(spec: FilterSpec | None = None, *, limit: int = SETTINGS.top_n) -> pd.DataFrame:
    """Return the busiest cities.

    Args:
        spec: Active filters.
        limit: Number of cities.

    Returns:
        DataFrame with ``City``, ``State``, ``Accidents`` and mean coordinates.
    """
    spec = spec or FilterSpec()
    sql = f"""
        SELECT City, State,
               COUNT(*)::BIGINT AS Accidents,
               ROUND(AVG(Start_Lat), 4) AS Latitude,
               ROUND(AVG(Start_Lng), 4) AS Longitude
        FROM {ACCIDENTS_VIEW}
        WHERE {{where}} AND City IS NOT NULL AND Is_Valid_Record = TRUE
        GROUP BY City, State
        ORDER BY Accidents DESC
        LIMIT {int(limit)}
    """
    return spec.select(sql, DB)


def coordinate_distribution(spec: FilterSpec | None = None) -> pd.DataFrame:
    """Return latitude and longitude summary statistics.

    Args:
        spec: Active filters.

    Returns:
        DataFrame with one row for latitude and one for longitude, each with
        count, mean, standard deviation, range and the 1st/99th percentiles.
        Percentiles are more informative than min/max, which are outliers.
    """
    spec = spec or FilterSpec()
    sql = f"""
        SELECT
            COUNT(Start_Lat)::BIGINT AS Latitude_Count,
            ROUND(AVG(Start_Lat), 4) AS Latitude_Mean,
            ROUND(STDDEV_SAMP(Start_Lat), 4) AS Latitude_Std,
            ROUND(MIN(Start_Lat), 4) AS Latitude_Min,
            ROUND(QUANTILE_CONT(Start_Lat, 0.01), 4) AS Latitude_P01,
            ROUND(QUANTILE_CONT(Start_Lat, 0.99), 4) AS Latitude_P99,
            ROUND(MAX(Start_Lat), 4) AS Latitude_Max,
            COUNT(Start_Lng)::BIGINT AS Longitude_Count,
            ROUND(AVG(Start_Lng), 4) AS Longitude_Mean,
            ROUND(STDDEV_SAMP(Start_Lng), 4) AS Longitude_Std,
            ROUND(MIN(Start_Lng), 4) AS Longitude_Min,
            ROUND(QUANTILE_CONT(Start_Lng, 0.01), 4) AS Longitude_P01,
            ROUND(QUANTILE_CONT(Start_Lng, 0.99), 4) AS Longitude_P99,
            ROUND(MAX(Start_Lng), 4) AS Longitude_Max
        FROM {ACCIDENTS_VIEW}
        WHERE {{where}} AND Is_Valid_Record = TRUE
    """
    row = spec.select(sql, DB)
    if row.empty:
        return pd.DataFrame()

    value = row.iloc[0]
    return pd.DataFrame(
        [
            {
                "Coordinate": "Latitude",
                "Count": int(value["Latitude_Count"]),
                "Mean": value["Latitude_Mean"],
                "Std": value["Latitude_Std"],
                "Min": value["Latitude_Min"],
                "P01": value["Latitude_P01"],
                "P99": value["Latitude_P99"],
                "Max": value["Latitude_Max"],
            },
            {
                "Coordinate": "Longitude",
                "Count": int(value["Longitude_Count"]),
                "Mean": value["Longitude_Mean"],
                "Std": value["Longitude_Std"],
                "Min": value["Longitude_Min"],
                "P01": value["Longitude_P01"],
                "P99": value["Longitude_P99"],
                "Max": value["Longitude_Max"],
            }
        ]
    )


def state_severity_by_map(spec: FilterSpec | None = None, *, limit: int = 50) -> pd.DataFrame:
    """Return every state with coordinates and its high-severity share.

    Args:
        spec: Active filters.
        limit: Cap on states returned.

    Returns:
        DataFrame suitable for a choropleth, with ``High_Severity_Rate_%``
        computed within each state -- not as a share of the national total.
    """
    spec = spec or FilterSpec()
    sql = f"""
        SELECT State,
               COUNT(*)::BIGINT AS Accidents,
               ROUND(AVG(Start_Lat), 4) AS Latitude,
               ROUND(AVG(Start_Lng), 4) AS Longitude,
               COALESCE(SUM(CASE WHEN Severity IN (3, 4) THEN 1 ELSE 0 END), 0)::BIGINT
                   AS High_Severity,
               ROUND(100.0 * SUM(CASE WHEN Severity IN (3, 4) THEN 1 ELSE 0 END)
                     / NULLIF(COUNT(*), 0), 4) AS High_Severity_Rate
        FROM {ACCIDENTS_VIEW}
        WHERE {{where}} AND State IS NOT NULL AND Is_Valid_Record = TRUE
        GROUP BY State
        ORDER BY Accidents DESC
        LIMIT {int(limit)}
    """
    return spec.select(sql, DB)


def geographic_coverage(spec: FilterSpec | None = None) -> pd.DataFrame:
    """Return the number of distinct states and cities represented.

    Args:
        spec: Active filters.

    Returns:
        DataFrame with ``Scope`` and ``Distinct_Values`` rows for states,
        counties and cities.
    """
    spec = spec or FilterSpec()
    sql = f"""
        SELECT
            COUNT(DISTINCT State)::BIGINT AS States,
            COUNT(DISTINCT County)::BIGINT AS Counties,
            COUNT(DISTINCT City)::BIGINT AS Cities
        FROM {ACCIDENTS_VIEW}
        WHERE {{where}}
    """
    row = spec.select(sql, DB)
    if row.empty:
        return pd.DataFrame(columns=["Scope", "Distinct_Values"])
    value = row.iloc[0]
    return pd.DataFrame(
        [
            {"Scope": "States", "Distinct_Values": int(value["States"])},
            {"Scope": "Counties", "Distinct_Values": int(value["Counties"])},
            {"Scope": "Cities", "Distinct_Values": int(value["Cities"])},
        ]
    )


def top_feature_counts(
    spec: FilterSpec | None = None, *, limit: int = SETTINGS.top_n
) -> pd.DataFrame:
    """Return the busiest states as a plain count table.

    Args:
        spec: Active filters.
        limit: Number of states.

    Returns:
        DataFrame with ``State`` and ``Accidents``.
    """
    return Aggregator().counts_by("State", spec, limit=limit, db=DB)

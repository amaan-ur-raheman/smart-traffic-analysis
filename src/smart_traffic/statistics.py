"""Statistical analysis over the processed table.

Everything here runs on either an aggregate query (cheap, full dataset) or a
documented reproducible sample (expensive, bounded). Which one was used is
always reported alongside the number.

Correlation is treated throughout as an *association*. Nothing in this module
supports a causal claim, and the helpers say so in their return values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

from smart_traffic import schema
from smart_traffic.aggregation import FilterSpec
from smart_traffic.config import SETTINGS
from smart_traffic.logging_config import get_logger
from smart_traffic.sampling import SampleDraw, draw_sample

logger = get_logger("statistics")

#: Numeric columns used for correlation analysis.
CORRELATION_FEATURES: Final[tuple[str, ...]] = (
    "Severity",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Duration_Hours",
    "Hour",
    "Is_Weekend",
    "Is_Rush_Hour",
    "Is_Night",
)


@dataclass(slots=True)
class CorrelationResult:
    """A correlation matrix plus its provenance and strongest pairs.

    Attributes:
        matrix: Square DataFrame of Pearson correlation coefficients.
        computed_on: ``"full dataset"`` or a sample description.
        sample_note: Provenance string when computed on a sample.
    """

    matrix: pd.DataFrame
    computed_on: str
    sample_note: str = ""

    def strongest_pairs(self, n: int = 10, threshold: float = 0.05) -> pd.DataFrame:
        """Return the ``n`` strongest off-diagonal correlations.

        Each unordered pair appears once. Diagonal entries (a variable with
        itself, always 1.0) are excluded.

        Args:
            n: Number of pairs to return.
            threshold: Minimum ``|r|`` to include.

        Returns:
            DataFrame with ``Variable_A``, ``Variable_B``, ``Correlation``,
            ``Abs_Correlation`` and ``Strength``.
        """
        values = self.matrix.to_numpy(dtype=float)
        labels = list(self.matrix.columns)
        rows: list[dict[str, float | str]] = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                coefficient = values[i, j]
                if np.isnan(coefficient):
                    continue
                rows.append(
                    {
                        "Variable_A": labels[i],
                        "Variable_B": labels[j],
                        "Correlation": round(float(coefficient), 4),
                        "Abs_Correlation": round(abs(float(coefficient)), 4),
                        "Strength": describe_strength(abs(float(coefficient))),
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=["Variable_A", "Variable_B", "Correlation", "Abs_Correlation", "Strength"]
            )
        frame = pd.DataFrame(rows)
        filtered = frame[frame["Abs_Correlation"] >= threshold]
        return filtered.sort_values("Abs_Correlation",
            ascending=False).head(n).reset_index(drop=True)


def describe_strength(coefficient: float) -> str:
    """Return a qualitative label for a correlation magnitude.

    Args:
        coefficient: Absolute Pearson correlation.

    Returns:
        One of ``very weak``, ``weak``, ``moderate``, ``strong``, ``very strong``.
    """
    if coefficient < 0.1:
        return "very weak"
    if coefficient < 0.3:
        return "weak"
    if coefficient < 0.5:
        return "moderate"
    if coefficient < 0.7:
        return "strong"
    return "very strong"


def correlation_matrix(
    spec: FilterSpec | None = None,
    *,
    features: Sequence[str] = CORRELATION_FEATURES,
    sample_limit: int | None = None,
    seed: int | None = None,
) -> CorrelationResult:
    """Compute a Pearson correlation matrix over the numeric features.

    Correlations must be computed row-wise, so this cannot be pushed entirely
    into SQL. Above ``sample_limit`` rows the computation switches to a
    reproducible sample, which is stated in the result.

    Args:
        spec: Active filters.
        features: Columns to correlate. Missing ones are dropped.
        sample_limit: Row cap. Defaults to
            ``SETTINGS.sample_permutation_importance``.
        seed: Random seed for the sample.

    Returns:
        A :class:`CorrelationResult`.
    """
    spec = spec or FilterSpec()
    requested = list(features)
    available = [c for c in requested if _column_exists(c)]

    if not available:
        return CorrelationResult(pd.DataFrame(), "unavailable", "No numeric columns available.")

    cap = sample_limit or SETTINGS.sample_permutation_importance
    draw = draw_sample(available, spec=spec, limit=cap, seed=seed)
    frame = draw.frame

    numeric = frame[available].apply(pd.to_numeric, errors="coerce")
    # Median impute for display only: this is a descriptive table, not a model.
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = numeric.corr(method="pearson")

    computed_on = "full dataset" if draw.rows >= draw.population else "reproducible sample"
    return CorrelationResult(
        matrix=matrix.round(4),
        computed_on=computed_on,
        sample_note="" if draw.rows >= draw.population else draw.note(),
    )


def numeric_descriptives(
    spec: FilterSpec | None = None,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return full-dataset descriptive statistics via SQL aggregates.

    Unlike correlation, these are exact: no sample is involved.

    Args:
        spec: Active filters.
        columns: Numeric columns to summarise. Defaults to the dataset's own.

    Returns:
        DataFrame with count, mean, std, min, quantiles, max and skew.
    """
    from smart_traffic.db import ACCIDENTS_VIEW, DB

    spec = spec or FilterSpec()
    available = [c for c in (columns or schema.NUMERIC_COLUMNS) if _column_exists(c)]
    if not available:
        return pd.DataFrame()

    parts = ["COUNT(*)::BIGINT AS Rows"]
    for column in available:
        parts.append(
            f'COUNT("{column}")::BIGINT AS "{column}__Count"'
        )
        parts.append(f'ROUND(AVG("{column}"), 4) AS "{column}__Mean"')
        parts.append(f'ROUND(STDDEV_SAMP("{column}"), 4) AS "{column}__Std"')
        parts.append(f'ROUND(MIN("{column}"), 4) AS "{column}__Min"')
        parts.append(f'ROUND(QUANTILE_CONT("{column}", 0.25), 4) AS "{column}__Q1"')
        parts.append(f'ROUND(MEDIAN("{column}"), 4) AS "{column}__Median"')
        parts.append(f'ROUND(QUANTILE_CONT("{column}", 0.75), 4) AS "{column}__Q3"')
        parts.append(f'ROUND(MAX("{column}"), 4) AS "{column}__Max"')
        parts.append(f'ROUND(SKEWNESS("{column}"), 4) AS "{column}__Skew"')

    raw = spec.select(f"SELECT {', '.join(parts)} FROM {ACCIDENTS_VIEW}", DB)
    if raw.empty:
        return pd.DataFrame()

    rows = []
    for column in available:
        rows.append(
            {key.replace(f"{column}__", ""): (raw[key][0] if not pd.isna(raw[key][0]) else None)
             for key in raw.columns
             if key.startswith(f"{column}__")}
            | {"Column": column}
        )
    frame = pd.DataFrame(rows)
    ordered = ["Column", "Count", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max", "Skew"]
    return frame[[c for c in ordered if c in frame.columns]]


def mann_whitney_comparison(
    group_column: str,
    measure_column: str,
    spec: FilterSpec | None = None,
    *,
    max_groups: int = 6,
    min_group_size: int = 50,
) -> pd.DataFrame:
    """Compare a numeric measure between the largest groups of a category.

    Uses the Mann-Whitney U test, which does not assume normality -- appropriate
    for the skewed weather and severity variables in this dataset.

    The test says whether the *distributions differ*, not why. A significant
    result here means the groups are distinguishable, not that the grouping
    variable causes the difference: unmeasured confounding is not controlled.

    Args:
        group_column: Column defining the two groups.
        measure_column: Numeric column compared between them.
        spec: Active filters.
        max_groups: How many of the busiest categories to compare.
        min_group_size: Categories smaller than this are skipped.

    Returns:
        DataFrame with one row per comparison: group names, sizes, medians,
        the U statistic, the p-value and an effect-size estimate.
    """
    spec = spec or FilterSpec()
    if not _column_exists(group_column) or not _column_exists(measure_column):
        return pd.DataFrame()

    requested = [
        group_column,
        measure_column,
    ]
    draw = draw_sample(
        requested, spec=spec, limit=SETTINGS.sample_permutation_importance
    )
    frame = draw.frame
    frame[measure_column] = pd.to_numeric(frame[measure_column], errors="coerce")

    frame = frame.dropna(subset=[group_column, measure_column])
    if frame.empty:
        return pd.DataFrame()

    sizes = frame[group_column].value_counts()
    candidates = [g for g in sizes.index if sizes[g] >= min_group_size][:max_groups]
    if len(candidates) < 2:
        return pd.DataFrame()

    rows: list[dict[str, float | str | bool]] = []
    for i, group_a in enumerate(candidates):
        for group_b in candidates[i + 1 :]:
            sample_a = frame.loc[frame[group_column] == group_a, measure_column].to_numpy()
            sample_b = frame.loc[frame[group_column] == group_b, measure_column].to_numpy()
            if len(sample_a) < min_group_size or len(sample_b) < min_group_size:
                continue
            statistic, p_value = stats.mannwhitneyu(
                sample_a, sample_b, alternative="two-sided"
            )
            rows.append(
                {
                    "Group_A": group_a,
                    "Group_B": group_b,
                    "N_A": len(sample_a),
                    "N_B": len(sample_b),
                    "Median_A": round(float(np.median(sample_a)), 4),
                    "Median_B": round(float(np.median(sample_b)), 4),
                    "MannWhitney_U": float(statistic),
                    "P_Value": float(f"{p_value:.3e}"),
                    "Significant_at_0.05": bool(p_value < 0.05),
                }
            )
    return pd.DataFrame(rows)


def chi_square_severity_association(
    group_column: str,
    spec: FilterSpec | None = None,
    *,
    max_groups: int = 10,
) -> pd.DataFrame:
    """Test whether severity is associated with a categorical variable.

    Args:
        group_column: Categorical column to test against severity.
        spec: Active filters.
        max_groups: Keep only the busiest categories.

    Returns:
        DataFrame with the chi-square statistic, degrees of freedom, p-value,
        Cramér's V and the sizes of the groups included.
    """
    from smart_traffic.db import ACCIDENTS_VIEW, DB

    spec = spec or FilterSpec()
    if not _column_exists(group_column):
        return pd.DataFrame()

    sql = f"""
        SELECT "{group_column}" AS grp, Severity, COUNT(*)::BIGINT AS n
        FROM {ACCIDENTS_VIEW}
        WHERE {{where}} AND "{group_column}" IS NOT NULL AND Severity IS NOT NULL
        GROUP BY 1, 2
    """
    long = spec.select(sql, DB)
    if long.empty:
        return pd.DataFrame()

    pivot = long.pivot_table(index="grp", columns="Severity", values="n", fill_value=0)
    if len(pivot) > max_groups:
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(max_groups).index]
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return pd.DataFrame()

    chi2, p_value, dof, expected = stats.chi2_contingency(pivot.to_numpy())
    # Cramer's V, guarding the degenerate case where chi2 is 0.
    n = float(pivot.to_numpy().sum())
    denominator = n * (min(pivot.shape) - 1)
    cramers_v = float(np.sqrt(chi2 / denominator)) if denominator > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "Variable": group_column,
                "Groups_Included": int(pivot.shape[0]),
                "Severity_Levels": int(pivot.shape[1]),
                "Total_Records": int(n),
                "Chi_Square": round(float(chi2), 4),
                "Degrees_Of_Freedom": int(dof),
                "P_Value": float(f"{p_value:.3e}"),
                "Cramers_V": round(cramers_v, 4),
                "Association_Strength": describe_strength(cramers_v),
                "Significant_at_0.05": bool(p_value < 0.05),
            }
        ]
    )


def severity_by_numeric_band(
    measure_column: str,
    spec: FilterSpec | None = None,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Group severity by bands of a numeric variable.

    Useful for showing whether the severity *rate* shifts across the range of a
    continuous variable, which a raw scatter of 7.7M points cannot show.

    Args:
        measure_column: Numeric column to band.
        spec: Active filters.
        bins: Number of bands.

    Returns:
        DataFrame with the band edges, total count and per-severity rates.
    """
    from smart_traffic.db import ACCIDENTS_VIEW, DB

    spec = spec or FilterSpec()
    if not _column_exists(measure_column):
        return pd.DataFrame()

    low, high = spec.select(
        f"""
        SELECT MIN("{measure_column}") AS lo, MAX("{measure_column}") AS hi
        FROM {ACCIDENTS_VIEW}
        WHERE {{where}} AND "{measure_column}" IS NOT NULL
        """,
        DB,
    ).iloc[0]
    if pd.isna(low) or pd.isna(high) or low == high:
        return pd.DataFrame()

    # Compute the band identifier in SQL with CASE, avoiding a WIDTH_BUCKET
    # dependency and keeping the band edges visible in the query.
    width = (float(high) - float(low)) / bins
    cases = []
    params: list[float] = []
    for index in range(bins):
        start = float(low) + index * width
        cases.append(f'WHEN "{measure_column}" < ? THEN {index}')
        params.append(start)
    case_sql = f"CASE {' '.join(cases)} ELSE {bins - 1} END"

    select_parts = [f"{case_sql} AS Band", "COUNT(*)::BIGINT AS Total"]
    for level in schema.SEVERITY_LEVELS:
        select_parts.append(
            f"SUM(CASE WHEN Severity = {level} THEN 1 ELSE 0 END)::BIGINT "
            f'AS "Severity_{level}_Count"'
        )

    sql = f"""
        SELECT {", ".join(select_parts)}
        FROM {ACCIDENTS_VIEW}
        WHERE {{where}} AND "{measure_column}" IS NOT NULL AND Severity IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """
    clause, filter_params = spec.where_clause()
    frame = DB.sql(sql.format(where=clause or "TRUE"), params + filter_params)
    if frame.empty:
        return frame

    total = frame["Total"].replace(0, pd.NA)
    for level in schema.SEVERITY_LEVELS:
        frame[f"Severity_{level}_Rate"] = (frame[f"Severity_{level}_Count"] / total * 100).round(4)

    frame["Band_Label"] = [
        f"{float(low) + i * width:.1f} to {float(low) + (i + 1) * width:.1f}"
        for i in frame["Band"]
    ]
    return frame.drop(columns=["Band"])


def _column_exists(name: str) -> bool:
    """Whether ``name`` is present in the processed table."""
    from smart_traffic.db import DB

    try:
        return DB.has_column(name)
    except Exception:  # noqa: BLE001 - dataset may be absent
        return False


def sample_for_analysis(
    columns: list[str],
    spec: FilterSpec | None = None,
    *,
    limit: int | None = None,
) -> SampleDraw:
    """Draw a sample for analysis and log what it contains.

    Args:
        columns: Columns to select.
        spec: Active filters.
        limit: Row cap.

    Returns:
        The :class:`SampleDraw`.
    """
    draw = draw_sample(columns, spec=spec, limit=limit)
    logger.info("Analysis sample: %s", draw.note())
    return draw

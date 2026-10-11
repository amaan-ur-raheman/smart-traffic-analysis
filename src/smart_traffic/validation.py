"""Data quality validation.

Two layers:

* :func:`validate_processed_table` runs aggregate checks in SQL against the
  processed table, so it works on 7.7M rows without loading them.
* :func:`validate_records` validates an in-memory DataFrame. Used by tests and
  by the prediction form, and is the reference definition of "invalid".

The distinction that matters throughout: an *invalid* value is one that
violates a domain rule (severity outside 1-4, coordinates outside the USA box).
A *missing* value is one that was never recorded. They are reported separately
and never conflated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

import pandas as pd

from smart_traffic import schema
from smart_traffic.config import SETTINGS
from smart_traffic.db import ACCIDENTS_VIEW, Database
from smart_traffic.logging_config import get_logger

logger = get_logger("validation")

CheckStatus = Literal["pass", "warn", "fail", "info"]


@dataclass(slots=True)
class ValidationCheck:
    """A single named validation result.

    Attributes:
        name: Human-readable label shown in the UI.
        value: The measured value.
        status: ``pass`` / ``warn`` / ``fail`` / ``info``.
        detail: Optional explanation, especially for failures.
        reference: Published reference figure for comparison, if one exists.
    """

    name: str
    value: Any
    status: CheckStatus
    detail: str = ""
    reference: float | int | None = None

    @property
    def matches_reference(self) -> bool | None:
        """Whether ``value`` equals ``reference``, or ``None`` if no reference."""
        if self.reference is None:
            return None
        try:
            return int(round(float(self.value))) == int(self.reference)
        except (TypeError, ValueError):
            return False

    def as_row(self) -> dict[str, Any]:
        """Flatten to a dictionary for tabular display or CSV export."""
        return {
            "Check": self.name,
            "Value": self.value,
            "Status": self.status,
            "Detail": self.detail,
            "Reference": self.reference,
            "Matches reference": self.matches_reference,
        }


@dataclass(slots=True)
class ValidationReport:
    """Collection of validation checks plus derived summary counters."""

    checks: list[ValidationCheck] = field(default_factory=list)
    source: str = ""
    row_count: int = 0

    def add(
        self,
        name: str,
        value: Any,
        status: CheckStatus,
        detail: str = "",
        reference: float | int | None = None,
    ) -> None:
        """Append a check to the report."""
        self.checks.append(ValidationCheck(name, value, status, detail, reference))

    @property
    def passed(self) -> int:
        """Number of checks with status ``pass``."""
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def warnings(self) -> int:
        """Number of checks with status ``warn``."""
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def failures(self) -> int:
        """Number of checks with status ``fail``."""
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def is_healthy(self) -> bool:
        """Whether no check failed. Warnings still count as healthy."""
        return self.failures == 0

    def to_dataframe(self) -> pd.DataFrame:
        """Return all checks as a DataFrame.

        ``Value`` mixes counts, percentages and date strings, so it is
        stringified here. A single Arrow type cannot represent that column, and
        letting the render layer guess produces a confusing conversion error
        rather than a readable table.
        """
        frame = pd.DataFrame([c.as_row() for c in self.checks])
        frame["Value"] = frame["Value"].map(str)
        return frame

    def to_csv(self) -> str:
        """Return the checks as CSV text for download."""
        return self.to_dataframe().to_csv(index=False)


# ---------------------------------------------------------------------------
# Coordinate helpers (shared with ingestion)
# ---------------------------------------------------------------------------


def is_valid_latitude(lat: Any) -> bool:
    """Whether ``lat`` is a finite number inside the dataset's latitude range."""
    try:
        value = float(lat)
    except (TypeError, ValueError):
        return False
    if pd.isna(value):
        return False
    return schema.LAT_MIN <= value <= schema.LAT_MAX


def is_valid_longitude(lng: Any) -> bool:
    """Whether ``lng`` is a finite number inside the dataset's longitude range."""
    try:
        value = float(lng)
    except (TypeError, ValueError):
        return False
    if pd.isna(value):
        return False
    return schema.LNG_MIN <= value <= schema.LNG_MAX


def is_valid_severity(value: Any) -> bool:
    """Whether ``value`` is one of the four recorded severity categories."""
    try:
        return int(value) in schema.SEVERITY_LEVELS
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# In-memory record validation (reference implementation)
# ---------------------------------------------------------------------------


def validate_records(df: pd.DataFrame) -> dict[str, Any]:
    """Validate an in-memory accident DataFrame.

    Args:
        df: Frame containing at least ``Severity``, ``Start_Lat``, ``Start_Lng``.

    Returns:
        Dictionary with counts for rows, duplicates, invalid severity, invalid
        coordinates and missing values.
    """
    result: dict[str, Any] = {
        "rows": int(len(df)),
        "duplicate_ids": 0,
        "missing_ids": 0,
        "invalid_severity": 0,
        "invalid_coordinates": 0,
        "unparsed_start_time": 0,
        "invalid_duration": 0,
    }

    if df.empty:
        return result

    if schema.ID in df.columns:
        result["missing_ids"] = int(df[schema.ID].isna().sum())
        result["duplicate_ids"] = int(df[schema.ID].duplicated().sum())

    if schema.SEVERITY in df.columns:
        severity = pd.to_numeric(df[schema.SEVERITY], errors="coerce")
        invalid = severity.isna() | ~severity.isin(list(schema.SEVERITY_LEVELS))
        result["invalid_severity"] = int(invalid.sum())

    lat_col = df[schema.START_LAT] if schema.START_LAT in df.columns else None
    lng_col = df[schema.START_LNG] if schema.START_LNG in df.columns else None
    if lat_col is not None and lng_col is not None:
        lat = pd.to_numeric(lat_col, errors="coerce")
        lng = pd.to_numeric(lng_col, errors="coerce")
        bad_lat = lat.isna() | (lat < schema.LAT_MIN) | (lat > schema.LAT_MAX)
        bad_lng = lng.isna() | (lng < schema.LNG_MIN) | (lng > schema.LNG_MAX)
        result["invalid_coordinates"] = int((bad_lat | bad_lng).sum())

    if schema.START_TIME in df.columns and pd.api.types.is_datetime64_any_dtype(
        df[schema.START_TIME]
    ):
        result["unparsed_start_time"] = int(df[schema.START_TIME].isna().sum())

    if schema.DURATION_MINUTES in df.columns:
        duration = pd.to_numeric(df[schema.DURATION_MINUTES], errors="coerce")
        # A negative duration means End_Time precedes Start_Time -- a data
        # integrity problem, distinct from a missing End_Time.
        result["invalid_duration"] = int((duration < 0).sum())

    return result


# ---------------------------------------------------------------------------
# Aggregate validation via SQL
# ---------------------------------------------------------------------------

SEVERITY_COUNT_SQL: Final = """
    SELECT
        Severity,
        COUNT(*)::BIGINT AS Accidents
    FROM {table}
    WHERE Severity IS NOT NULL
    GROUP BY Severity
    ORDER BY Severity
"""


def validate_processed_table(db: Database | None = None, source: str = "") -> ValidationReport:
    """Run every aggregate validation check against the processed table.

    Args:
        db: Optional database handle. Defaults to the module singleton.
        source: Label describing where the data came from.

    Returns:
        A populated :class:`ValidationReport`.
    """
    from smart_traffic.db import DB

    handle = db or DB
    handle.ensure_views()

    report = ValidationReport(source=source or handle.parquet_path.name)

    total = int(handle.scalar(f"SELECT COUNT(*) FROM {ACCIDENTS_VIEW}") or 0)
    report.row_count = total
    report.add(
        "Total records",
        total,
        "pass" if total > 0 else "fail",
        "Rows present in the processed table.",
        reference=schema.REFERENCE_TOTALS["total_records"],
    )
    if total == 0:
        return report

    unique_ids = int(handle.scalar(f"SELECT COUNT(DISTINCT ID) FROM {ACCIDENTS_VIEW}") or 0)
    report.add(
        "Unique accident IDs",
        unique_ids,
        "pass" if unique_ids == total else "warn",
        "Distinct values in the ID column.",
        reference=schema.REFERENCE_TOTALS["unique_ids"],
    )
    report.add(
        "Duplicate IDs",
        total - unique_ids,
        "pass" if unique_ids == total else "fail",
        "Records sharing an ID with an earlier record.",
    )

    missing_ids = int(
        handle.scalar(f"SELECT COUNT(*) FROM {ACCIDENTS_VIEW} WHERE ID IS NULL") or 0
    )
    report.add("Missing accident IDs", missing_ids, "pass" if missing_ids == 0 else "fail")

    # -- severity ----------------------------------------------------------
    severity_counts = handle.sql(SEVERITY_COUNT_SQL.format(table=ACCIDENTS_VIEW))
    if schema.SEVERITY in severity_counts.columns:
        invalid_sev = int(
            handle.scalar(
                f"SELECT COUNT(*) FROM {ACCIDENTS_VIEW} "
                "WHERE Severity IS NULL OR Severity NOT IN (1, 2, 3, 4)"
            )
            or 0
        )
        report.add(
            "Invalid severity values",
            invalid_sev,
            "pass" if invalid_sev == 0 else "fail",
            "Records whose Severity is outside the documented 1-4 categories.",
        )
        observed = dict(
            zip(severity_counts["Severity"].tolist(), severity_counts["Accidents"].tolist(),
                strict=False)
        )
        for level in schema.SEVERITY_LEVELS:
            count = int(observed.get(level, 0))
            reference = schema.REFERENCE_TOTALS[f"severity_{level}"]
            delta = abs(count - reference)
            tolerance = max(1, int(reference * 0.001))
            report.add(
                f"Severity {level} count",
                count,
                "info",
                "Computed from the loaded dataset."
                if delta <= tolerance
                else f"Differs from the published reference by {count - reference:+,}.",
                reference=reference,
            )

    # -- coordinates -------------------------------------------------------
    invalid_coords = int(
        handle.scalar(
            f"SELECT COUNT(*) FROM {ACCIDENTS_VIEW} "
            f"WHERE Start_Lat IS NULL OR Start_Lng IS NULL "
            f"OR Start_Lat < {schema.LAT_MIN} OR Start_Lat > {schema.LAT_MAX} "
            f"OR Start_Lng < {schema.LNG_MIN} OR Start_Lng > {schema.LNG_MAX}"
        )
        or 0
    )
    report.add(
        "Invalid coordinates",
        invalid_coords,
        "pass" if invalid_coords == 0 else "warn",
        f"Records with missing or out-of-range Start_Lat/Start_Lng "
        f"(valid latitude {schema.LAT_MIN}-{schema.LAT_MAX}, "
        f"longitude {schema.LNG_MIN}-{schema.LNG_MAX}).",
    )

    # -- timestamps --------------------------------------------------------
    unparsed = int(
        handle.scalar(f"SELECT COUNT(*) FROM {ACCIDENTS_VIEW} WHERE Start_Time IS NULL") or 0
    )
    report.add(
        "Unparsed start timestamps",
        unparsed,
        "pass" if unparsed == 0 else "warn",
        "Rows whose Start_Time could not be parsed into a timestamp.",
    )

    if "End_Time" in handle.columns():
        bad_duration = int(
            handle.scalar(
                f"SELECT COUNT(*) FROM {ACCIDENTS_VIEW} "
                f"WHERE End_Time IS NOT NULL AND Start_Time IS NOT NULL "
                "AND End_Time < Start_Time"
            )
            or 0
        )
        missing_end = int(
            handle.scalar(f"SELECT COUNT(*) FROM {ACCIDENTS_VIEW} WHERE End_Time IS NULL") or 0
        )
        report.add(
            "Missing end timestamps",
            missing_end,
            "info" if missing_end == 0 else "warn",
            (
                "Every record carries an End_Time. Duration is therefore computable "
                "for the whole dataset -- and is still unusable as a severity "
                "predictor, because it is only known once the incident has ended."
                if missing_end == 0
                else "End_Time is absent for some records. Duration is unavailable "
                "for those rows, and duration must not be used as a severity "
                "predictor regardless."
            ),
        )
        report.add(
            "Negative durations",
            bad_duration,
            "warn" if bad_duration else "pass",
            "Records where End_Time precedes Start_Time.",
        )

    # -- temporal coverage -------------------------------------------------
    date_span = handle.sql(
        f"SELECT MIN(Start_Time) AS first, MAX(Start_Time) AS last FROM {ACCIDENTS_VIEW}"
    )
    if not date_span.empty:
        first, last = date_span["first"][0], date_span["last"][0]
        report.add(
            "Coverage start",
            str(first),
            "info",
            "Earliest parsed accident timestamp.",
        )
        report.add(
            "Coverage end",
            str(last),
            "info",
            "Latest parsed accident timestamp. Note that the final year of this "
            "dataset is partial and is not comparable to full years.",
        )

    return report


def missing_value_profile(db: Database | None = None) -> pd.DataFrame:
    """Return per-column missing counts and percentages.

    Args:
        db: Optional database handle. Defaults to the module singleton.

    Returns:
        DataFrame with ``Column``, ``Missing_Count``, ``Present_Count``,
        ``Missing_Percent``, sorted by missing count descending.
    """
    from smart_traffic.db import DB

    handle = db or DB
    handle.ensure_views()

    columns = handle.columns()
    total = int(handle.scalar(f"SELECT COUNT(*) FROM {ACCIDENTS_VIEW}") or 0)
    if not columns or total == 0:
        return pd.DataFrame(columns=["Column", "Missing_Count", "Present_Count", "Missing_Percent"])

    parts = [
        f"SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END)::BIGINT AS \"{col}\""
        for col in columns
    ]
    frame = handle.sql(f"SELECT {', '.join(parts)} FROM {ACCIDENTS_VIEW}")

    result = pd.DataFrame(
        {
            "Column": columns,
            "Missing_Count": [int(frame[col][0]) for col in columns],
        }
    )
    result["Present_Count"] = total - result["Missing_Count"]
    result["Missing_Percent"] = (result["Missing_Count"] / total * 100).round(4)
    return result.sort_values("Missing_Count", ascending=False).reset_index(drop=True)


def numeric_summary(db: Database | None = None) -> pd.DataFrame:
    """Return descriptive statistics for the known numeric columns.

    Args:
        db: Optional database handle. Defaults to the module singleton.

    Returns:
        DataFrame with count, mean, std, min, median, max and missing count.
    """
    from smart_traffic.db import DB

    handle = db or DB
    handle.ensure_views()
    available = [c for c in schema.NUMERIC_COLUMNS if handle.has_column(c)]

    if not available:
        return pd.DataFrame(
            columns=[
                "Column",
                "Count",
                "Mean",
                "Std",
                "Min",
                "Median",
                "Max",
                "Missing_Count",
            ]
        )

    selects = ", ".join(
        f'"{col}" AS "{col}"' for col in available
    )
    frame = handle.sql(f"SELECT {selects} FROM {ACCIDENTS_VIEW}")
    rows = []
    for col in available:
        series = pd.to_numeric(frame[col], errors="coerce")
        rows.append(
            {
                "Column": col,
                "Count": int(series.notna().sum()),
                "Mean": round(float(series.mean()), 4) if series.notna().any() else None,
                "Std": round(float(series.std()), 4) if series.notna().sum() > 1 else None,
                "Min": round(float(series.min()), 4) if series.notna().any() else None,
                "Median": round(float(series.median()), 4) if series.notna().any() else None,
                "Max": round(float(series.max()), 4) if series.notna().any() else None,
                "Missing_Count": int(series.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def rush_hour_hours() -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the configured rush-hour windows as inclusive start / exclusive end."""
    return SETTINGS.rush_hour_morning, SETTINGS.rush_hour_evening

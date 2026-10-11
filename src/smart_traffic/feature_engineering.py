"""Feature engineering.

Two entry points that must agree with each other:

* :func:`add_derived_features` operates on an in-memory DataFrame. This is the
  reference definition and is what the tests assert against.
* :func:`derived_feature_sql` produces the equivalent DuckDB expression list,
  used by ingestion so 7.7M rows are transformed without leaving SQL.

Leakage policy
--------------
Features are split into two groups:

``SAFE_AT_ONSET``
    Knowable when an accident is first reported. These are the only features
    permitted as severity predictors.

``POST_ONSET_ONLY``
    Knowable only after the incident ends (``Duration_*`` and anything derived
    from ``End_Time``). These are excluded from the predictive pipeline and are
    used only for description. ``assert_no_leakage`` enforces this.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from smart_traffic import schema
from smart_traffic.config import SETTINGS

#: Trailing UTC offset in a timestamp, e.g. ``"22:00:00.000000-08:00"``.
#:
#: The pattern requires a full ``HH:MM:SS`` immediately before the offset so that
#: the ``-14`` in a date like ``2016-12-14`` is never mistaken for one. The
#: optional fractional-seconds group covers the original file's microsecond
#: format, and the optional whitespace covers the variant that writes a space
#: before the offset. Rows appear in both forms, and a column containing both
#: otherwise fails to parse at all with "Mixed timezones detected".
_TZ_SUFFIX_PATTERN: Final = r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*[+-]\d{2}:?\d{2}$"

#: Month number -> abbreviated English name.
MONTH_NAMES: Final[tuple[str, ...]] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

#: pandas ``dayofweek`` index -> English name (0 = Monday).
DAY_NAMES: Final[tuple[str, ...]] = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)


def parse_mixed_timestamps(values: pd.Series) -> pd.Series:
    """Parse a timestamp column that may mix formats and timezone offsets.

    The raw ``Start_Time`` column is predominantly ``YYYY-MM-DD HH:MM:SS.ffffff``
    with a minority of rows carrying a UTC offset. Local wall-clock time is what
    the temporal analysis needs, so offsets are stripped rather than converted --
    converting would silently shift those rows' hour of day into UTC.

    Args:
        values: Raw timestamp values of mixed or unknown type.

    Returns:
        A ``datetime64[ns]`` Series with unparseable entries as ``NaT``.
    """
    series = pd.Series(values, dtype="object") if values.dtype != "object" else values

    if series.empty:
        return pd.Series([], dtype="datetime64[ns]")

    # Strip a trailing UTC offset, keeping the local wall-clock time. The
    # replacement keeps capture group 1 (the time), discarding the offset.
    if series.dtype == "object":
        series = series.astype("string").str.replace(
            _TZ_SUFFIX_PATTERN, r"\1", regex=True
        )

    try:
        return pd.to_datetime(series.astype("object"), format="mixed", errors="coerce")
    except (ValueError, TypeError):
        # Extremely old pandas without ``format="mixed"``: fall back to the
        # default inference path.
        return pd.to_datetime(series, errors="coerce")


def is_rush_hour_mask(hours: pd.Series) -> pd.Series:
    """Return a boolean mask for the configured rush-hour windows.

    The default windows are 07:00-09:59 and 16:00-18:59 -- hours 7, 8, 9 and
    16, 17, 18 -- a standard weekday commuting definition. Both stated endpoints
    are inclusive. Rows with a missing hour are ``False``.

    Args:
        hours: Integer clock hour in 0-23.

    Returns:
        Boolean Series aligned to ``hours``.
    """
    (morning_start, morning_end), (evening_start, evening_end) = (
        SETTINGS.rush_hour_morning,
        SETTINGS.rush_hour_evening,
    )
    numeric = pd.to_numeric(hours, errors="coerce")
    in_window = numeric.between(morning_start, morning_end) | numeric.between(
        evening_start, evening_end
    )
    return in_window.fillna(False).astype(bool)


def day_night_category(sunrise_sunset: pd.Series | None) -> pd.Series:
    """Classify accidents as Day / Night / Unknown.

    Args:
        sunrise_sunset: The dataset's own ``Sunrise_Sunset`` column. When absent
            or unrecognised, every row is ``"Unknown"``.

    Returns:
        Series of ``"Day"``, ``"Night"`` or ``"Unknown"``.
    """
    if sunrise_sunset is None:
        return pd.Series(SETTINGS.unknown_label, index=None, dtype="object")
    normalised = sunrise_sunset.astype("string").str.strip().str.lower()
    result = pd.Series(SETTINGS.unknown_label, index=sunrise_sunset.index, dtype="object")
    result[normalised == "night"] = SETTINGS.night_label
    result[normalised == "day"] = SETTINGS.day_label
    return result


def compute_duration_minutes(start: pd.Series, end: pd.Series | None) -> pd.Series:
    """Compute accident duration in minutes.

    Negative durations (an ``End_Time`` earlier than ``Start_Time``) are a data
    integrity problem, so they become ``NaN`` rather than a negative number.

    Args:
        start: Parsed start timestamps.
        end: Parsed end timestamps, or ``None``.

    Returns:
        Float Series of minutes, ``NaN`` where duration is unavailable or invalid.
    """
    if end is None:
        return pd.Series(float("nan"), index=start.index, dtype="float64")

    delta = (end - start).dt.total_seconds() / 60.0
    valid = delta.where(delta >= 0)
    # Rounded here so the value is byte-identical to the SQL path's
    # ``ROUND(epoch / 60, 4)``; without this the two ingestion strategies
    # disagree in the last decimal place.
    return valid.round(4)


def add_derived_features(
    df: pd.DataFrame,
    *,
    add_duration: bool = True,
    add_day_night: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``df`` with derived temporal and categorical features.

    The input frame is never mutated.

    Args:
        df: Frame that at minimum contains ``Start_Time``.
        add_duration: Whether to compute ``Duration_Minutes`` / ``Duration_Hours``.
        add_day_night: Whether to derive Day/Night from ``Sunrise_Sunset``.

    Returns:
        A new DataFrame with the derived columns appended.
    """
    out = df.copy()

    if schema.START_TIME in out.columns:
        start = parse_mixed_timestamps(out[schema.START_TIME])
        out["Start_Time"] = start

        out["Year"] = start.dt.year
        out["Month"] = start.dt.month
        out["Day"] = start.dt.day
        out["Hour"] = start.dt.hour
        # dayofweek: Monday=0 .. Sunday=6
        out["DayOfWeek"] = start.dt.dayofweek
        out["Day_Name"] = start.dt.dayofweek.map(
            lambda i: DAY_NAMES[int(i)] if pd.notna(i) else None
        )
        out["Month_Name"] = start.dt.month.map(
            lambda m: MONTH_NAMES[int(m) - 1] if pd.notna(m) and 1 <= int(m) <= 12 else None
        )
        out["Is_Weekend"] = out["DayOfWeek"].ge(5).fillna(False).astype(bool)
        out["Is_Rush_Hour"] = is_rush_hour_mask(out["Hour"])

        if add_day_night:
            source = out["Sunrise_Sunset"] if "Sunrise_Sunset" in out.columns else None
            out["Day_Night"] = day_night_category(source)
            # Nullable 1/0: Unknown stays NA so downstream imputation is visible
            # rather than being silently folded into "Day".
            out["Is_Night"] = out["Day_Night"].map(
                {SETTINGS.night_label: 1, SETTINGS.day_label: 0}
            ).astype("Int8")

    if add_duration and schema.END_TIME in out.columns:
        end = parse_mixed_timestamps(out[schema.END_TIME])
        out["End_Time"] = end
        minutes = compute_duration_minutes(out["Start_Time"], end)
        out[schema.DURATION_MINUTES] = minutes
        # Hours are derived from the *unrounded* interval rather than from the
        # already-rounded minutes. Rounding twice makes the two ingestion
        # strategies disagree in the last decimal, because pandas rounds halves
        # to even and DuckDB rounds them away from zero.
        out[schema.DURATION_HOURS] = ((end - out["Start_Time"]).dt.total_seconds() /
            3600.0).round(4)
        out.loc[out[schema.DURATION_MINUTES].isna(), schema.DURATION_HOURS] = None

    if schema.SEVERITY in out.columns:
        out[schema.SEVERITY] = pd.to_numeric(out[schema.SEVERITY], errors="coerce").astype("Int8")

    return out


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

#: Features that are only knowable after the incident has ended.
POST_ONSET_ONLY: Final[frozenset[str]] = frozenset(
    {schema.END_TIME, schema.DURATION_MINUTES, schema.DURATION_HOURS}
)

#: Identifiers and target-derived columns that must never be predictors.
FORBIDDEN_PREDICTORS: Final[frozenset[str]] = frozenset(
    {schema.ID, schema.SEVERITY, schema.DURATION_MINUTES, schema.DURATION_HOURS, schema.END_TIME}
)


def assert_no_leakage(features: list[str]) -> None:
    """Raise if a feature set contains a leaking column.

    Args:
        features: Column names intended for use as model predictors.

    Raises:
        ValueError: If any forbidden column is present.
    """
    offenders = sorted(set(features) & FORBIDDEN_PREDICTORS)
    if offenders:
        raise ValueError(
            "Target leakage detected. The following columns are not available at "
            f"accident onset and must not be predictors: {', '.join(offenders)}"
        )


# ---------------------------------------------------------------------------
# SQL projection for large-scale ingestion
# ---------------------------------------------------------------------------


def derived_feature_sql(
    table_sql: str,
    *,
    available: set[str],
    add_duration: bool = True,
    add_day_night: bool = True,
) -> str:
    """Build a SQL query deriving the same features as :func:`add_derived_features`.

    The statement is deliberately two-staged. An inner SELECT parses the mixed
    timestamp formats into real ``TIMESTAMP`` values; the outer SELECT derives
    calendar fields from them. Doing this in one stage would fail, because the
    raw file already contains a ``Start_Time`` column of type ``VARCHAR`` and
    DuckDB binds to the physical column rather than the freshly aliased one.

    Args:
        table_sql: An inline table expression, e.g. ``"read_csv('x.csv', ...)"``.
        available: Column names present in the source. Absent features are
            omitted instead of raising.
        add_duration: Whether to include duration columns.
        add_day_night: Whether to include Day/Night columns.

    Returns:
        A full ``SELECT`` statement string.
    """
    has_end = add_duration and schema.END_TIME in available

    # -- inner stage: parse timestamps ------------------------------------
    inner = [f'"{schema.ID}"', f'"{schema.SEVERITY}"']
    inner.append(f'{_parse_start_sql(schema.START_TIME)} AS "Start_Time"')
    if has_end:
        inner.append(f'{_parse_start_sql(schema.END_TIME)} AS "End_Time"')
    for column in (
        schema.START_LAT, schema.START_LNG, schema.END_LAT, schema.END_LNG,
        "Distance(mi)", "City", "County", "State", "Zipcode", "Timezone",
        "Wind_Direction", "Sunrise_Sunset", "Civil_Twilight",
        "Nautical_Twilight", "Astronomical_Twilight",
    ):
        if column in available:
            inner.append(f'"{column}"')
    for column in schema.NUMERIC_COLUMNS:
        if column in available:
            inner.append(f'TRY_CAST("{column}" AS DOUBLE) AS "{column}"')
    if schema.WEATHER_CONDITION in available:
        inner.append(f'"{schema.WEATHER_CONDITION}"')
    for column in schema.BOOLEAN_COLUMNS:
        if column in available:
            inner.append(f'TRY_CAST("{column}" AS BOOLEAN) AS "{column}"')

    inner_sql = ",\n                ".join(inner)

    # -- outer stage: derive calendar and categorical features ------------
    outer = ['"ID"', '"Severity"', '"Start_Time"']
    if has_end:
        outer.append('"End_Time"')
    for column in (
        schema.START_LAT, schema.START_LNG, schema.END_LAT, schema.END_LNG,
        "Distance(mi)", "City", "County", "State", "Zipcode", "Timezone",
        "Wind_Direction", "Sunrise_Sunset", "Civil_Twilight",
        "Nautical_Twilight", "Astronomical_Twilight",
    ):
        if column in available:
            outer.append(f'"{column}"')
    for column in schema.NUMERIC_COLUMNS:
        if column in available:
            outer.append(f'"{column}"')
    if schema.WEATHER_CONDITION in available:
        outer.append('"Weather_Condition"')
    for column in schema.BOOLEAN_COLUMNS:
        if column in available:
            outer.append(f'"{column}"')

    outer.append('CAST(EXTRACT(year FROM "Start_Time") AS INTEGER) AS "Year"')
    outer.append('CAST(EXTRACT(month FROM "Start_Time") AS INTEGER) AS "Month"')
    outer.append('CAST(EXTRACT(day FROM "Start_Time") AS INTEGER) AS "Day"')
    outer.append('CAST(EXTRACT(hour FROM "Start_Time") AS INTEGER) AS "Hour"')
    # strftime emits exactly the English names used by the pandas reference
    # implementation (Monday..Sunday, Jan..Dec).
    outer.append('strftime("Start_Time", \'%b\') AS "Month_Name"')
    outer.append('strftime("Start_Time", \'%A\') AS "Day_Name"')
    # DuckDB isodow: 1=Monday .. 7=Sunday. Shift to 0=Monday so this matches
    # the pandas ``dt.dayofweek`` convention used by the chunked path.
    outer.append(
        '(CAST(EXTRACT(isodow FROM "Start_Time") AS INTEGER) - 1) AS "DayOfWeek"'
    )

    (morning_start, morning_end), (evening_start, evening_end) = (
        SETTINGS.rush_hour_morning,
        SETTINGS.rush_hour_evening,
    )
    hour_expr = 'CAST(EXTRACT(hour FROM "Start_Time") AS INTEGER)'
    # DuckDB ``dow``: 0=Sunday .. 6=Saturday. Weekend is 0 (Sun) or 6 (Sat).
    dow_expr = 'CAST(EXTRACT(dow FROM "Start_Time") AS INTEGER)'
    outer.append(f"({dow_expr} IN (0, 6)) AS \"Is_Weekend\"")
    outer.append(
        f"(({hour_expr} BETWEEN {morning_start} AND {morning_end}) "
        f"OR ({hour_expr} BETWEEN {evening_start} AND {evening_end})) "
        'AS "Is_Rush_Hour"'
    )

    if add_day_night and "Sunrise_Sunset" in available:
        normalised = 'lower(trim(CAST("Sunrise_Sunset" AS VARCHAR)))'
        outer.append(
            f"CASE {normalised} "
            f"WHEN 'night' THEN '{SETTINGS.night_label}' "
            f"WHEN 'day' THEN '{SETTINGS.day_label}' "
            f"ELSE '{SETTINGS.unknown_label}' END AS \"Day_Night\""
        )
        outer.append(
            f"CASE {normalised} WHEN 'night' THEN 1 WHEN 'day' THEN 0 "
            'END AS "Is_Night"'
        )

    if has_end:
        outer.append(
            'CASE WHEN "End_Time" IS NULL OR "End_Time" < "Start_Time" THEN NULL '
            'ELSE ROUND(EXTRACT(epoch FROM ("End_Time" - "Start_Time")) / 60.0, 4) '
            f'END AS "{schema.DURATION_MINUTES}"'
        )
        # Hours come from the same raw interval, not from the rounded minutes,
        # so both ingestion strategies agree bit for bit.
        outer.append(
            'CASE WHEN "End_Time" IS NULL OR "End_Time" < "Start_Time" THEN NULL '
            'ELSE ROUND(EXTRACT(epoch FROM ("End_Time" - "Start_Time")) / 3600.0, 4) '
            f'END AS "{schema.DURATION_HOURS}"'
        )

    outer_sql = ",\n            ".join(outer)
    return (
        f"SELECT\n            {outer_sql}\n        FROM (\n"
        f"            SELECT\n                {inner_sql}\n"
        f"            FROM {table_sql}\n        ) AS parsed"
    )


def _parse_start_sql(column: str) -> str:
    """Return a DuckDB expression parsing a mixed-format timestamp column.

    A trailing UTC offset is trimmed before parsing so the local wall-clock time
    is preserved -- the same choice the pandas path makes, and the reason hour-of-
    day analytics stay comparable across both ingestion strategies.

    Args:
        column: Bare, quoted column name.

    Returns:
        A SQL expression yielding ``TIMESTAMP`` or ``NULL``.
    """
    # The capture group keeps the time and discards the offset, so local
    # wall-clock time survives. Requiring HH:MM:SS before the offset stops the
    # day component of a date from ever being mistaken for one.
    stripped = (
        f"regexp_replace(CAST(\"{column}\" AS VARCHAR), "
        r"'(\d{2}:\d{2}:\d{2}(\.\d+)?)\s*[+-]\d{2}:?\d{2}$', '\1')"
    )
    return (
        f"COALESCE(try_strptime({stripped}, '%Y-%m-%d %H:%M:%S.%f'), "
        f"try_strptime({stripped}, '%Y-%m-%d %H:%M:%S'), "
        f"try_strptime({stripped}, '%Y-%m-%dT%H:%M:%S'))"
    )


def rush_hour_definition_text() -> str:
    """Return a human-readable statement of the rush-hour definition in use."""
    (morning_start, morning_end), (evening_start, evening_end) = (
        SETTINGS.rush_hour_morning,
        SETTINGS.rush_hour_evening,
    )
    return (
        f"Rush hour is defined as {morning_start:02d}:00-{morning_end:02d}:59 and "
        f"{evening_start:02d}:00-{evening_end:02d}:59 local time, inclusive of both "
        f"endpoints."
    )

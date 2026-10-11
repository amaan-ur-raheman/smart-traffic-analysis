"""Canonical schema knowledge for the US Accidents dataset.

The original dataset (``US_Accidents_March23.csv``) has 46 columns. This module
defines which of those the application knows about and how each one is typed,
grouped, and used. Nothing here reads a file -- these are declarations used by
validation, ingestion, feature engineering and modelling.

Design rule: the *required* list is deliberately small. A file missing a column
outside this list can still be processed, and the missing ones are reported as
warnings rather than hard failures.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

ID: Final = "ID"

# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------

SEVERITY: Final = "Severity"
SEVERITY_LEVELS: Final[tuple[int, ...]] = (1, 2, 3, 4)

# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------

START_TIME: Final = "Start_Time"
END_TIME: Final = "End_Time"

TIMESTAMP_COLUMNS: Final[tuple[str, ...]] = (START_TIME, END_TIME)

# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------

START_LAT: Final = "Start_Lat"
START_LNG: Final = "Start_Lng"
END_LAT: Final = "End_Lat"
END_LNG: Final = "End_Lng"

COORDINATE_COLUMNS: Final[tuple[str, ...]] = (START_LAT, START_LNG, END_LAT, END_LNG)

# The contiguous USA (including Alaska and Hawaii) bounding box. Coordinates
# outside it are treated as invalid rather than silently clipped.
LAT_MIN: Final[float] = 18.0
LAT_MAX: Final[float] = 72.0
LNG_MIN: Final[float] = -180.0
LNG_MAX: Final[float] = -40.0

# --------------------------------------------------------------------------
# Numeric measurement columns
# --------------------------------------------------------------------------

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "Distance(mi)",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
)

DURATION_MINUTES: Final = "Duration_Minutes"
DURATION_HOURS: Final = "Duration_Hours"

# --------------------------------------------------------------------------
# Categorical columns
# --------------------------------------------------------------------------

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "Description",
    "Street",
    "City",
    "County",
    "State",
    "Zipcode",
    "Timezone",
    "Weather_Condition",
    "Wind_Direction",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
)

STATE: Final = "State"
CITY: Final = "City"
WEATHER_CONDITION: Final = "Weather_Condition"

# --------------------------------------------------------------------------
# Boolean road / amenity indicators (True / False in the raw file)
# --------------------------------------------------------------------------

BOOLEAN_COLUMNS: Final[tuple[str, ...]] = (
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Turning_Loop",
)

# Friendly display labels used on the road-features dashboard.
ROAD_FEATURE_LABELS: Final[dict[str, str]] = {
    "Traffic_Signal": "Traffic signals present",
    "Crossing": "Crossings present",
    "Junction": "Junctions present",
    "Stop": "Stop signs present",
    "Railway": "Railways present",
    "Give_Way": "Give-way signs present",
    "Roundabout": "Roundabouts present",
    "Bump": "Road bumps present",
    "No_Exit": "No-exit signs present",
    "Amenity": "Amenity nearby",
    "Station": "Station nearby",
    "Traffic_Calming": "Traffic calming present",
    "Turning_Loop": "Turning loop present",
}

# --------------------------------------------------------------------------
# Required columns -- absent means we cannot meaningfully process the file
# --------------------------------------------------------------------------

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (ID, SEVERITY, START_TIME)

# --------------------------------------------------------------------------
# Optional-but-expected columns -- absence is a warning, not a failure
# --------------------------------------------------------------------------

EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    (END_TIME, START_LAT, START_LNG, END_LAT, END_LNG, "Distance(mi)", "City", "State",
        WEATHER_CONDITION)
    + NUMERIC_COLUMNS
    + BOOLEAN_COLUMNS
)

# --------------------------------------------------------------------------
#: Exact header of the shipped ``US_Accidents_March23.csv`` file, verified
#: against the real 3 GB dataset. This is the authority on column names: the
#: generated test fixture draws its header from this list, so a typo in the
#: constants below would be invisible to the test suite and would silently drop
#: a column during ingestion. :func:`matches_source_header` exists to catch
#: exactly that class of bug.
CANONICAL_SOURCE_HEADER: Final[tuple[str, ...]] = (
    "ID", "Source", "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng",
    "End_Lat", "End_Lng", "Distance(mi)", "Description", "Street", "City", "County",
    "State", "Zipcode", "Country", "Timezone", "Airport_Code", "Weather_Timestamp",
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Direction", "Wind_Speed(mph)", "Precipitation(in)",
    "Weather_Condition", "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
    "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop", "Sunrise_Sunset", "Civil_Twilight",
    "Nautical_Twilight", "Astronomical_Twilight",
)


def matches_source_header(columns: Iterable[str]) -> list[str]:
    """Return names this module expects that the supplied file lacks.

    Args:
        columns: Header of the file actually supplied.

    Returns:
        Columns referenced by this module that are absent from ``columns``. A
        non-empty result means a constant here is spelled wrong, and ingestion
        would silently drop that column.
    """
    present = set(columns)
    expected = (
        set(REQUIRED_COLUMNS)
        | set(EXPECTED_COLUMNS)
        | set(NUMERIC_COLUMNS)
        | set(BOOLEAN_COLUMNS)
        | set(CATEGORICAL_COLUMNS)
    )
    return sorted(expected - present)


# Full ordered list of columns in the original dataset (documentation value)
# --------------------------------------------------------------------------

ORIGINAL_COLUMNS: Final[tuple[str, ...]] = (
    "ID",
    # `Source` is present in the shipped file and is deliberately excluded from
    # the processed table: it names the reporting feed, not anything about the
    # accident.
    "Severity",
    "Start_Time",
    "End_Time",
    "Start_Lat",
    "Start_Lng",
    "End_Lat",
    "End_Lng",
    "Distance(mi)",
    "Description",
    "Street",
    "City",
    "County",
    "State",
    "Zipcode",
    "Timezone",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Direction",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Condition",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Turning_Loop",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
)

# --------------------------------------------------------------------------
# Reference figures published for the original dataset. These are used ONLY
# to flag discrepancies on the data-quality page; they are never used as
# analytical output and never displayed as this project's results.
# --------------------------------------------------------------------------

REFERENCE_TOTALS: Final[dict[str, int]] = {
    "total_records": 7_728_394,
    "unique_ids": 7_728_394,
    "severity_1": 67_366,
    "severity_2": 6_156_981,
    "severity_3": 1_299_337,
    "severity_4": 204_710,
}

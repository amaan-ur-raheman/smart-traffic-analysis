"""Synthetic dataset generator.

The original CSV is ~3 GB and must be downloaded from Kaggle. This module
produces a **schema-faithful** stand-in so the entire pipeline -- ingestion,
validation, analytics, clustering, modelling, and the Streamlit UI -- can be
exercised and tested without it.

What this is NOT: the synthetic file contains no real accidents. Its figures
must never be reported as results. Generated rows are stamped with a
``SYNTH-`` ID prefix and the app marks any run on synthetic data as such in the
sidebar and on every page header.

The generator deliberately reproduces the *structural* characteristics that
matter for exercising the code: mixed timestamp formats, timezone offsets on a
subset of rows, missing weather values, an imbalanced severity distribution,
and boolean road indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from smart_traffic import schema
from smart_traffic.logging_config import get_logger

logger = get_logger("synthetic")

#: Prefix applied to every generated ID so synthetic rows are identifiable.
SYNTHETIC_ID_PREFIX = "SYNTH-"

#: Severity probabilities chosen to mimic a strongly imbalanced target:
#: most records at severity 2, a long tail at 3 and 4, few at 1.
_SEVERITY_WEIGHTS = (0.009, 0.797, 0.168, 0.026)

_STATES = (
    "CA", "FL", "TX", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
)
_STATE_WEIGHTS = np.array(
    [
        0.225, 0.114, 0.075, 0.048, 0.040, 0.033, 0.031, 0.030, 0.028, 0.023,
        0.021, 0.020, 0.019, 0.018, 0.017, 0.016, 0.015, 0.015, 0.014, 0.013,
    ]
)
_STATE_WEIGHTS = _STATE_WEIGHTS / _STATE_WEIGHTS.sum()

_CITIES_BY_STATE = {
    "CA": ("Los Angeles", "San Francisco", "San Diego", "San Jose", "Sacramento"),
    "FL": ("Miami", "Orlando", "Tampa", "Jacksonville", "Fort Lauderdale"),
    "TX": ("Houston", "Dallas", "Austin", "San Antonio", "Fort Worth"),
    "NY": ("New York", "Buffalo", "Rochester", "Syracuse", "Albany"),
    "PA": ("Philadelphia", "Pittsburgh", "Allentown", "Erie", "Harrisburg"),
    "IL": ("Chicago", "Aurora", "Naperville", "Joliet", "Rockford"),
    "OH": ("Columbus", "Cleveland", "Cincinnati", "Toledo", "Akron"),
    "GA": ("Atlanta", "Augusta", "Columbus", "Macon", "Savannah"),
    "NC": ("Charlotte", "Raleigh", "Greensboro", "Durham", "Winston-Salem"),
    "MI": ("Detroit", "Grand Rapids", "Warren", "Ann Arbor", "Lansing"),
}

_DEFAULT_CITIES = ("Springfield", "Riverside", "Fairview", "Greenville", "Bristol")

_WEATHER = (
    "Fair", "Mostly Cloudy", "Cloudy", "Partly Cloudy", "Overcast",
    "Light Rain", "Heavy Rain", "Fog", "Smoke", "Haze", "Thunderstorm",
    "Snow", "Light Snow", "Wintry Mix", "Blowing Snow", "Drizzle",
    "Rain and Thunderstorm", "Light Rain Shower", "Scattered Showers",
)
_WEATHER_WEIGHTS = np.array(
    [
        0.33, 0.13, 0.106, 0.085, 0.06, 0.045, 0.02, 0.04, 0.015, 0.02,
        0.012, 0.025, 0.012, 0.008, 0.005, 0.015, 0.008, 0.03, 0.02,
    ]
)
_WEATHER_WEIGHTS = _WEATHER_WEIGHTS / _WEATHER_WEIGHTS.sum()

_WIND_DIRECTIONS = ("Calm", "N", "NE", "E", "SE", "S", "SW", "W", "NW", "Variable")
_TWILIGHTS = ("Day", "Civil Twilight", "Nautical Twilight", "Astronomical Twilight")
_TZ_OFFSETS = ("", "", "", "", "", " -05:00", " -06:00", " -07:00", " -08:00", " +00:00")


@dataclass(slots=True)
class SyntheticSpec:
    """Parameters controlling generation.

    Attributes:
        rows: Number of records to produce.
        start: First accident timestamp.
        end: Last accident timestamp (exclusive).
        seed: Random seed controlling all draws.
        duplicate_fraction: Fraction of rows duplicated by ``ID`` to exercise
            duplicate detection.
        bad_severity_fraction: Fraction of rows given an out-of-range severity.
        bad_coordinate_fraction: Fraction of rows given impossible coordinates.
        null_weather_fraction: Fraction of rows with missing weather readings.
    """

    rows: int = 50_000
    start: str = "2016-02-01"
    end: str = "2023-04-01"
    seed: int = 20240101
    duplicate_fraction: float = 0.0
    bad_severity_fraction: float = 0.0
    bad_coordinate_fraction: float = 0.0
    null_weather_fraction: float = 0.15


_STREET_NAMES: tuple[str, ...] = ("Main", "Oak", "Elm", "Cedar")


def _street_names(rng: np.random.Generator, n: int) -> list[str]:
    """Generate plausible street names, e.g. ``"142 Oak St"``."""
    numbers = rng.integers(1, 9999, n)
    picks = rng.integers(0, len(_STREET_NAMES), n)
    return [
        f"{int(number)} {_STREET_NAMES[int(pick)]} St"
        for number, pick in zip(numbers, picks, strict=True)
    ]


def generate_accidents(spec: SyntheticSpec | None = None) -> pd.DataFrame:
    """Generate a schema-faithful synthetic accident frame.

    Args:
        spec: Generation parameters. Defaults to :class:`SyntheticSpec`.

    Returns:
        A DataFrame whose columns match the original CSV exactly.
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n = int(spec.rows)

    start_ts = pd.Timestamp(spec.start)
    end_ts = pd.Timestamp(spec.end)
    span = int((end_ts - start_ts).total_seconds())

    # -- timestamps --------------------------------------------------------
    seconds = rng.integers(0, max(span, 1), n)
    base = start_ts + pd.to_timedelta(seconds, unit="s")
    # Keep the original file's quirk: 2023 coverage stops in March.
    base = base.where(base < pd.Timestamp("2023-04-01"), pd.Timestamp("2023-04-01"))

    start_strings: list[str] = []
    end_strings: list[str] = []
    hour = base.hour.to_numpy()
    for i in range(n):
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        start_strings.append(base[i].strftime(fmt))
        duration_minutes = float(np.clip(rng.lognormal(1.1, 0.9), 5, 24 * 60))
        end_time = base[i] + pd.Timedelta(minutes=duration_minutes)
        offset = _TZ_OFFSETS[rng.integers(0, len(_TZ_OFFSETS))]
        if offset and i % 5 == 0:
            # A minority of rows carry a timezone offset, as in the original.
            start_strings[-1] = start_strings[-1] + offset
            end_strings.append(end_time.strftime(fmt) + offset)
        else:
            end_strings.append(end_time.strftime(fmt))

    # -- geography ---------------------------------------------------------
    state_idx = rng.choice(len(_STATES), size=n, p=_STATE_WEIGHTS)
    states = np.array(_STATES)[state_idx]
    cities = np.empty(n, dtype=object)
    lats = np.empty(n)
    lngs = np.empty(n)
    for i, st in enumerate(states):
        pool = _CITIES_BY_STATE.get(str(st), _DEFAULT_CITIES)
        cities[i] = pool[rng.integers(0, len(pool))]
        lats[i] = rng.uniform(25.0, 48.0)
        lngs[i] = rng.uniform(-124.0, -68.0)

    # -- weather -----------------------------------------------------------
    weather_idx = rng.choice(len(_WEATHER), size=n, p=_WEATHER_WEIGHTS)
    weather = np.array(_WEATHER)[weather_idx]

    temperature = np.clip(rng.normal(58, 20, n), -40, 120)
    humidity = np.clip(rng.normal(65, 20, n), 1, 100)
    pressure = np.clip(rng.normal(30.0, 0.25, n), 28.5, 31.2)
    visibility = np.clip(rng.lognormal(2.0, 0.5, n), 0.1, 30)
    wind_speed = np.clip(rng.gamma(2.0, 4.0, n), 0, 70)
    precipitation = np.clip(rng.gamma(1.2, 0.15, n), 0, 4)

    # Correlate visibility and precipitation with the weather label so the
    # weather analysis has a genuine signal to find.
    wet = np.isin(weather, ["Light Rain", "Heavy Rain", "Drizzle", "Rain and Thunderstorm",
                            "Light Rain Shower", "Scattered Showers"])
    visibility = np.where(wet, np.minimum(visibility, 5.0), visibility)
    precipitation = np.where(wet, np.maximum(precipitation, 0.05), precipitation)
    visibility = np.where(weather == "Fog", np.minimum(visibility, 1.0), visibility)

    # Night is more common for the small hours; drive Sunrise_Sunset from hour.
    sunrise_sunset = np.where((hour >= 6) & (hour < 19), "Day", "Night")

    frame = pd.DataFrame(
        {
            schema.ID: [f"{SYNTHETIC_ID_PREFIX}{i:09d}" for i in range(n)],
            # Present in the real file and deliberately excluded from the
            # processed table: Source names the reporting feed, Country is
            # always "US", Airport_Code and Weather_Timestamp are near-empty.
            "Source": np.array(["Source1", "Source2"])[rng.integers(0, 2, n)],
            "Country": np.array(["US"])[rng.integers(0, 1, n)],
            "Airport_Code": np.array(["", "", "KLAF"])[rng.integers(0, 3, n)],
            schema.SEVERITY: rng.choice([1, 2, 3, 4], size=n, p=_SEVERITY_WEIGHTS),
            schema.START_TIME: start_strings,
            schema.END_TIME: end_strings,
            schema.START_LAT: np.round(lats, 6),
            schema.START_LNG: np.round(lngs, 6),
            schema.END_LAT: np.round(lats + rng.normal(0, 0.01, n), 6),
            schema.END_LNG: np.round(lngs + rng.normal(0, 0.01, n), 6),
            "Distance(mi)": np.round(rng.lognormal(1.0, 1.0, n), 3),
            "Description": ["Accident reported via the traffic event feed."] * n,
            "Street": _street_names(rng, n),
            "City": cities,
            "County": ["County"] * n,
            "State": states,
            "Zipcode": [str(rng.integers(10001, 99950)) for _ in range(n)],
            "Timezone": ["US/Eastern", "US/Central", "US/Mountain", "US/Pacific"][:1] * n,
            "Temperature(F)": np.round(temperature, 1),
            "Wind_Chill(F)": np.round(temperature - rng.uniform(0, 8, n), 1),
            "Humidity(%)": np.round(humidity, 0),
            "Pressure(in)": np.round(pressure, 2),
            "Visibility(mi)": np.round(visibility, 3),
            "Wind_Direction": np.array(_WIND_DIRECTIONS)[rng.integers(0, len(_WIND_DIRECTIONS), n)],
            "Wind_Speed(mph)": np.round(wind_speed, 3),
            "Precipitation(in)": np.round(precipitation, 3),
            schema.WEATHER_CONDITION: weather,
            "Sunrise_Sunset": sunrise_sunset,
            "Civil_Twilight": np.array(_TWILIGHTS)[rng.integers(0, 4, n)],
            "Nautical_Twilight": np.array(_TWILIGHTS)[rng.integers(0, 4, n)],
            "Astronomical_Twilight": np.array(_TWILIGHTS)[rng.integers(0, 4, n)],
        }
    )

    # Boolean road indicators with plausible marginal frequencies.
    indicator_rates = {
        "Amenity": 0.13, "Bump": 0.005, "Crossing": 0.11, "Give_Way": 0.04,
        "Junction": 0.07, "No_Exit": 0.006, "Railway": 0.04, "Roundabout": 0.006,
        "Station": 0.08, "Stop": 0.03, "Traffic_Calming": 0.012,
        "Traffic_Signal": 0.15, "Turning_Loop": 0.002,
    }
    for column, rate in indicator_rates.items():
        frame[column] = rng.random(n) < rate

    # Weather_Timestamp is the observation time, unrelated to the accident.
    frame["Weather_Timestamp"] = [
        (base[i] - pd.Timedelta(minutes=float(rng.integers(0, 60)))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        for i in range(n)
    ]

    # -- missingness -------------------------------------------------------
    for column in ("Wind_Chill(F)", "Visibility(mi)", "Precipitation(in)",
                   "Wind_Speed(mph)", "Pressure(in)", "Humidity(%)"):
        mask = rng.random(n) < (spec.null_weather_fraction / 2)
        frame.loc[mask, column] = np.nan
    frame.loc[rng.random(n) < 0.44, schema.END_LNG] = np.nan
    frame.loc[rng.random(n) < 0.44, schema.END_LAT] = np.nan
    frame.loc[rng.random(n) < 0.001, "Description"] = None

    # -- deliberate defects (only when asked for) --------------------------
    if spec.bad_severity_fraction > 0:
        count = int(n * spec.bad_severity_fraction)
        if count:
            idx = rng.choice(n, count, replace=False)
            frame.loc[idx, schema.SEVERITY] = rng.choice([0, 5, 7], count)

    if spec.bad_coordinate_fraction > 0:
        count = int(n * spec.bad_coordinate_fraction)
        if count:
            idx = rng.choice(n, count, replace=False)
            frame.loc[idx, schema.START_LAT] = rng.choice([0.0, 95.0, -200.0], count)
            frame.loc[idx, schema.START_LNG] = rng.choice([0.0, 190.0, -400.0], count)

    if spec.duplicate_fraction > 0:
        count = int(n * spec.duplicate_fraction)
        if count:
            idx = rng.choice(n, count, replace=False)
            duplicates = frame.iloc[idx].copy()
            frame = pd.concat([frame, duplicates], ignore_index=True)

    return frame[list(schema.CANONICAL_SOURCE_HEADER)]


def write_synthetic_csv(
    path: Path,
    spec: SyntheticSpec | None = None,
) -> Path:
    """Generate a synthetic CSV and write it to ``path``.

    Args:
        path: Destination path. Parent directories are created.
        spec: Generation parameters.

    Returns:
        The path written.
    """
    spec = spec or SyntheticSpec()
    frame = generate_accidents(spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info(
        "Wrote %s synthetic rows to %s (%.1f MB)", f"{len(frame):,}", path,
        path.stat().st_size / 1e6,
    )
    return path


def is_synthetic(path: Path | None) -> bool:
    """Whether the given source CSV is a generated synthetic file.

    Detection is based on the ``SYNTH-`` ID prefix in the first data row.

    Args:
        path: Candidate CSV path.

    Returns:
        ``True`` when the file was produced by this module.
    """
    if path is None or not Path(path).exists():
        return False
    try:
        first = pd.read_csv(path, nrows=50, usecols=[schema.ID])
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    return bool(first[schema.ID].astype(str).str.startswith(SYNTHETIC_ID_PREFIX).any())

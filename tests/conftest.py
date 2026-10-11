"""Shared pytest fixtures.

Tests run against a small synthetic dataset built in a temporary directory. The
full 3 GB CSV is never required, and never downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smart_traffic import schema
from smart_traffic.synthetic import SyntheticSpec, generate_accidents


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    """A 200-row synthetic frame with the original CSV's exact columns."""
    return generate_accidents(SyntheticSpec(rows=200, seed=1))


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    """A frame containing deliberate data defects.

    Includes out-of-range severities, impossible coordinates, a duplicated ID,
    mixed timestamp formats and timezone offsets.
    """
    frame = generate_accidents(SyntheticSpec(rows=100, seed=2))

    frame.loc[0, schema.SEVERITY] = 7
    frame.loc[1, schema.SEVERITY] = 0
    frame.loc[2, schema.START_LAT] = 95.0
    frame.loc[3, schema.START_LNG] = -400.0
    frame.loc[4, schema.START_TIME] = "not-a-timestamp"
    frame.loc[5, schema.START_TIME] = "2018-06-15 14:30:00"
    frame.loc[6, schema.START_TIME] = "2018-06-15T14:30:00"
    frame.loc[7, schema.START_TIME] = "2018-06-15 14:30:00-05:00"
    frame.loc[8, schema.END_TIME] = "2016-01-01 00:00:00"  # before Start_Time

    duplicate = frame.iloc[[9, 10]].copy()
    duplicate[schema.ID] = frame.loc[9, schema.ID]
    return pd.concat([frame, duplicate], ignore_index=True)


@pytest.fixture
def known_temporal_frame() -> pd.DataFrame:
    """A frame with hand-picked timestamps for exact feature assertions."""
    return pd.DataFrame(
        {
            schema.ID: ["A", "B", "C", "D", "E", "F", "G"],
            schema.SEVERITY: [1, 2, 3, 4, 2, 3, 1],
            #  0=Monday 07:00 rush   1=Tuesday 08:00 rush
            #  2=Wednesday 17:00 rush 3=Thursday 12:00 not rush
            #  4=Friday 18:00 rush    5=Saturday 03:00 weekend
            #  6=Sunday 23:00 weekend
            schema.START_TIME: [
                "2021-01-04 07:15:00.000000",  # Monday 07:15
                "2021-01-05 08:45:00.000000",  # Tuesday 08:45
                "2021-01-06 17:30:00.000000",  # Wednesday 17:30
                "2021-01-07 12:00:00.000000",  # Thursday 12:00
                "2021-01-08 18:00:00.000000",  # Friday 18:00
                "2021-01-09 03:00:00.000000",  # Saturday 03:00
                "2021-01-10 23:00:00.000000",  # Sunday 23:00
            ],
            schema.END_TIME: [
                "2021-01-04 07:45:00.000000",  # +30 min
                "2021-01-05 09:00:00.000000",  # +15 min
                "2021-01-06 18:00:00.000000",  # +30 min
                "2021-01-07 14:00:00.000000",  # +120 min
                "2021-01-08 18:05:00.000000",  # +5 min
                "2021-01-09 04:00:00.000000",  # +60 min
                "2021-01-10 23:30:00.000000",  # +30 min
            ],
            "Sunrise_Sunset": [
                "Night", "Day", "Day", "Day", "Night", "Night", "Night",
            ],
            schema.START_LAT: [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0],
            schema.START_LNG: [-75.0, -76.0, -77.0, -78.0, -79.0, -80.0, -81.0],
        }
    )


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """A 500-row synthetic CSV on disk."""
    frame = generate_accidents(SyntheticSpec(rows=500, seed=3))
    path = tmp_path / "tiny.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def processed_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a processed Parquet table from a small synthetic CSV.

    Redirects every managed directory into the test's temporary directory via
    the environment variables ``Paths`` reads at property-access time. Rebinding
    ``config.PATHS`` would not work: modules import the object by name, and the
    properties resolve lazily, so the environment is the correct seam.

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        Path to the built Parquet file.
    """
    import smart_traffic.db as db_module
    from smart_traffic import config as config_module
    from smart_traffic.ingestion import build_from_csv
    from smart_traffic.synthetic import SyntheticSpec, write_synthetic_csv

    monkeypatch.setenv("STA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("STA_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("STA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("STA_LOG_DIR", str(tmp_path / "logs"))

    config_module.PATHS.ensure_dirs()

    csv_path = write_synthetic_csv(
        tmp_path / "data" / "synthetic" / "test.csv",
        SyntheticSpec(rows=800, seed=4),
    )
    build_from_csv(csv_path, method="chunked", chunk_size=200)

    # Every test gets a fresh connection bound to the temporary Parquet file.
    handle = db_module.Database(
        db_path=tmp_path / "data" / "processed" / "test.duckdb",
        parquet_path=config_module.PATHS.accidents_parquet,
    )
    monkeypatch.setattr(db_module, "DB", handle)
    return config_module.PATHS.accidents_parquet

@pytest.fixture
def test_db():
    """Return the database handle bound to the temporary Parquet file.

    Imported lazily rather than at module scope: a module-level ``from
    smart_traffic.db import DB`` would capture the original singleton before
    :func:`processed_table` rebinds it, silently testing against the real
    project dataset.
    """
    import smart_traffic.db as db_module

    db_module.DB.reset()
    yield db_module.DB
    db_module.DB.reset()

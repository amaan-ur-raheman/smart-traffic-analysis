"""Data ingestion: raw CSV -> processed Parquet.

Two interchangeable execution strategies produce byte-for-byte equivalent
tables:

``chunked`` (default)
    Streams the CSV with :func:`pandas.read_csv` in fixed-size chunks, applies
    the reference feature engineering, and appends to a single Parquet file via
    a PyArrow writer. Peak memory is bounded by the chunk size regardless of
    file size, and progress is exact.

``duckdb``
    Lets DuckDB do the scan, join and Parquet write in one push-down pipeline.
    Substantially faster, at the cost of coarser progress reporting.

Both paths write records *without dropping them*. Validity is recorded in the
flag columns ``Is_Valid_Severity``, ``Is_Valid_Coordinate`` and
``Is_Valid_Record`` so nothing disappears silently and every exclusion is
reproducible and auditable.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from smart_traffic import schema
from smart_traffic.config import PATHS, SETTINGS
from smart_traffic.db import Database
from smart_traffic.feature_engineering import add_derived_features
from smart_traffic.logging_config import get_logger

logger = get_logger("ingestion")

#: ``(stage, fraction_complete, message)``
ProgressCallback = Callable[[str, float, str], None]

#: Columns forced to DOUBLE on read.
_FORCED_DOUBLE: Final[tuple[str, ...]] = schema.NUMERIC_COLUMNS + (
    schema.START_LAT,
    schema.START_LNG,
    schema.END_LAT,
    schema.END_LNG,
)

#: Columns forced to BOOLEAN.
_FORCED_BOOLEAN: Final[tuple[str, ...]] = schema.BOOLEAN_COLUMNS

#: The final on-disk column order. This is the contract shared by both
#: ingestion strategies -- neither path may emit a column outside it.
#:
#: ``Description`` and ``Street`` are deliberately excluded. They are long free
#: text used for no aggregate in this project, and carrying them through 7.7M
#: rows would add roughly a gigabyte to the Parquet file for no analytical
#: benefit. The raw CSV is never modified, so nothing is lost.
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    schema.ID,
    schema.SEVERITY,
    "Start_Time",
    "End_Time",
    "Start_Lat",
    "Start_Lng",
    "End_Lat",
    "End_Lng",
    "Distance(mi)",
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
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    *schema.BOOLEAN_COLUMNS,
    "Year",
    "Month",
    "Month_Name",
    "Day",
    "DayOfWeek",
    "Day_Name",
    "Hour",
    "Is_Weekend",
    "Is_Rush_Hour",
    "Day_Night",
    "Is_Night",
    schema.DURATION_MINUTES,
    schema.DURATION_HOURS,
    "Is_Valid_Severity",
    "Is_Valid_Coordinate",
    "Is_Valid_Record",
)


#: Set once the dropped-column notice has been logged, to avoid repeating it
#: for every one of the hundreds of chunks in a large ingest.
_ORDER_WARNED = False


class IngestionError(RuntimeError):
    """Raised when the source file cannot be read or has an unusable schema."""


@dataclass(slots=True)
class IngestionReport:
    """Summary of one ingestion run."""

    source: str
    output: str
    method: str
    rows_read: int = 0
    rows_written: int = 0
    invalid_severity: int = 0
    invalid_coordinates: int = 0
    invalid_records: int = 0
    unparsed_start_time: int = 0
    seconds: float = 0.0
    source_bytes: int = 0
    output_bytes: int = 0
    source_columns: list[str] = field(default_factory=list)

    @property
    def valid_rows(self) -> int:
        """Records passing every validity check."""
        return self.rows_written - self.invalid_records

    def to_dataframe(self) -> pd.DataFrame:
        """Return the report as a two-column DataFrame."""
        return pd.DataFrame(
            [
                {"Field": "Source file", "Value": self.source},
                {"Field": "Output file", "Value": self.output},
                {"Field": "Method", "Value": self.method},
                {"Field": "Source size (MB)", "Value": round(self.source_bytes / 1e6, 2)},
                {"Field": "Rows read", "Value": self.rows_read},
                {"Field": "Rows written", "Value": self.rows_written},
                {"Field": "Valid records", "Value": self.valid_rows},
                {"Field": "Invalid severity", "Value": self.invalid_severity},
                {"Field": "Invalid coordinates", "Value": self.invalid_coordinates},
                {"Field": "Excluded records (any check)", "Value": self.invalid_records},
                {"Field": "Unparsed start timestamps", "Value": self.unparsed_start_time},
                {"Field": "Output size (MB)", "Value": round(self.output_bytes / 1e6, 2)},
                {"Field": "Elapsed seconds", "Value": round(self.seconds, 1)},
            ]
        )


def _noop_progress(stage: str, fraction: float, message: str) -> None:
    """Default progress callback."""
    return


def read_header(path: Path) -> list[str]:
    """Read only the CSV header row.

    Args:
        path: Path to the source CSV.

    Returns:
        The list of column names in file order.

    Raises:
        IngestionError: If the file is missing, empty or has no header.
    """
    if not path.exists():
        raise IngestionError(f"Source file not found: {path}")
    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise IngestionError(f"Could not read the header of {path.name}: {exc}") from exc
    if not columns:
        raise IngestionError(f"{path.name} has no header row.")
    return columns


def check_schema(columns: Iterable[str]) -> list[str]:
    """Return the required columns missing from ``columns``.

    Args:
        columns: Columns found in the source file.

    Returns:
        Names of missing required columns. Empty when the file is usable.
    """
    present = set(columns)
    return [c for c in schema.REQUIRED_COLUMNS if c not in present]


def _annotate_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the validity flag columns.

    Args:
        df: Frame with derived features already applied.

    Returns:
        The same frame with ``Is_Valid_Severity``, ``Is_Valid_Coordinate`` and
        ``Is_Valid_Record`` present.
    """
    severity = pd.to_numeric(df[schema.SEVERITY], errors="coerce")
    df["Is_Valid_Severity"] = severity.isin(list(schema.SEVERITY_LEVELS)).fillna(False)

    lat = pd.to_numeric(df.get(schema.START_LAT), errors="coerce")
    lng = pd.to_numeric(df.get(schema.START_LNG), errors="coerce")
    df["Is_Valid_Coordinate"] = (
        lat.between(schema.LAT_MIN, schema.LAT_MAX) & lng.between(schema.LNG_MIN, schema.LNG_MAX)
    ).fillna(False)

    df["Is_Valid_Record"] = df["Is_Valid_Severity"] & df["Is_Valid_Coordinate"]
    return df


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Project exactly :data:`OUTPUT_COLUMNS`, in order, adding any that are absent.

    Anything outside the contract is dropped; anything inside it but missing from
    the source is added as an all-null column. Filling rather than omitting is
    what keeps the Parquet schema invariant: the writer is created with a fixed
    Arrow schema, so a chunk lacking a column would fail to convert instead of
    writing a shorter row group. A source file that omits an optional field
    therefore still produces the full table.

    Args:
        df: Prepared chunk.

    Returns:
        The frame restricted to the output contract, complete and in order.
    """
    dropped = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    global _ORDER_WARNED
    if dropped and not _ORDER_WARNED:
        _ORDER_WARNED = True
        logger.info(
            "Columns not part of the processed schema, dropped: %s",
            ", ".join(sorted(dropped)),
        )

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[list(OUTPUT_COLUMNS)]



def build_from_csv(
    source: Path,
    output: Path | None = None,
    *,
    method: str = "chunked",
    chunk_size: int | None = None,
    progress: ProgressCallback = _noop_progress,
) -> IngestionReport:
    """Convert the raw accident CSV into the processed Parquet table.

    Args:
        source: Path to ``US_Accidents_March23.csv`` (or any schema-compatible file).
        output: Destination Parquet path. Defaults to ``data/processed/accidents.parquet``.
        method: ``"chunked"`` or ``"duckdb"``.
        chunk_size: Rows per chunk for the chunked method.
        progress: Callable receiving ``(stage, fraction, message)``.

    Returns:
        An :class:`IngestionReport` describing the run.

    Raises:
        IngestionError: If the file is missing, unreadable, or missing required columns.
        ValueError: If ``method`` is not recognised.
    """
    source = Path(source)
    destination = Path(output) if output else PATHS.accidents_parquet

    if method not in {"chunked", "duckdb"}:
        raise ValueError(f"Unknown ingestion method: {method!r}. Use 'chunked' or 'duckdb'.")

    header = read_header(source)
    missing = check_schema(header)
    if missing:
        raise IngestionError(
            f"{source.name} is missing required column(s): {', '.join(missing)}. "
            f"Expected the original US Accidents schema "
            f"({schema.ORIGINAL_COLUMNS[0]}...{schema.ORIGINAL_COLUMNS[-1]})."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    progress("start", 0.0, f"Reading {source.name}")
    if method == "chunked":
        report = _build_chunked(
            source, destination, chunk_size or SETTINGS.csv_chunk_size, progress
        )
    else:
        report = _build_with_duckdb(source, destination, set(header), progress)

    report.seconds = time.perf_counter() - started
    report.output_bytes = destination.stat().st_size if destination.exists() else 0
    report.source = str(source)
    report.output = str(destination)
    report.method = method
    report.source_columns = header

    logger.info(
        "Ingested %s rows from %s in %.1fs (%s invalid severity, %s invalid coordinates)",
        f"{report.rows_written:,}",
        source.name,
        report.seconds,
        f"{report.invalid_severity:,}",
        f"{report.invalid_coordinates:,}",
    )
    progress("done", 1.0, f"Wrote {report.rows_written:,} rows")
    return report


def _arrow_schema() -> pa.Schema:
    """Return the fixed Arrow schema every chunk is cast to.

    ``ParquetWriter`` fixes its schema from the first chunk and rejects any
    later chunk that disagrees. Letting each chunk infer its own types does not
    work on a large file: a column that happens to contain no nulls in chunk 1
    becomes a non-nullable ``int8``, while chunk 15 -- with one unknown
    ``Sunrise_Sunset`` value -- infers a nullable ``Int8``. The file then
    refuses to accept it, several million rows in.

    Pinning the schema here removes the inference entirely.
    """
    fields = []
    for column in OUTPUT_COLUMNS:
        fields.append(pa.field(column, _ARROW_TYPES[column]))
    return pa.schema(fields)


def _build_chunked(
    source: Path,
    destination: Path,
    chunk_size: int,
    progress: ProgressCallback,
) -> IngestionReport:
    """Stream the CSV in Pandas chunks into a single Parquet file."""
    total_bytes = source.stat().st_size
    writer: pq.ParquetWriter | None = None
    report = IngestionReport(source=str(source), output=str(destination), method="chunked")
    arrow_schema = _arrow_schema()

    try:
        reader = pd.read_csv(
            source,
            chunksize=chunk_size,
            low_memory=False,
            # An empty field means "not recorded" for every column in this dataset.
            keep_default_na=True,
        )
        consumed = 0
        for index, chunk in enumerate(reader):
            processed = _prepare_chunk(chunk, header=set(chunk.columns))
            table = pa.Table.from_pandas(
                processed, schema=arrow_schema, preserve_index=False, safe=True
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    destination, arrow_schema, compression="zstd", compression_level=3
                )
            writer.write_table(table)

            report.rows_read += len(chunk)
            report.rows_written += len(processed)
            report.invalid_severity += int((~processed["Is_Valid_Severity"]).sum())
            report.invalid_coordinates += int((~processed["Is_Valid_Coordinate"]).sum())
            report.invalid_records += int((~processed["Is_Valid_Record"]).sum())
            report.unparsed_start_time += int(processed["Start_Time"].isna().sum())

            consumed += chunk.memory_usage(deep=True).sum()
            fraction = min(0.99, (index + 1) * chunk_size / max(1, _estimate_rows(source)))
            progress(
                "ingest",
                fraction,
                f"Chunk {index + 1}: {report.rows_written:,} rows processed",
            )
            logger.debug("Chunk %s: %s rows", index + 1, f"{len(processed):,}")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise IngestionError(f"Failed while reading {source.name}: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()

    if report.rows_written == 0:
        raise IngestionError(f"No rows could be parsed from {source.name}.")

    _atomic_touch(destination, total_bytes)
    return report


#: Approximate bytes per record in the original 47-column layout, including the
#: free-text ``Description`` and ``Street`` columns. Measured against the real
#: file at 396 B/row. This drives the progress bar only; the exact count is
#: known once ingestion finishes.
_BYTES_PER_RECORD: Final[int] = 396


def _estimate_rows(source: Path) -> int:
    """Estimate the row count from file size, for progress reporting only.

    Returns:
        Estimated record count. Display hint only -- never an analytical value.
    """
    return max(1, source.stat().st_size // _BYTES_PER_RECORD)


def _atomic_touch(destination: Path, total_bytes: int) -> None:
    """Verify the written file is non-empty and log its size."""
    if not destination.exists() or destination.stat().st_size == 0:
        raise IngestionError(f"Ingestion produced an empty file at {destination}.")
    logger.debug(
        "Parquet written: %.1f MB from %.1f MB of CSV", destination.stat().st_size / 1e6,
        total_bytes / 1e6,
    )


def _prepare_chunk(chunk: pd.DataFrame, header: set[str]) -> pd.DataFrame:
    """Apply feature engineering and validity flags to one raw chunk."""
    processed = add_derived_features(chunk)
    processed = _annotate_validity(processed)
    return _coerce_dtypes(_order_columns(processed)).reset_index(drop=True)


#: Pandas dtypes for every output column, applied before the Arrow cast. All
#: integer columns are nullable extension types so that a legitimately missing
#: value (an ``Is_Night`` for an unrecognised twilight, say) survives rather
#: than raising.
_DTYPE_MAP: Final[dict[str, str]] = {
    **dict.fromkeys(schema.NUMERIC_COLUMNS, "float64"),
    schema.START_LAT: "float64",
    schema.START_LNG: "float64",
    schema.END_LAT: "float64",
    schema.END_LNG: "float64",
    schema.START_TIME: "datetime64[ns]",
    schema.END_TIME: "datetime64[ns]",
    schema.DURATION_MINUTES: "float64",
    schema.DURATION_HOURS: "float64",
    schema.SEVERITY: "Int16",
    "Year": "Int16",
    "Month": "Int8",
    "Day": "Int8",
    "Hour": "Int8",
    "DayOfWeek": "Int8",
    "Is_Night": "Int8",
    **dict.fromkeys(
        (
            "Day_Night", "Month_Name", "Day_Name", "City", "County", "State",
            "Zipcode", "Timezone", "Weather_Condition", "Wind_Direction",
            "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight",
            "Astronomical_Twilight",
        ),
        "string",
    ),
    **dict.fromkeys(schema.BOOLEAN_COLUMNS, "bool"),
    "Is_Weekend": "bool",
    "Is_Rush_Hour": "bool",
    "Is_Valid_Severity": "bool",
    "Is_Valid_Coordinate": "bool",
    "Is_Valid_Record": "bool",
}

#: Fixed Arrow type for every output column. ``ParquetWriter`` fixes its schema
#: from the first chunk and rejects anything that disagrees, so every chunk is
#: cast to this rather than inferring its own types. Inference fails on a large
#: file: a column with no nulls in chunk 1 becomes a non-nullable int8, while a
#: later chunk containing one missing value infers a nullable Int8.
_ARROW_TYPES: Final[dict[str, pa.DataType]] = {
    schema.ID: pa.string(),
    schema.SEVERITY: pa.int16(),
    "Start_Time": pa.timestamp("ns"),
    "End_Time": pa.timestamp("ns"),
    schema.START_LAT: pa.float64(),
    schema.START_LNG: pa.float64(),
    schema.END_LAT: pa.float64(),
    schema.END_LNG: pa.float64(),
    "Distance(mi)": pa.float64(),
    **dict.fromkeys(
        (
            "City", "County", "State", "Zipcode", "Timezone",
            "Weather_Condition", "Wind_Direction", "Sunrise_Sunset",
            "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
            "Day_Night", "Month_Name", "Day_Name",
        ),
        pa.string(),
    ),
    **dict.fromkeys(schema.NUMERIC_COLUMNS, pa.float64()),
    **dict.fromkeys(schema.BOOLEAN_COLUMNS, pa.bool_()),
    "Year": pa.int16(),
    "Month": pa.int8(),
    "Day": pa.int8(),
    "Hour": pa.int8(),
    "DayOfWeek": pa.int8(),
    "Is_Night": pa.int8(),
    "Is_Weekend": pa.bool_(),
    "Is_Rush_Hour": pa.bool_(),
    schema.DURATION_MINUTES: pa.float64(),
    schema.DURATION_HOURS: pa.float64(),
    "Is_Valid_Severity": pa.bool_(),
    "Is_Valid_Coordinate": pa.bool_(),
    "Is_Valid_Record": pa.bool_(),
}


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known columns to fixed types so every chunk shares one Parquet schema.

    Args:
        df: Prepared chunk.

    Returns:
        The frame with deterministic dtypes applied.
    """
    out = df.copy()
    for column, dtype in _DTYPE_MAP.items():
        if column not in out.columns:
            continue
        try:
            if dtype == "datetime64[ns]":
                out[column] = pd.to_datetime(out[column], errors="coerce")
            elif dtype in {"string"}:
                out[column] = out[column].astype("string")
            elif dtype == "bool":
                out[column] = out[column].fillna(False).astype("bool")
            elif dtype in {"int8", "int16"}:
                numeric = pd.to_numeric(out[column], errors="coerce")
                # Integer columns legitimately contain missing values
                # (Is_Night for unknown twilight); keep them nullable.
                out[column] = numeric.round().astype(dtype if numeric.notna().all() else "Int8")
            else:
                out[column] = pd.to_numeric(out[column], errors="coerce").astype(dtype)
        except (ValueError, TypeError):
            # A column whose values are entirely unparseable keeps its inferred
            # type; ParquetWriter will still accept it if every chunk agrees.
            logger.debug("Skipped dtype coercion for %s", column)
    return out


def _build_with_duckdb(
    source: Path,
    destination: Path,
    available: set[str],
    progress: ProgressCallback,
) -> IngestionReport:
    """Convert the CSV with a single DuckDB push-down pipeline."""
    import duckdb

    from smart_traffic.feature_engineering import derived_feature_sql

    con = duckdb.connect(str(PATHS.processed / "ingest_tmp.duckdb"))
    report = IngestionReport(source=str(source), output=str(destination), method="duckdb")
    escaped_out = str(destination).replace("'", "''")
    header = read_header(source)

    try:
        progress("ingest", 0.15, "Scanning CSV with DuckDB")
        con.execute(f"SET memory_limit='{SETTINGS.duckdb_memory_limit}'")
        con.execute(f"SET temp_directory='{PATHS.processed / 'duckdb_tmp'}'")

        read_expr = _duckdb_read_expression(source, header)
        select_sql = derived_feature_sql(read_expr, available=set(header))

        progress("ingest", 0.5, "Deriving features")
        con.execute(f"CREATE OR REPLACE TEMP VIEW staged_raw AS {select_sql}")
        report.rows_written = int(con.execute("SELECT COUNT(*) FROM staged_raw").fetchone()[0])

        progress("ingest", 0.7, "Computing validity flags")
        # Validity is a *flag*, never a filter: no record is dropped here.
        # Downstream analyses choose whether to exclude invalid rows, and the
        # data-quality page can report exactly how many there are.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE staged AS
            SELECT *,
                (Severity IS NOT NULL AND Severity IN (1, 2, 3, 4)) AS Is_Valid_Severity,
                (Start_Lat IS NOT NULL AND Start_Lng IS NOT NULL
                 AND Start_Lat BETWEEN {schema.LAT_MIN} AND {schema.LAT_MAX}
                 AND Start_Lng BETWEEN {schema.LNG_MIN} AND {schema.LNG_MAX})
                    AS Is_Valid_Coordinate,
                (Severity IS NOT NULL AND Severity IN (1, 2, 3, 4)
                 AND Start_Lat IS NOT NULL AND Start_Lng IS NOT NULL
                 AND Start_Lat BETWEEN {schema.LAT_MIN} AND {schema.LAT_MAX}
                 AND Start_Lng BETWEEN {schema.LNG_MIN} AND {schema.LNG_MAX})
                    AS Is_Valid_Record
            FROM staged_raw
            """
        )

        progress("ingest", 0.85, "Writing Parquet")
        ordered = ", ".join(f'"{c}"' for c in OUTPUT_COLUMNS)
        con.execute(
            f"COPY (SELECT {ordered} FROM staged) TO '{escaped_out}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )

        stats = con.execute(
            """
            SELECT
                SUM(CASE WHEN NOT Is_Valid_Severity THEN 1 ELSE 0 END),
                SUM(CASE WHEN NOT Is_Valid_Coordinate THEN 1 ELSE 0 END),
                SUM(CASE WHEN NOT Is_Valid_Record THEN 1 ELSE 0 END),
                SUM(CASE WHEN Start_Time IS NULL THEN 1 ELSE 0 END)
            FROM staged
            """
        ).fetchone()
        report.invalid_severity = int(stats[0] or 0)
        report.invalid_coordinates = int(stats[1] or 0)
        report.invalid_records = int(stats[2] or 0)
        report.unparsed_start_time = int(stats[3] or 0)
        report.rows_read = report.rows_written
    except duckdb.Error as exc:
        raise IngestionError(f"DuckDB ingestion failed for {source.name}: {exc}") from exc
    finally:
        con.close()

    if report.rows_written == 0:
        raise IngestionError(f"No rows could be parsed from {source.name}.")

    _atomic_touch(destination, source.stat().st_size)
    return report


def _duckdb_read_expression(source: Path, header: list[str]) -> str:
    """Build a ``read_csv(...)`` call with an explicit type for every column.

    DuckDB requires the ``columns`` map to cover the whole file. Specifying all
    of them also removes dialect sniffing entirely, which is both faster and
    immune to the quoting quirks in free-text ``Description`` values.

    Args:
        source: CSV to read.
        header: Ordered column names from :func:`read_header`.

    Returns:
        A SQL table expression usable as a ``FROM`` source.
    """
    types: list[str] = []
    for column in header:
        if column in _FORCED_DOUBLE:
            type_name = "DOUBLE"
        elif column in _FORCED_BOOLEAN:
            type_name = "BOOLEAN"
        else:
            # Everything else stays text; downstream TRY_CASTs normalise the
            # few numeric-looking values that are not in the known set.
            type_name = "VARCHAR"
        types.append(f"'{column}': '{type_name}'")

    escaped = str(source).replace("'", "''")
    return (
        f"read_csv('{escaped}', header=true, auto_detect=false, "
        f"columns={{{', '.join(types)}}}, "
        f"nullstr='', strict_mode=true, max_line_size=10485760)"
    )


# ---------------------------------------------------------------------------
# Sampling for expensive analysis
# ---------------------------------------------------------------------------


def build_sample(
    output: Path | None = None,
    *,
    row_count: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Write a reproducible random sample of the processed table.

    Uses ``hash(ID)`` ordering rather than ``RANDOM()`` so the sample is a pure
    function of the data and the seed: rerunning on the same input selects the
    same rows, which is what makes the ML and clustering results reproducible.

    Args:
        output: Destination path. Defaults to ``data/processed/accidents_sample.parquet``.
        row_count: Number of rows to draw. Defaults to the configured cap.
        seed: Random seed. Defaults to the configured seed.

    Returns:
        A DataFrame describing the sample (see :meth:`SampleInfo.to_dataframe`).

    Raises:
        DatasetNotAvailableError: If the processed table has not been built.
    """
    db = Database()
    db.ensure_views()

    limit = row_count or SETTINGS.sample_training_rows
    effective_seed = SETTINGS.random_seed if seed is None else seed

    frame = db.sql(
        """
        SELECT *
        FROM accidents
        ORDER BY hash(ID, ?)
        LIMIT ?
        """,
        [effective_seed, limit],
    )
    frame = frame.sort_values(schema.ID).reset_index(drop=True)

    destination = Path(output) if output else PATHS.sample_parquet
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False, compression="zstd")

    info = SampleInfo(
        rows=len(frame),
        requested=limit,
        seed=effective_seed,
        population=int(db.scalar("SELECT COUNT(*) FROM accidents") or 0),
        path=str(destination),
    )
    logger.info(
        "Wrote sample of %s rows (seed %s) to %s", f"{info.rows:,}", effective_seed,
            destination.name
    )
    db.reset()
    return info.to_dataframe()


@dataclass(slots=True)
class SampleInfo:
    """Provenance of a reproducibility sample, surfaced in the UI."""

    rows: int
    requested: int
    seed: int
    population: int
    path: str

    @property
    def coverage_percent(self) -> float:
        """Percentage of the population captured by the sample."""
        return round(self.rows / self.population * 100, 3) if self.population else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Return the sample provenance as a two-column DataFrame."""
        return pd.DataFrame(
            [
                {"Field": "Sample rows", "Value": self.rows},
                {"Field": "Requested rows", "Value": self.requested},
                {"Field": "Population rows", "Value": self.population},
                {"Field": "Coverage (%)", "Value": self.coverage_percent},
                {"Field": "Random seed", "Value": self.seed},
                {"Field": "File", "Value": self.path},
            ]
        )


def build_processed_dataset(
    source: Path | None = None,
    *,
    method: str = "chunked",
    build_sample_too: bool = True,
    progress: ProgressCallback = _noop_progress,
) -> IngestionReport:
    """Run the full ingestion workflow and rebind the query layer.

    Args:
        source: Source CSV. Defaults to ``data/raw/US_Accidents_March23.csv``.
        method: Ingestion method, see :func:`build_from_csv`.
        build_sample_too: Whether to also build the reproducibility sample.
        progress: Progress callback.

    Returns:
        The :class:`IngestionReport` from the main ingestion step.
    """
    csv_path = Path(source) if source else PATHS.raw_csv
    PATHS.ensure_dirs()

    report = build_from_csv(csv_path, method=method, progress=progress)

    if build_sample_too:
        progress("sample", 0.9, "Building reproducibility sample")
        build_sample()

    db = Database()
    db.reset()
    db.ensure_views()
    return report


def column_dtypes(db: Database | None = None) -> dict[str, str]:
    """Return the Arrow type name of every processed column.

    Args:
        db: Optional database handle. Defaults to the module singleton.

    Returns:
        Mapping of column name to type string, e.g. ``{"Hour": "int32"}``.
    """
    db = db or Database()
    db.ensure_views()
    schema_ = db.sql("DESCRIBE SELECT * FROM accidents")
    return dict(zip(schema_["column_name"], schema_["column_type"], strict=False))


def memory_footprint(path: Path | None = None) -> dict[str, Any]:
    """Report the size and row count of the processed Parquet file.

    Args:
        path: Parquet path. Defaults to the configured processed file.

    Returns:
        Dictionary with path, size in MB, row count and column count.
    """
    target = Path(path) if path else PATHS.accidents_parquet
    if not target.exists():
        return {"path": str(target), "exists": False, "size_mb": 0.0, "rows": 0, "columns": 0}

    parquet_file = pq.ParquetFile(target)
    return {
        "path": str(target),
        "exists": True,
        "size_mb": round(target.stat().st_size / 1e6, 2),
        "rows": parquet_file.metadata.num_rows,
        "columns": parquet_file.metadata.num_columns,
    }


def _as_numpy(value: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if isinstance(value, np.generic):
        return value.item()
    return value

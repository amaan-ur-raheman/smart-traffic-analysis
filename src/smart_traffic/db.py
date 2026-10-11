"""DuckDB-backed query layer.

Every aggregate in the application is expressed as SQL executed against the
processed Parquet table, so a chart costs one scan rather than a full pandas
materialisation. The connection is a per-process singleton because DuckDB's
Python connection is not thread-safe to share across Streamlit's reruns without
serialisation.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from smart_traffic.config import PATHS, SETTINGS
from smart_traffic.logging_config import get_logger

logger = get_logger("db")

#: Name under which the accidents table is registered inside DuckDB.
ACCIDENTS_VIEW = "accidents"
#: Name under which the training/sampling table is registered inside DuckDB.
SAMPLE_VIEW = "accidents_sample"


class DatasetNotAvailableError(RuntimeError):
    """Raised when no processed dataset has been built yet."""


class Database:
    """Thin, thread-safe wrapper around a DuckDB connection.

    The instance is lazily created and reused. Call :meth:`reset` after building
    new Parquet files so the views are rebound.
    """

    def __init__(self, db_path: Path | None = None, parquet_path: Path | None = None) -> None:
        """Bind to the given paths, or to the configured defaults.

        Args:
            db_path: DuckDB catalogue path, or ``None`` for the default.
            parquet_path: Processed Parquet path, or ``None`` for the default.

        The defaults are resolved lazily on each access rather than captured
        here, so a ``STA_DATA_DIR`` override takes effect even though this
        singleton is constructed at import time.
        """
        self._db_path = db_path
        self._parquet_path = parquet_path
        self._lock = threading.RLock()
        self._con: duckdb.DuckDBPyConnection | None = None
        self._registered = False

    @property
    def db_path(self) -> Path:
        """Path to the DuckDB catalogue file."""
        return self._db_path or PATHS.duckdb_file

    @property
    def parquet_path(self) -> Path:
        """Path to the processed accidents Parquet file."""
        return self._parquet_path or PATHS.accidents_parquet

    # -- connection lifecycle ---------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.db_path))
        con.execute(f"SET memory_limit='{SETTINGS.duckdb_memory_limit}'")
        # Let DuckDB spill to disk rather than fail when a sort exceeds memory.
        con.execute("SET temp_directory=? ", [str(self.db_path.parent / "duckdb_tmp")])
        return con

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Return the shared connection, opening it on first use."""
        with self._lock:
            if self._con is None:
                self._con = self._connect()
            return self._con

    def reset(self) -> None:
        """Close and drop the connection so Parquet files are re-registered."""
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None
            self._registered = False

    def close(self) -> None:
        """Close the connection if one is open."""
        self.reset()

    # -- availability ------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Whether the processed accidents Parquet file exists and is readable."""
        return self.parquet_path.exists() and self.parquet_path.stat().st_size > 0

    def require(self) -> None:
        """Raise :class:`DatasetNotAvailableError` when no dataset is present."""
        if not self.is_available:
            raise DatasetNotAvailableError(
                "No processed dataset found. Place the CSV at "
                f"{PATHS.raw_csv} and run: uv run sta build"
            )

    # -- view registration -------------------------------------------------

    def ensure_views(self) -> None:
        """Bind the processed Parquet files to DuckDB view names.

        Safe to call repeatedly; a no-op once the views are bound.
        """
        with self._lock:
            if self._registered:
                return
            self.require()
            escaped = str(self.parquet_path).replace("'", "''")
            try:
                self.connection.execute(
                    f"CREATE OR REPLACE VIEW {ACCIDENTS_VIEW} AS "
                    f"SELECT * FROM read_parquet('{escaped}')"
                )
                # DuckDB binds a view lazily: creating it over a truncated file
                # succeeds, and the failure only appears on the first real
                # query. Reading a single row forces the Parquet footer to be
                # parsed here, so corruption is reported as a dataset problem
                # instead of surfacing later as an opaque query error.
                self.connection.execute(
                    f"SELECT * FROM {ACCIDENTS_VIEW} LIMIT 1"
                ).fetchall()
            except duckdb.Error as exc:
                self._registered = False
                logger.error("Could not read %s: %s", self.parquet_path, exc)
                raise DatasetNotAvailableError(
                    f"The processed dataset at {self.parquet_path} could not be "
                    "read. It is most likely truncated or corrupt, which happens "
                    "if a build was interrupted. Rebuild it with: uv run sta build"
                ) from exc
            sample = PATHS.sample_parquet
            if sample.exists() and sample.stat().st_size > 0:
                sample_escaped = str(sample).replace("'", "''")
                self.connection.execute(
                    f"CREATE OR REPLACE VIEW {SAMPLE_VIEW} AS "
                    f"SELECT * FROM read_parquet('{sample_escaped}')"
                )
            else:
                self.connection.execute(
                    f"CREATE OR REPLACE VIEW {SAMPLE_VIEW} AS "
                    f"SELECT * FROM {ACCIDENTS_VIEW} LIMIT 0"
                )
            self._registered = True
            logger.debug("Registered DuckDB views against %s", self.parquet_path)

    # -- query helpers -----------------------------------------------------

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Run a query and return the result as a DataFrame.

        Args:
            query: SQL text. Must be a ``SELECT`` (or a statement returning rows).
            params: Optional positional parameters for ``?`` placeholders.

        Returns:
            Query results as a pandas DataFrame.

        Raises:
            DatasetNotAvailableError: If no processed dataset exists.
        """
        self.ensure_views()
        with self._lock:
            if params:
                return self.connection.execute(query, params).df()
            return self.connection.execute(query).df()

    def scalar(self, query: str, params: list[Any] | None = None) -> Any:
        """Run a query and return the first column of the first row.

        Args:
            query: SQL text.
            params: Optional positional parameters.

        Returns:
            The scalar value, or ``None`` when the query returns no rows.
        """
        self.ensure_views()
        with self._lock:
            if params:
                row = self.connection.execute(query, params).fetchone()
            else:
                row = self.connection.execute(query).fetchone()
        return row[0] if row else None

    def columns(self) -> list[str]:
        """Return the column names available in the accidents view."""
        self.ensure_views()
        with self._lock:
            rows = self.connection.execute(
                f"DESCRIBE SELECT * FROM {ACCIDENTS_VIEW}"
            ).fetchall()
        return [r[0] for r in rows]

    def has_column(self, name: str) -> bool:
        """Whether ``name`` exists in the processed accidents table."""
        try:
            return name in set(self.columns())
        except DatasetNotAvailableError:
            return False


#: Module-level singleton used by the UI and CLI.
DB = Database()

"""Application configuration, paths and tunable analysis parameters.

All filesystem locations are resolved relative to the project root so the app
behaves identically regardless of the working directory it is launched from.
Every path can be overridden with an environment variable, which keeps the app
free of hardcoded machine-specific values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the project root (the directory containing ``pyproject.toml``)."""
    override = os.environ.get("STA_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    # src/smart_traffic/config.py -> src/smart_traffic -> src -> <root>
    return Path(__file__).resolve().parents[2]


def _env_path(key: str, default: Path) -> Path:
    value = os.environ.get(key)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class AnalysisSettings:
    """Tunable parameters that define analysis behaviour.

    These are surfaced in the UI where a user might reasonably want to change
    them, and documented in the README so results are reproducible.
    """

    # -- Rush hour ---------------------------------------------------------
    # Default definition: 07:00-09:59 and 16:00-18:59 local clock hour, i.e.
    # hours 7, 8, 9 and 16, 17, 18. Both stated endpoints are inclusive, which
    # is the literal reading of "7 AM to 9 AM". The windows are configurable;
    # `feature_engineering.is_rush_hour_mask` and the SQL projection both read
    # them from here, so the two ingestion strategies cannot disagree.
    rush_hour_morning: tuple[int, int] = (7, 9)
    rush_hour_evening: tuple[int, int] = (16, 18)

    # -- Night -------------------------------------------------------------
    # An accident is "night" unless the dataset's own Sunrise_Sunset field
    # says "Day". Ambiguous values become "Unknown" rather than being guessed.
    night_label: str = "Night"
    day_label: str = "Day"
    unknown_label: str = "Unknown"

    # -- Reproducibility ---------------------------------------------------
    random_seed: int = 42

    # -- Sampling budgets --------------------------------------------------
    # Expensive operations (PCA scatter, clustering, training) never touch the
    # full dataset by default. These caps bound memory and runtime.
    sample_scatter_points: int = 40_000
    sample_clustering_points: int = 25_000
    sample_hierarchical_points: int = 3_000
    sample_training_rows: int = 400_000
    sample_permutation_importance: int = 20_000

    # -- Clustering search space ------------------------------------------
    kmeans_k_min: int = 2
    kmeans_k_max: int = 8
    hierarchical_k: int = 6

    # -- Bounded work ------------------------------------------------------
    # DuckDB is told how much memory it may use for large aggregations.
    duckdb_memory_limit: str = "4GB"
    csv_chunk_size: int = 250_000
    top_n: int = 10

    # -- Numeric features used for unsupervised analysis -------------------
    numeric_analysis_features: tuple[str, ...] = field(
        default=(
            "Temperature(F)",
            "Wind_Chill(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
            "Hour",
            "Is_Weekend",
            "Is_Rush_Hour",
            "Is_Night",
        )
    )


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem locations for data, models, artifacts and logs."""

    root: Path

    @property
    def data(self) -> Path:
        """Root of the data tree. Override with ``STA_DATA_DIR``."""
        return _env_path("STA_DATA_DIR", self.root / "data")

    @property
    def raw(self) -> Path:
        """Where the source CSV is placed."""
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        """Parquet tables and the DuckDB catalogue."""
        return self.data / "processed"

    @property
    def analytics(self) -> Path:
        """Exported reports and bundles."""
        return self.data / "analytics"

    @property
    def synthetic(self) -> Path:
        """Generated development datasets."""
        return self.data / "synthetic"

    @property
    def models(self) -> Path:
        """Persisted models. Override with ``STA_MODELS_DIR``."""
        return _env_path("STA_MODELS_DIR", self.root / "models")

    @property
    def artifacts(self) -> Path:
        """Intermediate artefacts. Override with ``STA_ARTIFACTS_DIR``."""
        return _env_path("STA_ARTIFACTS_DIR", self.root / "artifacts")

    @property
    def logs(self) -> Path:
        """Rotating log files. Override with ``STA_LOG_DIR``."""
        return _env_path("STA_LOG_DIR", self.root / "logs")

    @property
    def duckdb_file(self) -> Path:
        """On-disk DuckDB catalogue used by the app's query layer."""
        return self.processed / "traffic.duckdb"

    @property
    def accidents_parquet(self) -> Path:
        """Processed accident table, one row per valid record."""
        return self.processed / "accidents.parquet"

    @property
    def sample_parquet(self) -> Path:
        """Reproducible sample used for ML and clustering."""
        return self.processed / "accidents_sample.parquet"

    @property
    def raw_csv(self) -> Path:
        """Default location for the original Kaggle CSV."""
        return self.raw / "US_Accidents_March23.csv"

    @property
    def source_registry(self) -> Path:
        """JSON manifest describing the currently loaded dataset."""
        return self.processed / "source_registry.json"

    def ensure_dirs(self) -> None:
        """Create every managed directory. Safe to call repeatedly."""
        for directory in (
            self.data,
            self.raw,
            self.processed,
            self.analytics,
            self.synthetic,
            self.models,
            self.artifacts,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


SETTINGS = AnalysisSettings()
PATHS = Paths(root=_project_root())

#: Dataset label shown throughout the UI, sourced from the Kaggle page.
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents"
DATASET_DISPLAY_NAME = "US Accidents (2016-2023)"

#: Explanation of what ``Severity`` actually encodes in this dataset.
SEVERITY_MEANING = (
    "In this dataset `Severity` is the reported traffic impact / disruption level on a "
    "1-4 scale. It is not a measurement of injuries, fatalities, or human harm."
)

#: Shown wherever a rate or count could be misread as per-driver risk.
EXPOSURE_DISCLAIMER = (
    "These are counts of *recorded accidents*, not per-driver risk. The dataset "
    "contains no traffic-volume or exposure data, so frequency cannot be converted "
    "into a rate of accident per unit of driving."
)

ASSOCIATION_DISCLAIMER = (
    "All relationships shown here are observational associations in recorded data. "
    "They do not establish causation."
)

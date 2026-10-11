"""Exportable reports.

Every artefact carries provenance -- the dataset it came from, its row count,
its coverage period and the filter that was active -- because a CSV extracted
from a dashboard is meaningless without it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from smart_traffic.aggregation import FilterSpec
from smart_traffic.analytics import DatasetProfile, Finding
from smart_traffic.config import PATHS
from smart_traffic.logging_config import get_logger

logger = get_logger("reporting")


@dataclass(slots=True)
class ExportBundle:
    """A named collection of tables ready for download.

    Attributes:
        name: Human-readable bundle name.
        tables: Mapping of filename stem to DataFrame.
        provenance: Metadata recorded alongside every table.
    """

    name: str
    tables: dict[str, pd.DataFrame]
    provenance: dict[str, Any]

    def filenames(self) -> dict[str, str]:
        """Return a mapping of table label to a safe download filename."""
        stamp = self._stamp()
        return {
            label: f"{_slug(self.name)}_{_slug(label)}_{stamp}.csv"
            for label in self.tables
        }

    def as_zip(self, directory: Path | None = None) -> Path:
        """Write every table plus a provenance file into a single archive.

        Args:
            directory: Destination directory. Defaults to ``data/analytics``.

        Returns:
            Path to the written ``.zip``.
        """
        import zipfile

        target = Path(directory) if directory else PATHS.analytics
        target.mkdir(parents=True, exist_ok=True)
        archive = target / f"{_slug(self.name)}_{self._stamp()}.zip"

        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for label, filename in self.filenames().items():
                bundle.writestr(filename, self.tables[label].to_csv(index=False))
            bundle.writestr(
                f"{_slug(self.name)}_provenance.json",
                json.dumps(self.provenance, indent=2, default=str),
            )
        logger.info("Wrote export bundle %s", archive.name)
        return archive

    def _stamp(self) -> str:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def provenance(
    profile: DatasetProfile | None = None,
    spec: FilterSpec | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance record attached to every export.

    Args:
        profile: Optional dataset profile.
        spec: Optional active filters.
        extra: Additional key/value pairs to merge in.

    Returns:
        A JSON-serialisable dictionary.
    """
    record: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "application": "Smart Traffic Analysis and Accident Severity Prediction",
    }
    if profile is not None:
        record.update(
            {
                "records_analysed": profile.total_records,
                "records_passing_validation": profile.valid_records,
                "unique_accident_ids": profile.unique_ids,
                "coverage_start": profile.coverage_start,
                "coverage_end": profile.coverage_end,
                "states": profile.distinct_states,
                "cities": profile.distinct_cities,
                "partial_final_year": profile.partial_year,
            }
        )
    if spec is not None:
        record["active_filters"] = spec.describe() or ["None - full dataset"]
    if extra:
        record.update(extra)
    return record


def build_analytics_bundle(
    tables: dict[str, pd.DataFrame],
    profile: DatasetProfile | None = None,
    spec: FilterSpec | None = None,
    *,
    name: str = "traffic_analytics",
) -> ExportBundle:
    """Assemble a bundle of analytical tables.

    Args:
        tables: Mapping of display label to DataFrame.
        profile: Dataset profile for provenance.
        spec: Active filters for provenance.
        name: Bundle name.

    Returns:
        The assembled :class:`ExportBundle`.
    """
    return ExportBundle(
        name=name,
        tables={label: frame for label, frame in tables.items() if frame is not None},
        provenance=provenance(profile, spec),
    )


def findings_frame(findings: list[Finding]) -> pd.DataFrame:
    """Convert findings into an exportable table.

    Args:
        findings: Computed findings.

    Returns:
        DataFrame with Category, Finding, Evidence and Caveat.
    """
    if not findings:
        return pd.DataFrame(columns=["Category", "Finding", "Evidence", "Caveat"])
    return pd.DataFrame([f.as_dict() for f in findings])


def validation_summary_frame(report: Any) -> pd.DataFrame:
    """Convert a validation report into an exportable table.

    Args:
        report: A :class:`smart_traffic.validation.ValidationReport`.

    Returns:
        The report as a DataFrame.
    """
    return report.to_dataframe()


def write_markdown_report(
    path: Path,
    *,
    title: str,
    sections: list[tuple[str, str]],
) -> Path:
    """Write a Markdown report.

    Args:
        path: Destination path.
        title: Report title.
        sections: Ordered ``(heading, body)`` pairs.

    Returns:
        The path written.
    """
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip(), ""])
    lines.extend(["---", "", f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC}_", ""])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def safe_download_name(name: str, *, max_length: int = 80) -> str:
    """Return a filesystem-safe download name.

    Strips path separators and traversal sequences, then truncates. Used on any
    user-influenced label before it becomes a filename.

    Args:
        name: Desired name.
        max_length: Maximum length after sanitisation.

    Returns:
        A safe filename fragment.
    """
    cleaned = re.sub(r"[^\w\s.-]", "", str(name)).strip()
    cleaned = cleaned.replace("..", "_").replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r"[\s]+", "_", cleaned)
    return (cleaned or "export")[:max_length]


def _slug(value: str) -> str:
    """Convert an arbitrary label to a lowercase slug."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

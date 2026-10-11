"""Application-level analytics: KPIs and computed findings.

Everything here derives from the dataset at call time. Nothing is hardcoded, so
a different dataset produces different findings rather than stale ones -- which
is the whole reason findings are generated rather than written down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from smart_traffic import schema
from smart_traffic.aggregation import Aggregator, FilterSpec
from smart_traffic.config import (
    ASSOCIATION_DISCLAIMER,
    EXPOSURE_DISCLAIMER,
    SEVERITY_MEANING,
)
from smart_traffic.db import DB
from smart_traffic.logging_config import get_logger

logger = get_logger("analytics")


@dataclass(slots=True)
class DatasetProfile:
    """Headline facts about the loaded dataset.

    Attributes:
        total_records: Rows in the processed table.
        valid_records: Rows passing every validity check.
        unique_ids: Distinct accident IDs.
        coverage_start: Earliest parsed timestamp.
        coverage_end: Latest parsed timestamp.
        distinct_states: Number of states represented.
        distinct_cities: Number of cities represented.
        severity_counts: Severity distribution.
        years_present: Years with at least one record.
        partial_year: The final year, which is incomplete in this dataset.
    """

    total_records: int
    valid_records: int
    unique_ids: int
    coverage_start: str
    coverage_end: str
    distinct_states: int
    distinct_cities: int
    severity_counts: pd.DataFrame
    years_present: list[int]
    partial_year: int | None = None

    def high_severity_count(self) -> int:
        """Number of severity 3 and 4 records."""
        if self.severity_counts.empty:
            return 0
        return int(
            self.severity_counts.loc[
                self.severity_counts["Severity"].isin([3, 4]), "Accidents"
            ].sum()
        )

    def high_severity_rate(self) -> float:
        """Percentage of records at severity 3 or 4."""
        total = self.total_records
        return round(self.high_severity_count() / total * 100, 4) if total else 0.0

    def most_common_severity(self) -> int | None:
        """The severity category with the most records."""
        if self.severity_counts.empty:
            return None
        return int(self.severity_counts.loc[self.severity_counts["Accidents"].idxmax(), "Severity"])


def build_profile(spec: FilterSpec | None = None) -> DatasetProfile:
    """Compute the dataset headline facts.

    Args:
        spec: Active filters. Headline facts reflect the filtered view.

    Returns:
        A populated :class:`DatasetProfile`.
    """
    spec = spec or FilterSpec()
    aggregator = Aggregator(DB)

    total = aggregator.total(spec)
    span = spec.select(
        "SELECT MIN(Start_Time) AS lo, MAX(Start_Time) AS hi FROM accidents", DB
    )
    coverage_start = "" if span.empty else str(span["lo"][0])
    coverage_end = "" if span.empty else str(span["hi"][0])

    coverage = spec.select(
        """
        SELECT COUNT(DISTINCT State)::BIGINT AS states,
               COUNT(DISTINCT City)::BIGINT AS cities,
               COUNT(DISTINCT Year)::BIGINT AS years
        FROM accidents
        """,
        DB,
    )
    valid = int(
        spec.select(
            "SELECT COUNT(*) AS n FROM accidents WHERE Is_Valid_Record = TRUE", DB
        )["n"][0]
    )

    years = _year_list(spec)
    partial = years[-1] if years else None

    return DatasetProfile(
        total_records=total,
        valid_records=valid,
        unique_ids=int(DB.scalar("SELECT COUNT(DISTINCT ID) FROM accidents") or 0),
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        distinct_states=int(coverage["states"][0]) if not coverage.empty else 0,
        distinct_cities=int(coverage["cities"][0]) if not coverage.empty else 0,
        severity_counts=aggregator.severity_counts(spec, DB),
        years_present=years,
        partial_year=partial,
    )


def _year_list(spec: FilterSpec) -> list[int]:
    """Return the sorted list of years present under the filter."""
    frame = spec.select("SELECT DISTINCT Year FROM accidents WHERE Year IS NOT NULL", DB)
    return sorted(int(y) for y in frame["Year"].tolist())


def kpi_cards(spec: FilterSpec | None = None) -> dict[str, Any]:
    """Return the KPI figures shown on the overview dashboard.

    Args:
        spec: Active filters.

    Returns:
        Ordered mapping of KPI label to a display-ready value, plus the raw
        integers under ``_raw`` for downstream use.
    """
    spec = spec or FilterSpec()
    aggregator = Aggregator(DB)
    severity = aggregator.severity_counts(spec, DB)
    total = int(severity["Accidents"].sum()) if not severity.empty else 0

    by_level = (
        dict(zip(severity["Severity"], severity["Accidents"], strict=False))
        if not severity.empty
        else {}
    )
    counts = {level: int(by_level.get(level, 0)) for level in schema.SEVERITY_LEVELS}
    high = counts[3] + counts[4]
    high_pct = round(high / total * 100, 2) if total else 0.0

    return {
        "Total accidents": f"{total:,}",
        "Severity 1": f"{counts[1]:,}",
        "Severity 2": f"{counts[2]:,}",
        "Severity 3": f"{counts[3]:,}",
        "Severity 4": f"{counts[4]:,}",
        "Severity 3 + 4": f"{high:,}",
        "Severity 3 + 4 (%)": f"{high_pct:.2f}%",
        "_raw": {
            "total": total,
            **{f"severity_{level}": counts[level] for level in schema.SEVERITY_LEVELS},
            "high_severity": high,
            "high_severity_pct": high_pct,
        },
    }


@dataclass(slots=True)
class Finding:
    """A single computed insight.

    Attributes:
        headline: The claim, stated concretely.
        evidence: The numbers it rests on.
        caveat: Why the claim should not be over-read.
        category: Grouping used for display.
    """

    headline: str
    evidence: str
    caveat: str = ""
    category: str = "General"

    def as_dict(self) -> dict[str, str]:
        """Flatten to a dictionary for export."""
        return {
            "Category": self.category,
            "Finding": self.headline,
            "Evidence": self.evidence,
            "Caveat": self.caveat,
        }


def generate_findings(
    spec: FilterSpec | None = None,
    *,
    model_summary: dict[str, Any] | None = None,
) -> list[Finding]:
    """Compute the key findings for the current data and filters.

    Each finding is derived from a live query. When a query returns nothing the
    corresponding finding is omitted rather than reported with placeholder text.

    Args:
        spec: Active filters.
        model_summary: Optional model metadata dict (see
            :meth:`smart_traffic.modeling.TrainedModel.to_metadata`) used to add
            a model-performance finding.

    Returns:
        A list of :class:`Finding` objects.
    """
    spec = spec or FilterSpec()
    aggregator = Aggregator(DB)
    findings: list[Finding] = []
    total = aggregator.total(spec)

    if total == 0:
        return findings

    # -- severity ----------------------------------------------------------
    severity = aggregator.severity_counts(spec, DB)
    if not severity.empty:
        dominant = severity.loc[severity["Accidents"].idxmax()]
        findings.append(
            Finding(
                headline=(
                    f"Severity {int(dominant['Severity'])} is the most frequently recorded "
                    f"category, at {int(dominant['Accidents']):,} of {total:,} records "
                    f"({dominant['Percentage']:.1f}%)."
                ),
                evidence=(
                    "Counts by category: "
                    + ", ".join(
                        f"Severity {int(r['Severity'])} = {int(r['Accidents']):,}"
                        for _, r in severity.iterrows()
                    )
                ),
                caveat=SEVERITY_MEANING,
                category="Severity",
            )
        )
        high = int(severity.loc[severity["Severity"].isin([3, 4]), "Accidents"].sum())
        findings.append(
            Finding(
                headline=(
                    f"{high:,} records ({high / total * 100:.1f}%) are recorded at severity 3 "
                    f"or 4."
                ),
                evidence=f"Denominator: all {total:,} records matching the current filters.",
                caveat=(
                    "A high severity category records a larger disruption to traffic flow. "
                    "It is not a count of injuries or deaths."
                ),
                category="Severity",
            )
        )

    # -- time --------------------------------------------------------------
    hours = aggregator.counts_by("Hour", spec, db=DB)
    if not hours.empty:
        peak = hours.loc[hours["Accidents"].idxmax()]
        share = peak["Accidents"] / total * 100
        findings.append(
            Finding(
                headline=(
                    f"The {int(peak['Hour']):02d}:00 hour records the most accidents: "
                    f"{int(peak['Accidents']):,} ({share:.1f}% of all records)."
                ),
                evidence=(
                    f"Busiest hour: {int(peak['Hour']):02d}:00-"
                    f"{int(peak['Hour']) + 1:02d}:59."
                ),
                caveat=EXPOSURE_DISCLAIMER,
                category="Time",
            )
        )

    days = aggregator.counts_by("Day_Name", spec, db=DB)
    if not days.empty:
        quietest = days.loc[days["Accidents"].idxmin()]
        busiest = days.loc[days["Accidents"].idxmax()]
        findings.append(
            Finding(
                headline=(
                    f"{busiest['Day_Name']} records the most accidents "
                    f"({int(busiest['Accidents']):,}); {quietest['Day_Name']} records the "
                    f"fewest ({int(quietest['Accidents']):,})."
                ),
                evidence=(
                    f"Spread of {int(busiest['Accidents']) - int(quietest['Accidents']):,} "
                    f"records between the busiest and quietest weekday."
                ),
                caveat=EXPOSURE_DISCLAIMER,
                category="Time",
            )
        )

    coverage = aggregator.coverage_by_year(spec, DB)
    if not coverage.empty and len(coverage) > 1:
        full_years = coverage[~coverage["Is_Partial"]]
        partial = coverage[coverage["Is_Partial"]]
        if not full_years.empty and not partial.empty:
            best_full = full_years.loc[full_years["Accidents"].idxmax()]
            year = int(best_full["Year"])
            partial_year = int(partial.iloc[0]["Year"])
            partial_span = str(partial.iloc[0]["Coverage"] or "part of the year")
            findings.append(
                Finding(
                    headline=(
                        f"{year} records the most accidents of any complete year, with "
                        f"{int(best_full['Accidents']):,} records."
                    ),
                    evidence=(
                        f"Excludes {partial_year}, which covers only {partial_span} and is "
                        f"therefore not comparable to a full year."
                    ),
                    caveat=(
                        "Reporting volume rose over the period, but the dataset does not "
                        "indicate whether that reflects more crashes, more reporting, or "
                        "both."
                    ),
                    category="Time",
                )
            )

    # -- geography ---------------------------------------------------------
    states = aggregator.counts_by("State", spec, limit=1, db=DB)
    if not states.empty:
        top_state = states.iloc[0]
        states_all = aggregator.counts_by("State", spec, db=DB)
        share = top_state["Accidents"] / total * 100
        findings.append(
            Finding(
                headline=(
                    f"{top_state['State']} has the most recorded accidents "
                    f"({int(top_state['Accidents']):,}, {share:.1f}% of all records)."
                ),
                evidence=(
                    f"{len(states_all)} states appear in the data. Top five: "
                    + ", ".join(
                        f"{r['State']} ({int(r['Accidents']):,})"
                        for _, r in states_all.head(5).iterrows()
                    )
                ),
                caveat=(
                    "This reflects both how many accidents occur and how many are reported "
                    "to the source. It is not a measure of per-driver risk."
                ),
                category="Geography",
            )
        )

    # -- weather -----------------------------------------------------------
    weather = aggregator.counts_by("Weather_Condition", spec, limit=1, db=DB)
    if not weather.empty:
        row = weather.iloc[0]
        findings.append(
            Finding(
                headline=(
                    f"'{row['Weather_Condition']}' is the most frequently recorded weather "
                    f"condition, on {int(row['Accidents']):,} records "
                    f"({row['Accidents'] / total * 100:.1f}%)."
                ),
                evidence=(
                    "Weather is a recorded attribute of the incident, not an independently "
                    "measured exposure."
                ),
                caveat=(
                    "This does not establish that the condition causes accidents; conditions "
                    "are recorded at the time of report, and the dataset does not include "
                    "traffic volume."
                ),
                category="Conditions",
            )
        )

    # -- day type ----------------------------------------------------------
    for column, predicate_label in (
        ("Is_Rush_Hour", "rush-hour periods"),
        ("Is_Weekend", "weekends"),
        ("Day_Night", "night-time records"),
    ):
        frame = _boolean_split(spec, column, predicate_label)
        if frame is None or frame.empty or frame.shape[0] < 2:
            continue
        counts = frame.set_index(frame.columns[0])["Accidents"]
        leader = counts.idxmax()
        findings.append(
            Finding(
                headline=(
                    f"{leader} accounts for {int(counts.max()):,} of {total:,} records "
                    f"({counts.max() / total * 100:.1f}%)."
                ),
                evidence=(
                    " vs ".join(f"{idx}: {int(value):,}" for idx, value in counts.items())
                ),
                caveat=EXPOSURE_DISCLAIMER,
                category="Time",
            )
        )
        break  # one time-based comparison is enough for the summary

    # -- model -------------------------------------------------------------
    if model_summary:
        best_key = max(
            model_summary,
            key=lambda k: model_summary[k].get("metrics", {}).get("Macro_F1", float("-inf")),
        )
        best = model_summary[best_key]
        metrics = best.get("metrics", {})
        findings.append(
            Finding(
                headline=(
                    f"{best['label']} scored the highest macro F1 of the models trained "
                    f"({metrics.get('Macro_F1', float('nan')):.3f}) with balanced accuracy "
                    f"{metrics.get('Balanced_Accuracy', float('nan')):.3f}."
                ),
                evidence=(
                    f"Evaluated on {best.get('test_rows', 0):,} held-out records, "
                    f"trained on {best.get('train_rows', 0):,}."
                ),
                caveat=(
                    "These figures describe performance on held-out records from this "
                    "dataset only. They do not establish real-world forecasting skill, and "
                    "the model is not validated for traffic-safety decision making."
                ),
                category="Model",
            )
        )

    return findings


def _boolean_split(spec: FilterSpec, column: str, label: str) -> pd.DataFrame | None:
    """Return counts split by a boolean or small categorical column."""
    if column == "Day_Night":
        return Aggregator(DB).counts_by("Day_Night", spec, db=DB)

    sql = f"""
        SELECT "{column}" AS "{column}", COUNT(*)::BIGINT AS Accidents
        FROM accidents
        WHERE {{where}}
        GROUP BY "{column}"
    """
    try:
        frame = spec.select(sql, DB)
    except Exception:  # noqa: BLE001 - column may be absent
        return None
    if frame.empty:
        return None
    frame[frame.columns[0]] = frame[frame.columns[0]].map(
        {True: "During " + label, False: "Outside " + label}
    )
    return frame


def executive_summary(
    profile: DatasetProfile,
    findings: list[Finding],
    *,
    max_findings: int = 5,
) -> str:
    """Compose a short narrative summary suitable for a presentation.

    Args:
        profile: Dataset headline facts.
        findings: Computed findings.
        max_findings: How many findings to include.

    Returns:
        A prose summary. Every figure in it comes from the arguments.
    """
    lines = [
        f"This project analyses {profile.total_records:,} recorded traffic accidents "
        f"spanning {profile.coverage_start[:10]} to {profile.coverage_end[:10]}, "
        f"across {profile.distinct_states} states and {profile.distinct_cities:,} cities.",
        "",
        "Key observations:",
    ]
    for index, finding in enumerate(findings[:max_findings], start=1):
        lines.append(f"{index}. {finding.headline} {finding.caveat}".strip())

    lines.extend(
        [
            "",
            f"Interpretation note: {SEVERITY_MEANING}",
            f"{ASSOCIATION_DISCLAIMER}",
            f"{EXPOSURE_DISCLAIMER}",
        ]
    )
    return "\n".join(lines)


def filter_impact(spec: FilterSpec) -> pd.DataFrame:
    """Describe which filters are active, for the sidebar summary.

    Args:
        spec: The active filter set.

    Returns:
        DataFrame with ``Filter`` and ``Value`` rows.
    """
    labels = spec.describe()
    if not labels:
        return pd.DataFrame(columns=["Filter", "Value"])
    return pd.DataFrame({"Filter": [label.split(":")[0] for label in labels],
                         "Value": [label.split(":", 1)[1].strip() for label in labels]})


def dataset_notes() -> list[str]:
    """Return standing caveats that apply to every analytical page."""
    return [
        SEVERITY_MEANING,
        EXPOSURE_DISCLAIMER,
        ASSOCIATION_DISCLAIMER,
        (
            "The last calendar year present in this dataset is incomplete and must "
            "not be compared against full years."
        ),
    ]

"""Page 6 — Weather and road conditions.

Recorded weather attributes and road indicators compared against severity.
Coverage warnings are shown for weather fields that are frequently unrecorded,
because a distribution drawn mostly from imputed values would misrepresent the
data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from smart_traffic import schema
from smart_traffic.aggregation import Aggregator, FilterSpec
from smart_traffic.config import ASSOCIATION_DISCLAIMER, SETTINGS
from smart_traffic.db import DB
from smart_traffic.statistics import severity_by_numeric_band
from smart_traffic.ui import (
    charts_row,
    download_table,
    empty_state,
    global_filters,
    methodology_notes,
    page_header,
    require_dataset,
    section,
    sidebar_header,
)
from smart_traffic.visualizations import (
    bar_chart,
    scatter,
    severity_rate_chart,
)


def _weather_scatter(spec: FilterSpec, limit: int = 20_000) -> pd.DataFrame:
    """Return a reproducible sample of temperature against visibility."""
    from smart_traffic.sampling import draw_sample

    draw = draw_sample(
        ["Temperature(F)", "Visibility(mi)"], spec=spec, limit=limit
    )
    return draw.frame.dropna()


def _indicator_severity(spec: FilterSpec, indicators: list[str]) -> list[dict]:
    """Return the high-severity rate for records carrying each indicator."""
    rows = []
    for indicator in indicators:
        clause, params = spec.where_clause()
        sql = f"""
            SELECT
                COUNT(*)::BIGINT AS Total,
                COALESCE(SUM(CASE WHEN Severity IN (3, 4) THEN 1 ELSE 0 END), 0)::BIGINT
                    AS High
            FROM accidents
            WHERE {clause or 'TRUE'} AND "{indicator}" = TRUE
        """
        row = DB.sql(sql, params)
        if row.empty:
            continue
        total = int(row["Total"][0]) or 0
        high = int(row["High"][0]) or 0
        if total < 25:
            continue
        rows.append(
            {
                "Road_Feature": indicator,
                "Label": schema.ROAD_FEATURE_LABELS.get(indicator, indicator),
                "Total": total,
                "High_Severity": high,
                "High_Severity_Rate": round(high / total * 100, 4),
            }
        )
    return sorted(rows, key=lambda r: r["High_Severity_Rate"], reverse=True)


st.set_page_config(page_title="Weather & Road Conditions", page_icon="🌧️", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Weather and Road Conditions",
    "Recorded environmental and infrastructure attributes, compared against "
    "recorded severity.",
    icon="🌧️",
)

spec = global_filters(show_cities=False)
aggregator = Aggregator(DB)

# -- coverage --------------------------------------------------------------
section(
    "Field coverage",
    "Weather readings are frequently unrecorded in this dataset. Charts below are "
    "computed from recorded values only, so a sparse field shows a smaller "
    "denominator than the total record count.",
)

total_records = aggregator.total(spec)
rows = []
for column in schema.NUMERIC_COLUMNS:
    if not DB.has_column(column):
        continue
    present = int(
        spec.select(
            f'SELECT COUNT(*) AS n FROM accidents WHERE {{where}} AND "{column}" IS NOT NULL',
            DB,
        )["n"][0]
    )
    rows.append({"Field": column, "Recorded": present})

coverage = pd.DataFrame(rows)
if not coverage.empty:
    coverage["Missing"] = total_records - coverage["Recorded"]
    coverage["Coverage (%)"] = (
        (coverage["Recorded"] / total_records * 100).round(2) if total_records else 0.0
    )
    coverage = coverage.sort_values("Coverage (%)")

    low_coverage = coverage[coverage["Coverage (%)"] < 90]
    if not low_coverage.empty:
        names = ", ".join(
            f"{row['Field']} ({row['Coverage (%)']:.1f}%)"
            for _, row in low_coverage.iterrows()
        )
        st.warning(
            f"**Partially recorded fields:** {names}. Results for these fields are "
            f"based on the records where they were present, not the full dataset.",
            icon="⚠️",
        )
    st.dataframe(coverage, width="stretch", hide_index=True)


# -- weather ---------------------------------------------------------------
st.markdown("---")
section(
    "Recorded weather conditions",
    "Weather is an attribute recorded at the time of report, not an independently "
    "measured exposure.",
)

weather_counts = aggregator.counts_by("Weather_Condition", spec, limit=SETTINGS.top_n, db=DB)
weather_rates = aggregator.severity_rates(
    "Weather_Condition", spec, limit=SETTINGS.top_n, min_total=25, db=DB
)

if weather_counts.empty:
    empty_state("No weather data is available for the current filters.")
else:
    charts_row(
        [
            bar_chart(
                weather_counts, "Weather_Condition", "Accidents",
                title=f"Most recorded weather conditions (top {SETTINGS.top_n})",
            ),
            scatter(
                _weather_scatter(spec),
                "Temperature(F)", "Visibility(mi)",
                title="Temperature vs visibility",
                x_title="Temperature (°F)", y_title="Visibility (miles)",
                opacity=0.25,
            ),
        ]
    )

    if not weather_rates.empty:
        section(
            "Severity within each weather condition",
            "Rate = records of that severity divided by all records in the same "
            "weather condition.",
        )
        st.plotly_chart(
            severity_rate_chart(weather_rates, "Weather_Condition",
                                title="Severity composition by weather condition"),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "Conditions with very few records produce unstable rates; groups below "
            "25 records are excluded from this chart."
        )
        download_table(weather_rates, "Download weather severity rates", key="dl_weather_rates",
                       filename_hint="weather_severity_rates")

# -- numeric weather bands -------------------------------------------------
st.markdown("---")
section(
    "Severity across bands of numeric weather readings",
    "Records are grouped into bands of each measurement so the severity rate can "
    "be read without plotting millions of points.",
)

measure = st.selectbox(
    "Weather measurement",
    ["Temperature(F)", "Visibility(mi)", "Humidity(%)", "Wind_Speed(mph)",
     "Precipitation(in)", "Pressure(in)"],
    index=0,
)
bands = severity_by_numeric_band(measure, spec, bins=10)

if bands.empty:
    st.info(f"No records with a recorded {measure} for the current filters.")
else:
    st.plotly_chart(
        bar_chart(
            bands, "Band_Label", "Total",
            title=f"Record volume by {measure} band",
            color="#8da0cb",
            x_title=f"{measure} band",
        ),
        width="stretch",
        config={"displaylogo": False},
    )

    high_level = 3
    rate_column = f"Severity_{high_level}_Rate"
    if rate_column in bands.columns:
        st.plotly_chart(
            bar_chart(
                bands, "Band_Label", rate_column,
                title=f"Severity {high_level} rate within each {measure} band",
                color="#d7191c",
                x_title=f"{measure} band",
                y_title=f"Share of the band at severity {high_level} (%)",
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "The denominator for every band is the number of records in that band. "
        "A band with few records yields a rate that can move a lot from one to the next."
    )
    download_table(bands, "Download banded severity rates", key="dl_bands",
                   filename_hint="severity_by_band")

# -- road features ---------------------------------------------------------
st.markdown("---")
section(
    "Road and infrastructure indicators",
    "Boolean flags recorded against each accident. 'Present' means the flag was "
    "set for that record.",
)

road = aggregator.boolean_feature_counts(spec, DB)
if road.empty:
    empty_state("No road indicator columns are present in the processed table.")
else:
    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(
            bar_chart(
                road, "Label", "Accidents",
                title="Records with each road or amenity indicator present",
                horizontal=True,
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    with right:
        st.dataframe(road, width="stretch", hide_index=True)
        st.caption(
            "Counts are not independent of each other: a single accident can carry "
            "several indicators at once, so these do not sum to the record total."
        )

# -- road vs severity ------------------------------------------------------
st.markdown("---")
section(
    "Road indicators and severity",
    "Severity rates within records that have each indicator versus those that do not.",
)

with st.spinner("Computing severity rates by road indicator..."):
    indicator_rows = _indicator_severity(spec, road["Road_Feature"].tolist())

if indicator_rows:
    frame = pd.DataFrame(indicator_rows)
    st.plotly_chart(
        bar_chart(
            frame, "Label", "High_Severity_Rate",
            title="Severity 3+4 rate when the indicator is present",
            color="#d7191c",
            horizontal=True,
            y_title="Share of records at severity 3 or 4 (%)",
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "This is the rate *within records that have the indicator*. It is not the "
        "indicator's share of all accidents, and it does not mean the indicator "
        "causes a more severe accident."
    )
    download_table(frame, "Download indicator severity rates", key="dl_indicator",
                   filename_hint="road_indicator_severity_rates")

methodology_notes(include_severity=False)
st.markdown(f"_{ASSOCIATION_DISCLAIMER}_")


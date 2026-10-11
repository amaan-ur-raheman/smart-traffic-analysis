"""Page 3 — Traffic overview dashboard.

The headline counts and the first look at where and when accidents are
recorded. Every chart aggregates in SQL against Parquet, so nothing here
materialises the full dataset in memory.
"""

from __future__ import annotations

import streamlit as st

from smart_traffic.aggregation import Aggregator
from smart_traffic.config import SETTINGS
from smart_traffic.db import DB
from smart_traffic.ui import (
    charts_row,
    download_table,
    empty_state,
    global_filters,
    methodology_notes,
    page_header,
    require_dataset,
    section,
    severity_kpi_row,
    sidebar_header,
)
from smart_traffic.visualizations import (
    bar_chart,
    line_chart,
    pie_chart,
    severity_distribution_chart,
)

st.set_page_config(page_title="Traffic Overview", page_icon="📊", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Traffic Overview",
    "Headline counts across the loaded dataset, respecting the filters in the sidebar.",
    icon="📊",
)

spec = global_filters(show_cities=False)
aggregator = Aggregator(DB)

with st.spinner("Computing overview..."):
    severity = aggregator.severity_counts(spec, DB)
    by_year = aggregator.coverage_by_year(spec, DB)
    by_state = aggregator.counts_by("State", spec, limit=SETTINGS.top_n, db=DB)
    by_city = aggregator.counts_by("City", spec, limit=SETTINGS.top_n, db=DB)
    by_hour = aggregator.counts_by("Hour", spec, order_desc=False, db=DB)
    by_month = aggregator.counts_by("Month_Name", spec, order_desc=False, db=DB)
    by_day = aggregator.counts_by("Day_Name", spec, order_desc=False, db=DB)

if severity.empty or severity["Accidents"].sum() == 0:
    empty_state(
        "No records match the current filters.",
        detail="Widen the filters in the sidebar, or reset them, to see results.",
    )
    st.stop()

# -- KPIs ------------------------------------------------------------------
section("Key indicators")
severity_kpi_row(spec)
methodology_notes(include_severity=False)

# -- severity --------------------------------------------------------------
section(
    "Severity distribution",
    "Counts and shares of each recorded severity category.",
)
charts_row(
    [
        severity_distribution_chart(severity, title="Recorded accidents by severity"),
        pie_chart(
            severity.rename(columns={"Percentage": "Share"}),
            "Severity", "Accidents",
            title="Share of the total",
        ),
    ]
)

# -- time ------------------------------------------------------------------
section("Accidents over time")
charts_row(
    [
        line_chart(
            by_year, "Year", "Accidents",
            title="Records per year",
            x_title="Year",
        ),
        bar_chart(
            by_month, "Month_Name", "Accidents",
            title="Records by month",
        ),
    ]
)

if not by_year.empty:
    partial = by_year[by_year["Is_Partial"]]
    if not partial.empty:
        year = int(partial.iloc[0]["Year"])
        coverage_warning_text = (
            f"⚠️ **{year} is incomplete** ({partial.iloc[0]['Coverage'] or 'part year'}) and "
            f"its lower count reflects shorter coverage, not fewer accidents. It is excluded "
            f"from any full-year comparison."
        )
        st.warning(coverage_warning_text, icon="📉")

st.caption("📋 Records by year, with the coverage span observed in each:")
st.dataframe(
    by_year.rename(columns={"Is_Partial": "Partial year"}),
    width="stretch", hide_index=True,
)

charts_row(
    [
        bar_chart(by_hour, "Hour", "Accidents", title="Records by hour of day"),
        bar_chart(by_day, "Day_Name", "Accidents", title="Records by day of week"),
    ]
)

# -- geography -------------------------------------------------------------
section(
    "Where accidents are recorded",
    "The largest counts in the loaded data. These reflect both incident volume "
    "and reporting practice — not per-driver risk.",
)
charts_row(
    [
        bar_chart(
            by_state, "State", "Accidents",
            title=f"Top {SETTINGS.top_n} states by recorded accidents",
        ),
        bar_chart(
            by_city, "City", "Accidents",
            title=f"Top {SETTINGS.top_n} cities by recorded accidents",
        ),
    ]
)

# -- export ----------------------------------------------------------------
section("Export")
st.markdown(
    "Download the tables behind this page. Each file records the filters "
    "active at export time."
)
left, mid, right = st.columns(3)
with left:
    download_table(severity, "Severity", key="ov_sev", filename_hint="severity_distribution")
with mid:
    download_table(by_year, "By year", key="ov_year", filename_hint="year_distribution")
with right:
    download_table(by_state, "By state", key="ov_state", filename_hint="state_distribution")

left, mid, right = st.columns(3)
with left:
    download_table(by_hour, "By hour", key="ov_hour", filename_hint="hour_distribution")
with mid:
    download_table(by_day, "By day", key="ov_day", filename_hint="day_distribution")
with right:
    download_table(by_city, "By city", key="ov_city", filename_hint="city_distribution")

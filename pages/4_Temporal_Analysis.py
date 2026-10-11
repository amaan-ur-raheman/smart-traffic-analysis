"""Page 4 — Temporal analysis.

Time patterns in recorded accidents, including rush-hour, weekday and
day/night breakdowns. Counts are deliberately labelled as recorded frequency:
without traffic-volume data they cannot be read as per-driver risk.
"""

from __future__ import annotations

import streamlit as st

from smart_traffic.aggregation import Aggregator
from smart_traffic.db import DB
from smart_traffic.feature_engineering import rush_hour_definition_text
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
from smart_traffic.visualizations import bar_chart, heatmap, pie_chart

st.set_page_config(page_title="Temporal Analysis", page_icon="⏰", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Temporal Analysis",
    "When accidents are recorded, by hour, day, month and year.",
    icon="⏰",
)

spec = global_filters(show_cities=False)
aggregator = Aggregator(DB)

with st.spinner("Computing time patterns..."):
    by_hour = aggregator.counts_by("Hour", spec, order_desc=False, db=DB)
    by_day = aggregator.counts_by("Day_Name", spec, order_desc=False, db=DB)
    by_month = aggregator.counts_by("Month_Name", spec, order_desc=False, db=DB)
    by_year = aggregator.coverage_by_year(spec, DB)
    day_type = aggregator.counts_by("Day_Night", spec, db=DB)
    hour_day = aggregator.heatmap("Day_Name", "Hour", spec, db=DB)
    month_year = aggregator.heatmap("Year", "Month_Name", spec, db=DB)

if by_hour.empty or by_hour["Accidents"].sum() == 0:
    empty_state(
        "No records match the current filters.",
        detail="Widen the filters in the sidebar to see temporal patterns.",
    )
    st.stop()

total = int(by_hour["Accidents"].sum())
st.caption(f"Showing {total:,} records for the current filters.")

# -- core patterns ---------------------------------------------------------
section(
    "Records by hour of day and day of week",
    "Local clock hour, derived from the recorded start timestamp.",
)
charts_row(
    [
        bar_chart(by_hour, "Hour", "Accidents", title="Records by hour of day", x_title="Hour"),
        bar_chart(by_day, "Day_Name", "Accidents", title="Records by day of week"),
    ]
)

if not hour_day.empty:
    st.plotly_chart(
        heatmap(hour_day, "Day_Name", title="Records by day of week and hour", height=460),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Darker cells record more accidents. This shows when accidents are "
        "recorded, not when driving is most dangerous per unit of exposure."
    )

# -- month / year ----------------------------------------------------------
section("Records by month and year")
charts_row(
    [
        bar_chart(by_month, "Month_Name", "Accidents", title="Records by month"),
        bar_chart(by_year, "Year", "Accidents", title="Records by year"),
    ]
)

if not by_year.empty:
    partial = by_year[by_year["Is_Partial"]]
    if not partial.empty:
        year = int(partial.iloc[0]["Year"])
        st.warning(
            f"**{year} covers only {partial.iloc[0]['Coverage'] or 'part of the year'}.** "
            f"Its count is lower because coverage is shorter, not because fewer "
            f"accidents occurred. It is not comparable to the full years above.",
            icon="📉",
        )

if not month_year.empty:
    with st.expander("Month-by-year heatmap", expanded=False):
        st.plotly_chart(
            heatmap(month_year, "Year", title="Records by year and month", height=520),
            width="stretch",
            config={"displaylogo": False},
        )

# -- categories ------------------------------------------------------------
section(
    "Weekday, rush hour and day/night comparisons",
    f"{rush_hour_definition_text()} 'Day' and 'Night' come from the dataset's own "
    f"Sunrise_Sunset field; unrecognised values are reported as Unknown rather "
    f"than being assumed.",
)

weekend = aggregator.counts_by("Is_Weekend", spec, db=DB)
rush = aggregator.counts_by("Is_Rush_Hour", spec, db=DB)

if not weekend.empty:
    weekend = weekend.copy()
    weekend["Category"] = weekend["Is_Weekend"].map({True: "Weekend", False: "Weekday"})
if not rush.empty:
    rush = rush.copy()
    rush["Category"] = rush["Is_Rush_Hour"].map(
        {True: "Rush hour", False: "Non-rush hour"}
    )

charts_row(
    [
        bar_chart(
            weekend, "Category", "Accidents", title="Weekday versus weekend",
            x_title="Day type",
        ),
        bar_chart(
            rush, "Category", "Accidents", title="Rush hour versus non-rush hour",
            x_title="Period",
        ),
    ],
)

if not day_type.empty:
    st.plotly_chart(
        pie_chart(day_type, "Day_Night", "Accidents", title="Day versus night"),
        width="stretch",
        config={"displaylogo": False},
    )

st.caption(
    "Proportions, not rates: a group holding 40% of the records does not "
    "necessarily carry 40% of the risk. See the Methodology page for why."
)
methodology_notes(include_severity=False)

# -- summary table ---------------------------------------------------------
section("Hourly breakdown")
hourly = by_hour.copy()
hourly["Share of records (%)"] = (hourly["Accidents"] / total * 100).round(2)
st.dataframe(hourly, width="stretch", hide_index=True)

left, mid, right = st.columns(3)
with left:
    download_table(by_hour, "By hour", key="t_hour", filename_hint="temporal_by_hour")
with mid:
    download_table(by_day, "By day", key="t_day", filename_hint="temporal_by_day")
with right:
    download_table(by_month, "By month", key="t_month", filename_hint="temporal_by_month")

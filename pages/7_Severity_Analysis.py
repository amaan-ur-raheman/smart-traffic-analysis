"""Page 7 — Severity analysis.

Severity distributions and rates across groups. Every rate on this page states
its denominator explicitly, because the most common way to misread this
dataset's numbers is to confuse "share of all accidents" with "rate within a
group".
"""

from __future__ import annotations

import streamlit as st

from smart_traffic import schema
from smart_traffic.aggregation import Aggregator
from smart_traffic.config import SETTINGS
from smart_traffic.db import DB
from smart_traffic.ui import (
    charts_row,
    download_table,
    empty_state,
    global_filters,
    info_box,
    methodology_notes,
    page_header,
    require_dataset,
    section,
    severity_kpi_row,
    sidebar_header,
)
from smart_traffic.visualizations import (
    bar_chart,
    pie_chart,
    severity_distribution_chart,
    severity_rate_chart,
)

st.set_page_config(page_title="Severity Analysis", page_icon="📉", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Severity Analysis",
    "How the recorded severity category varies across time, place and conditions.",
    icon="📉",
)

spec = global_filters()
aggregator = Aggregator(DB)

info_box(
    "info",
    "What Severity means here",
    "In this dataset `Severity` is the reported traffic impact or disruption level "
    "on a 1–4 scale. It is **not** a measurement of injuries, fatalities or human harm.",
)

severity = aggregator.severity_counts(spec, DB)
if severity.empty or severity["Accidents"].sum() == 0:
    empty_state(
        "No records match the current filters.",
        detail="Widen the filters in the sidebar to see severity analysis.",
    )
    st.stop()

# -- KPIs ------------------------------------------------------------------
section("Key indicators")
severity_kpi_row(spec)

# -- distribution ----------------------------------------------------------
section("Distribution")
charts_row(
    [
        severity_distribution_chart(severity, title="Records by severity category"),
        pie_chart(
            severity.rename(columns={"Percentage": "Share"}),
            "Severity", "Accidents",
            title="Share of the total",
        ),
    ]
)

# -- rate vs share explanation --------------------------------------------
st.markdown("---")
info_box(
    "warning",
    "Rate is not share",
    "A **rate** divides by the group's own total: *severity 3 records in Texas "
    "divided by all Texas records*. A **share** divides by the grand total: "
    "*Texas records divided by all records*. Every table below names which one "
    "it is showing.",
)

# -- by dimension ----------------------------------------------------------
DIMENSIONS = [
    ("Year", "By year", False),
    ("State", f"By state (top {SETTINGS.top_n})", True),
    ("Hour", "By hour of day", False),
    ("Day_Name", "By day of week", False),
    ("Month_Name", "By month", False),
    ("Weather_Condition", f"By weather condition (top {SETTINGS.top_n})", True),
    ("Day_Night", "By day versus night", False),
    ("Is_Weekend", "By weekday versus weekend", False),
    ("Is_Rush_Hour", "By rush hour category", False),
]

selected = st.multiselect(
    "Dimensions to analyse",
    [label for _, label, _ in DIMENSIONS],
    default=["By hour of day", "By weekday versus weekend"],
    help="Each dimension produces a within-group severity rate table and chart.",
)

for column, label, limit in DIMENSIONS:
    if label not in selected:
        continue

    st.markdown("---")
    section(
        f"Severity {label.lower()}",
        "Left: how accidents are distributed across groups. "
        "Right: the severity composition within each group.",
    )

    with st.spinner(f"Computing severity rates {label.lower()}..."):
        counts = aggregator.counts_by(column, spec, db=DB)
        rates = aggregator.severity_rates(column, spec, limit=limit, min_total=10, db=DB)

    if counts.empty:
        empty_state(f"No data available for {label.lower()}.")
        continue

    display = label.replace(" (top " + str(SETTINGS.top_n) + ")", "")
    charts_row(
        [
            bar_chart(
                counts.head(SETTINGS.top_n) if limit else counts,
                column, "Accidents",
                title=f"Records {display.lower()}",
                horizontal=bool(limit),
            ),
            severity_rate_chart(
                rates, column,
                title=f"Severity composition {display.lower()}",
                limit=None,
            ),
        ]
    )

    with st.expander(f"Table: severity rates {display.lower()}", expanded=False):
        if rates.empty:
            st.info("No groups met the minimum record threshold.")
        else:
            table = rates[
                [
                    column, "Total", "Share_%",
                    *[f"Severity_{lvl}_Count" for lvl in schema.SEVERITY_LEVELS],
                    *[f"Severity_{lvl}_Rate" for lvl in schema.SEVERITY_LEVELS],
                ]
            ]
            st.dataframe(
                table.rename(
                    columns={
                        "Total": "Records",
                        "Share_%": "Share of all (%)",
                        **{f"Severity_{lvl}_Count": f"Sev {lvl} count" for lvl in
                            schema.SEVERITY_LEVELS},
                        **{f"Severity_{lvl}_Rate": f"Sev {lvl} rate (%)" for lvl in
                            schema.SEVERITY_LEVELS},
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "**Records** is the denominator for the severity rate columns. "
                "**Share of all (%)** is the group's share of the whole filtered set "
                "and is a different quantity."
            )
            download_table(
                rates, f"Download {label.lower()} rates", key=f"dl_sev_{column}",
                filename_hint=f"severity_rates_by_{column.lower()}",
            )

# -- imbalance -------------------------------------------------------------
st.markdown("---")
section(
    "Class imbalance and why it matters",
    "The target is heavily imbalanced, which shapes both the metrics and the model.",
)

total = int(severity["Accidents"].sum())
largest = int(severity["Accidents"].max())
minority = int(severity["Accidents"].min())

left, right = st.columns([1, 1.2])
with left:
    st.metric("Largest class share", f"{largest / total * 100:.1f}%")
    st.metric("Smallest class share", f"{minority / total * 100:.2f}%")
    st.metric("Majority-to-minority ratio", f"{largest / max(minority, 1):,.0f} : 1")

with right:
    st.markdown(
        f"""
A model that predicted severity **{int(severity.loc[severity['Accidents'].idxmax(), 'Severity'])}**
for every record would already score
**{largest / total * 100:.1f}% accuracy** while being useless for the other three
categories. For that reason:

- Every model is trained with **balanced class weights**.
- Models are compared on **macro F1** and **balanced accuracy**, not accuracy.
- The confusion matrix is read alongside the metrics, because an aggregate score
  can hide a model that never predicts the rare classes at all.
"""
    )

methodology_notes()

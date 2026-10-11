"""Smart Traffic Analysis and Accident Severity Prediction.

Application entry point. The landing page states what the project is, what data
it is currently running on, and -- when nothing is loaded -- exactly how to get
the dataset.

Run with:

    uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from smart_traffic import __version__
from smart_traffic.aggregation import FilterSpec
from smart_traffic.analytics import build_profile
from smart_traffic.config import (
    ASSOCIATION_DISCLAIMER,
    DATASET_DISPLAY_NAME,
    DATASET_SOURCE_URL,
    EXPOSURE_DISCLAIMER,
    SEVERITY_MEANING,
)
from smart_traffic.db import DB
from smart_traffic.modeling import available_models
from smart_traffic.ui import (
    coverage_warning,
    download_table,
    info_box,
    initialise,
    methodology_notes,
    page_header,
    require_dataset,
    section,
    sidebar_footer,
    sidebar_header,
)
from smart_traffic.visualizations import workflow_diagram

WORKFLOW_STAGES = (
    "Raw dataset",
    "Data validation",
    "Feature engineering",
    "Exploratory analysis",
    "Statistical analysis",
    "Machine learning",
    "Interactive insights",
)


def main() -> None:
    """Render the landing page."""
    initialise()
    sidebar_header()

    if not require_dataset():
        sidebar_footer()
        return

    spec = st.session_state.get("filters") or FilterSpec()
    if not isinstance(spec, FilterSpec):
        spec = FilterSpec()
    sidebar_footer()

    page_header(
        "Smart Traffic Analysis and Accident Severity Prediction",
        "Pattern discovery, statistical analysis and severity classification "
        "across US traffic accident records, 2016 to 2023.",
        icon="🚦",
    )

    profile = build_profile(spec)

    # -- current dataset ---------------------------------------------------
    section(
        "Currently loaded dataset",
        "Every figure in this application is computed from the data described here.",
    )

    left, middle, right = st.columns(3)
    with left:
        st.metric("Records analysed", f"{profile.total_records:,}")
        st.metric("Unique accident IDs", f"{profile.unique_ids:,}")
        st.metric("States represented", f"{profile.distinct_states:,}")
    with middle:
        st.metric("Cities represented", f"{profile.distinct_cities:,}")
        st.metric(
            "Coverage",
            f"{profile.coverage_start[:10]} to {profile.coverage_end[:10]}",
        )
        st.metric("Years present", len(profile.years_present))
    with right:
        st.metric("Severity 3 + 4", f"{profile.high_severity_count():,}")
        st.metric("Severity 3 + 4 share", f"{profile.high_severity_rate():.1f}%")
        models = available_models()
        st.metric("Models trained", len(models))

    coverage_warning(profile)

    # -- problem and scope -------------------------------------------------
    st.markdown("---")
    left, right = st.columns([1.15, 1])
    with left:
        section("Problem statement")
        st.markdown(
            """
Traffic authorities need to understand **when**, **where** and **under what
conditions** accidents are recorded most often, and whether the recorded
severity of an accident can be anticipated from the circumstances known at the
moment it is reported.

This project answers that with the **US Accidents (2016–2023)** dataset: a
public, large-scale record of traffic events across the United States. At over
seven million rows it is too large to explore casually and too large to model
naively, which makes it a genuine data-engineering exercise rather than a
notebook exercise.
"""
        )
        st.markdown(
            f"**Dataset source:** [{DATASET_DISPLAY_NAME}]({DATASET_SOURCE_URL}) "
            f"(Sobhan Moosavi, Kaggle)."
        )

    with right:
        section("Analytical workflow")
        st.plotly_chart(
            workflow_diagram(WORKFLOW_STAGES),
            width="stretch",
            config={"displayModeBar": False, "displaylogo": False},
        )

    # -- severity mix ------------------------------------------------------
    st.markdown("---")
    section(
        "Severity distribution",
        "What the `Severity` column actually contains, and how it is distributed.",
    )
    severity = profile.severity_counts
    if severity.empty:
        st.info("No severity data is available for the current filters.")
    else:
        from smart_traffic.visualizations import severity_distribution_chart

        left, right = st.columns([1.4, 1])
        with left:
            st.plotly_chart(
                severity_distribution_chart(severity, title=None),
                width="stretch",
                config={"displaylogo": False},
            )
        with right:
            st.dataframe(
                severity.rename(
                    columns={
                        "Severity": "Severity",
                        "Accidents": "Records",
                        "Percentage": "Share (%)",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            download_table(severity, "Download severity distribution", key="dl_sev")

        info_box(
            "info",
            "Severity measures traffic impact, not harm",
            SEVERITY_MEANING,
        )

    # -- what the application does ----------------------------------------
    st.markdown("---")
    section("What the application covers")
    capabilities = [
        ("Data ingestion & quality",
         "Chunked CSV-to-Parquet conversion, schema and value validation, and\n"
         "missing-value profiling."),
        ("Temporal analysis",
         "Hour, day, month and year patterns, rush hour and weekday comparisons, heatmaps."),
        ("Geographic analysis",
         "Grid-aggregated density mapping, state and city hotspots, validated coordinates."),
        ("Weather & road conditions",
         "Recorded weather attributes and road indicators compared against severity."),
        ("Severity analysis",
         "Within-group severity rates with an explicit denominator on every figure."),
        ("Statistics",
         "Descriptive statistics, correlation, chi-square and Mann-Whitney tests."),
        ("PCA & clustering",
         "Dimensionality reduction and K-Means / hierarchical clustering on reproducible samples."),
        ("Severity prediction",
         "Logistic Regression, Decision Tree and Random Forest with leakage controls."),
        ("Explainability",
         "Feature importance from the fitted models, plus permutation importance."),
    ]
    for title, description in capabilities:
        st.markdown(f"- **{title}** — {description}")

    # -- limitations -------------------------------------------------------
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        section("Limitations")
        st.markdown(
            f"""
- **Not a measure of risk.** {EXPOSURE_DISCLAIMER}
- **Not causal.** {ASSOCIATION_DISCLAIMER}
- **Severity semantics.** {SEVERITY_MEANING}
- **Reporting bias.** The dataset reflects accidents reported to the source,
  which depends on local reporting practice and coverage.
- **Partial final year.** The last year present is incomplete and is excluded
  from full-year comparisons.
- **Models are exploratory.** They are trained and evaluated on this dataset
  only, and are not validated for operational traffic-safety use.
"""
        )
    with right:
        section("Methodology in brief")
        st.markdown(
            """
1. **Ingestion** streams the 3 GB CSV in fixed chunks into Parquet, so memory
   stays bounded regardless of file size. No record is dropped: invalid values
   are flagged, and the data-quality page reports exactly how many.
2. **Querying** runs aggregates in DuckDB directly against the Parquet file,
   so a chart costs one scan instead of a full materialisation.
3. **Modelling** uses only features available when an accident is first
   reported. Duration and `End_Time` are excluded by an enforced policy.
4. **Expensive analysis** (PCA, clustering, training) runs on reproducible
   samples selected by `hash(ID, seed)`, so reruns give identical results.
5. **Evaluation** uses macro F1 and balanced accuracy rather than accuracy,
   because the target is heavily imbalanced.
"""
        )

    methodology_notes()
    st.caption(f"Application version {__version__}. Metrics on: {DB.parquet_path.name}.")


main()

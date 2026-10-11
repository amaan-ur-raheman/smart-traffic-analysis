"""Page 14 — Methodology and limitations.

The reference page. Every other page links here for the caveats that qualify its
numbers, so that the qualifications live in one place and stay accurate.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from smart_traffic import schema
from smart_traffic.aggregation import Aggregator, FilterSpec
from smart_traffic.config import (
    ASSOCIATION_DISCLAIMER,
    DATASET_DISPLAY_NAME,
    DATASET_SOURCE_URL,
    EXPOSURE_DISCLAIMER,
    PATHS,
    SETTINGS,
    SEVERITY_MEANING,
)
from smart_traffic.db import DB
from smart_traffic.evaluation import MAJORITY_BASELINE_NOTE, PRIMARY_METRIC
from smart_traffic.feature_engineering import rush_hour_definition_text
from smart_traffic.modeling import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    EXCLUDED_FEATURES,
    MAX_ONE_HOT_CARDINALITY,
)
from smart_traffic.ui import (
    download_table,
    info_box,
    page_header,
    require_dataset,
    section,
    severity_regime_warning,
    sidebar_header,
)
from smart_traffic.visualizations import workflow_diagram

st.set_page_config(page_title="Methodology", page_icon="📖", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Methodology and Limitations",
    "How this application works, what its numbers mean, and what they cannot mean.",
    icon="📖",
)

STAGES = (
    "Raw dataset", "Data validation", "Feature engineering",
    "Exploratory analysis", "Statistical analysis",
    "Machine learning", "Interactive insights",
)

# -- problem ---------------------------------------------------------------
section("Problem statement")
st.markdown(
    f"""
This project analyses **{DATASET_DISPLAY_NAME}**, a public record of traffic
accidents reported across the United States between February 2016 and March 2023.

It asks two questions:

1. **Descriptive** — when, where and under what recorded conditions do accidents
   appear most often, and how does the recorded severity category vary across
   those groups?
2. **Predictive** — can the recorded severity category be anticipated from the
   circumstances known at the moment an accident is first reported?

The second question is the harder one, and its answer is qualified by everything
on this page.
"""
)
st.link_button("Dataset source on Kaggle", DATASET_SOURCE_URL)

# -- workflow --------------------------------------------------------------
st.markdown("---")
section("Analytical workflow")
st.plotly_chart(
    workflow_diagram(STAGES, height=150),
    width="stretch",
    config={"displayModeBar": False, "displaylogo": False},
)
st.caption(
    "Each stage feeds the next. Nothing is presented before the stage that "
    "produces it has run on the loaded data."
)

# -- pipeline --------------------------------------------------------------
st.markdown("---")
section(
    "Data pipeline",
    "How a 3 GB CSV is processed without loading it into memory.",
)

st.markdown(
    f"""
**Ingestion.** The source CSV is streamed in chunks of
{SETTINGS.csv_chunk_size:,} rows into a single compressed Parquet file. Peak memory
is bounded by the chunk size regardless of how large the file is. Two
interchangeable strategies are available and are tested to produce identical
results:

- `chunked` (default) — Pandas chunked reads. Exact progress reporting.
- `duckdb` — a single DuckDB push-down pipeline. Substantially faster.

**Record retention.** No record is silently dropped. Validity is recorded in the
flags `Is_Valid_Severity`, `Is_Valid_Coordinate` and `Is_Valid_Record`, so any
exclusion is reproducible and countable. The Data Quality page reports exactly how
many records each flag excludes.

**Timestamp parsing.** The source mixes timestamp formats, and a minority of rows
carry a UTC offset. Offsets are stripped rather than converted, so local
wall-clock time is preserved — converting would silently shift those rows' hour of
day, and hour-of-day is a primary analytical axis here.

**Querying.** Every aggregate runs as SQL against the Parquet file through DuckDB,
so a chart costs one scan rather than a full Pandas materialisation. This is the
single biggest reason the application stays responsive on 7.7 million rows.

**Sampling.** PCA, clustering, correlation tests and model training run on
reproducible samples selected by `hash(ID, seed)` rather than `RANDOM()`. Given the
same data and seed, the same rows are drawn every time, which is what makes
reported metrics comparable across runs.
"""
)

# -- feature engineering ---------------------------------------------------
st.markdown("---")
section("Derived features")
st.dataframe(
    pd.DataFrame(
        [
            {"Feature": "Year", "Definition": "Calendar year of Start_Time"},
            {"Feature": "Month / Month_Name", "Definition": "Calendar month (1-12) and its name"},
            {"Feature": "Day", "Definition": "Day of month"},
            {"Feature": "Hour", "Definition": "Local clock hour, 0-23"},
            {"Feature": "DayOfWeek / Day_Name",
             "Definition": "Weekday index (0=Monday) and its name"},
            {"Feature": "Is_Weekend", "Definition": "True for Saturday and Sunday"},
            {"Feature": "Is_Rush_Hour", "Definition": rush_hour_definition_text()},
            {"Feature": "Day_Night", "Definition": "From Sunrise_Sunset: Day, Night, or Unknown"},
            {"Feature": "Is_Night", "Definition": "1 for Night, 0 for Day, missing for Unknown"},
            {"Feature": "Duration_Minutes / Duration_Hours",
             "Definition": "End_Time minus Start_Time. Negative durations become missing. "
                           "POST-ONSET ONLY - never a severity predictor."},
        ]
    ),
    width="stretch", hide_index=True,
)

st.caption(
    "**Unknown** is preserved as its own category rather than folded into Day or "
    "Night. Imputing it would silently invent data for records where the source "
    "was ambiguous."
)

# -- modelling -------------------------------------------------------------
st.markdown("---")
section("Machine learning methodology")

info_box(
    "info",
    "Prediction scenario",
    "**Predict the recorded severity category of an accident using only "
    "information available when the accident is first reported.**",
)

st.markdown(
    """
**Leakage control.** The exclusion is enforced in code rather than by convention:
a feature list that violates the policy raises before any model is fitted.
"""
)

st.markdown("**Excluded features and why:**")
st.dataframe(
    pd.DataFrame(
        [{"Feature": name, "Reason": reason} for name, reason in EXCLUDED_FEATURES.items()]
    ),
    width="stretch", hide_index=True,
)

st.markdown(
    f"""
**Predictors used** ({len(DEFAULT_NUMERIC_FEATURES)} numeric,
{len(DEFAULT_CATEGORICAL_FEATURES)} categorical): time-of-day and calendar fields,
weather measurements, road indicators, and state.

**Preprocessing.** Imputation and scaling happen inside a scikit-learn
`Pipeline`, so they are fitted on the training split only and never see the test
set. Numeric variables are median-imputed; categoricals are most-frequent-imputed
and one-hot encoded with rare levels folded together and a cap of
{MAX_ONE_HOT_CARDINALITY} categories per variable, which bounds memory on
high-cardinality fields like `Weather_Condition`.

**Class imbalance.** Severity 2 dominates. {MAJORITY_BASELINE_NOTE} Every model
is therefore trained with balanced class weights, and models are selected on
**{PRIMARY_METRIC.replace('_', ' ').lower()}** and balanced accuracy rather than accuracy.

**Splitting.** The split is stratified, so the rare categories appear in the test
set in proportion. Without stratification a 20% test split of a category holding
under 1% of the data could leave too few examples to evaluate at all.
"""
)

# -- evaluation ------------------------------------------------------------
st.markdown("---")
section("Evaluation and metrics")
st.markdown(
    f"""
Models are evaluated on held-out records across **accuracy**, **balanced
accuracy**, **macro precision**, **macro recall**, **macro F1**, weighted F1, a
per-class report and a confusion matrix.

The selection metric is **{PRIMARY_METRIC.replace('_', ' ')}**. Accuracy is
reported but not used for selection, for the reason given above.

A point worth stating plainly: these models are evaluated on a held-out slice of
the *same dataset they were trained on*, drawn by the same process. They measure
how well the model reproduces recorded severity patterns in this data. They do
**not** measure forecasting skill against future incidents, which would require
temporal validation — training on one period and testing on a later one. That
experiment is not performed here, and its absence is a real limitation.
"""
)

# -- limitations -----------------------------------------------------------
st.markdown("---")
section("Limitations")

LIMITATIONS = [
    (
        "Severity is traffic impact, not harm",
        SEVERITY_MEANING + " Conclusions about human consequence cannot be drawn "
        "from this dataset at all.",
    ),
    (
        "Counts are frequency, not risk",
        EXPOSURE_DISCLAIMER + " A state with more recorded accidents may simply "
        "have more drivers, more reporting coverage, or both.",
    ),
    (
        "No causal claims",
        ASSOCIATION_DISCLAIMER + " Unmeasured confounding is not controlled for, "
        "so no comparison here identifies an effect.",
    ),
    (
        "Reporting bias",
        "The dataset reflects what was reported to the source. Reporting practice "
        "varies by jurisdiction and over time, which is a plausible contributor to "
        "the year-on-year increase and is not separable from real change.",
    ),
    (
        "Incomplete final year",
        "The last year present covers only part of the calendar year and is "
        "excluded from full-year comparisons on every page.",
    ),
    (
        "Class imbalance limits the ceiling",
        "With under 1% of records at severity 1, per-class recall for the rarest "
        "category will be low for every model tried. This is a property of the "
        "data, not a fixable modelling defect.",
    ),
    (
        "Sampled analysis",
        f"PCA, clustering, correlation and training use samples capped at "
        f"{SETTINGS.sample_clustering_points:,} rows or fewer. Statistics computed "
        f"on a sample carry sampling error and are labelled as such wherever they "
        f"appear.",
    ),
    (
        "Models are exploratory",
        "The models are trained and evaluated on one historical dataset. They are "
        "not validated for real-world traffic-safety use, are not calibrated, and "
        "should not inform operational decisions.",
    ),
    (
        "Free-text not modelled",
        "The `Description` and `Street` columns are excluded from the processed "
        "table. No natural-language feature extraction is claimed.",
    ),
]

for title, body in LIMITATIONS:
    st.markdown(f"**{title}.** {body}")

# -- validation protocol ---------------------------------------------------
st.markdown("---")
section("How the models are validated, and what that does not prove")
severity_regime_warning()

st.markdown("---")
section("Reproducibility")
st.dataframe(
    pd.DataFrame(
        [
            {"Setting": "Rush-hour windows", "Value":
                f"{SETTINGS.rush_hour_morning[0]:02d}:00–{SETTINGS.rush_hour_morning[1]:02d}:59, "
                f"{SETTINGS.rush_hour_evening[0]:02d}:00–{SETTINGS.rush_hour_evening[1]:02d}:59"},
            {"Setting": "Random seed", "Value": str(SETTINGS.random_seed)},
            {"Setting": "Training sample cap", "Value": f"{SETTINGS.sample_training_rows:,}"},
            {"Setting": "Clustering sample cap", "Value": f"{SETTINGS.sample_clustering_points:,}"},
            {"Setting": "PCA sample cap", "Value": f"{SETTINGS.sample_scatter_points:,}"},
            {"Setting": "Hierarchical hard cap", "Value": "5,000 rows"},
            {"Setting": "Valid latitude range",
             "Value": f"{schema.LAT_MIN:g} to {schema.LAT_MAX:g}"},
            {"Setting": "Valid longitude range",
             "Value": f"{schema.LNG_MIN:g} to {schema.LNG_MAX:g}"},
        ]
    ),
    width="stretch", hide_index=True,
)

st.caption(
    "Sampling uses `hash(ID, seed)` rather than `RANDOM()`, so the same seed "
    "selects the same rows on every run. Rerunning the pipeline on unchanged data "
    "reproduces the same metrics exactly."
)

# -- current data ----------------------------------------------------------
st.markdown("---")
section("Data currently loaded")
try:
    counts = Aggregator(DB).severity_counts(FilterSpec(), DB)
    st.dataframe(counts, width="stretch", hide_index=True)
    download_table(counts, "Download severity counts", key="dl_method_sev",
                   filename_hint="severity_counts")
except Exception as exc:  # noqa: BLE001 - the page must still render
    st.caption(f"Could not read the loaded table: {exc}")

st.caption(f"Processed table: `{DB.parquet_path.name}` · Models: `{PATHS.models}`")

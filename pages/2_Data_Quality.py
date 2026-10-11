"""Page 2 — Data ingestion and data quality.

Reports what was actually computed about the loaded data, including
discrepancies against the published reference figures for the original Kaggle
dataset. Discrepancies are surfaced rather than suppressed: a different row
count usually means a different file version, which the reader should know.
"""

from __future__ import annotations

import streamlit as st

from smart_traffic import schema
from smart_traffic.config import PATHS
from smart_traffic.db import DB
from smart_traffic.ui import (
    download_table,
    empty_state,
    page_header,
    require_dataset,
    section,
    sidebar_header,
)
from smart_traffic.validation import (
    missing_value_profile,
    numeric_summary,
    validate_processed_table,
)
from smart_traffic.visualizations import bar_chart

st.set_page_config(page_title="Data Quality", page_icon="🔍", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Data Ingestion and Quality",
    "Validation results computed from the loaded dataset. No record is dropped "
    "without being counted here.",
    icon="🔍",
)

spec = st.session_state.get("filters")
from smart_traffic.aggregation import FilterSpec  # noqa: E402

filter_spec = spec if isinstance(spec, FilterSpec) else FilterSpec()

with st.spinner("Running validation checks..."):
    report = validate_processed_table(DB)
    missing = missing_value_profile(DB)
    numerics = numeric_summary(DB)

# -- summary ---------------------------------------------------------------
section("Validation summary", f"{len(report.checks)} checks across the processed table.")

columns = st.columns(4)
columns[0].metric("Records", f"{report.row_count:,}")
columns[1].metric("Checks passed", f"{report.passed}")
columns[2].metric("Warnings", f"{report.warnings}")
columns[3].metric("Failures", f"{report.failures}")

if report.is_healthy:
    st.success(
        "No check failed. Warnings indicate incompleteness in the source data, "
        "which is expected for weather readings and end coordinates.",
        icon="✅",
    )
else:
    st.error(
        f"{report.failures} check(s) failed. Records affected are flagged in the "
        f"dataset rather than removed, so results below already exclude them.",
        icon="⛔",
    )

with st.expander("Full validation report", expanded=False):
    st.dataframe(report.to_dataframe(), width="stretch", hide_index=True)
    download_table(report.to_dataframe(), "Download validation report", key="dl_validation")

# -- reference comparison --------------------------------------------------
section(
    "Comparison with published reference figures",
    "These are the figures documented for the original Kaggle dataset. They are "
    "shown only to detect that a different file has been supplied; they are "
    "never used as this application's results.",
)

reference_rows = [c for c in report.checks if c.reference is not None]
if reference_rows:
    table = __import__("pandas").DataFrame([c.as_row() for c in reference_rows])
    st.dataframe(
        table.rename(columns={"Check": "Check", "Matches reference": "Matches published figure"}),
        width="stretch",
        hide_index=True,
    )
    mismatches = [c for c in reference_rows if c.matches_reference is False]
    if mismatches:
        names = ", ".join(c.name for c in mismatches)
        st.warning(
            f"**{len(mismatches)} figure(s) differ from the published values:** {names}. "
            f"This is expected if you are running a different file version, a subset, "
            f"or synthetic data. The difference is reported here rather than hidden.",
            icon="⚠️",
        )
    else:
        st.info("All comparable figures match the published values.", icon="✅")
else:
    st.info("The loaded dataset is too small for a reference comparison to be meaningful.")

# -- invalid values vs missing -------------------------------------------
section(
    "Invalid values versus missing values",
    "These are different problems and are reported separately: an invalid value "
    "violates a domain rule, while a missing value was simply never recorded.",
)

left, right = st.columns(2)
with left:
    st.markdown("**Invalid values**")
    st.markdown(
        f"""
- Severity outside the documented categories 1–4
- Coordinates outside the USA bounding box (latitude {schema.LAT_MIN:g}–{schema.LAT_MAX:g},
  longitude {schema.LNG_MIN:g}–{schema.LNG_MAX:g})

Records failing either check are flagged `Is_Valid_Record = FALSE` and excluded
from the analytical pages. They remain in the file and are counted here.
"""
    )
    invalid_frame = report.to_dataframe()
    invalid_frame = invalid_frame[
        invalid_frame["Check"].isin(
            [
                "Invalid severity values",
                "Invalid coordinates",
                "Negative durations",
                "Unparsed start timestamps",
            ]
        )
    ]
    if invalid_frame.empty:
        st.success("No invalid values were detected.", icon="✅")
    else:
        st.dataframe(invalid_frame, width="stretch", hide_index=True)

with right:
    st.markdown("**Missing values**")
    st.markdown(
        "Missing weather readings and end coordinates are normal in this dataset. "
        "They are left as missing and imputed inside the model pipeline, never "
        "silently dropped from the source."
    )
    if missing.empty:
        st.info("No missing-value profile available.")
    else:
        top_missing = missing[missing["Missing_Count"] > 0].head(12)
        if top_missing.empty:
            st.success("No missing values were detected.", icon="✅")
        else:
            st.plotly_chart(
                bar_chart(
                    top_missing.sort_values("Missing_Percent"),
                    "Column",
                    "Missing_Percent",
                    title="Missing values by column (top 12)",
                    color="#d7191c",
                    horizontal=True,
                    y_title="Missing (%)",
                ),
                width="stretch",
                config={"displaylogo": False},
            )

if not missing.empty:
    with st.expander("Complete missing-value profile", expanded=False):
        st.dataframe(missing, width="stretch", hide_index=True)
        download_table(missing, "Download missing-value profile", key="dl_missing")

# -- numeric summary -------------------------------------------------------
section(
    "Numeric feature summary",
    "Computed over the full dataset. Standard deviation is the sample standard deviation.",
)

if numerics.empty:
    empty_state(
        "No numeric columns are available.",
        detail="The processed table does not contain the expected measurement columns.",
    )
else:
    left, right = st.columns([1.6, 1])
    with left:
        st.dataframe(numerics, width="stretch", hide_index=True)
        download_table(numerics, "Download numeric summary", key="dl_numeric")
    with right:
        st.markdown("**Reading this table**")
        st.markdown(
            """
A large gap between `Count` and the row total means the field was frequently
not recorded — imputation during modelling fills those gaps using the training
median, and the imputation happens only on training data.

`Mean` and `Median` far apart indicate a skewed distribution, where the median
is the more representative central value.
"""
        )
        st.caption(f"Total rows in the processed table: {report.row_count:,}")

# -- ingestion provenance --------------------------------------------------
section("Ingestion provenance")
st.markdown(
    f"""
| Item | Value |
| --- | --- |
| Processed table | `{DB.parquet_path.name}` |
| Location | `{DB.parquet_path}` |
| Reproducibility sample | `{"present" if PATHS.sample_parquet.exists() else "not built"}` |
| Source CSV | `{"present" if PATHS.raw_csv.exists() else "not present"}` |
| Expected source location | `{PATHS.raw_csv}` |
    |

Rebuild from the command line at any time:

```bash
uv run sta build
uv run sta validate
```
"""
)

"""Page 8 — Statistical analysis.

Descriptive statistics, correlation and hypothesis tests. Correlation is
presented as association throughout; the tests say whether groups are
distinguishable, not why they differ.
"""

from __future__ import annotations

import streamlit as st

from smart_traffic.config import ASSOCIATION_DISCLAIMER, SETTINGS
from smart_traffic.statistics import (
    chi_square_severity_association,
    correlation_matrix,
    mann_whitney_comparison,
    numeric_descriptives,
)
from smart_traffic.ui import (
    download_table,
    empty_state,
    global_filters,
    info_box,
    methodology_notes,
    page_header,
    require_dataset,
    section,
    sidebar_header,
)
from smart_traffic.visualizations import correlation_heatmap, scatter

st.set_page_config(page_title="Statistical Analysis", page_icon="📐", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Statistical Analysis",
    "Descriptive statistics, correlation structure and group comparisons.",
    icon="📐",
)

spec = global_filters(show_cities=False)

info_box(
    "warning",
    "Correlation is not causation",
    ASSOCIATION_DISCLAIMER
    + " A strong correlation between two recorded variables may reflect a shared "
    "cause, a confounding factor, or the way the data was recorded. None of the "
    "figures here can distinguish these.",
)

# -- descriptive statistics -----------------------------------------------
section(
    "Descriptive statistics",
    "Computed over the full filtered dataset — no sampling is involved, so these "
    "figures are exact.",
)

with st.spinner("Computing descriptive statistics..."):
    descriptives = numeric_descriptives(spec)

if descriptives.empty:
    empty_state("No numeric columns are available for the current filters.")
else:
    left, right = st.columns([1.6, 1])
    with left:
        st.dataframe(descriptives, width="stretch", hide_index=True)
        download_table(descriptives, "Download descriptive statistics", key="dl_desc",
                       filename_hint="numeric_descriptives")
    with right:
        st.markdown("**How to read this**")
        st.markdown(
            """
- **Std** is the sample standard deviation. A value far larger than the mean
  signals a heavy-tailed distribution.
- **Skew** above 1 or below −1 means the mean is a poor summary; use **Median**.
- Weather variables here are often skewed and only partly recorded, which is why
  the modelling pipeline imputes rather than assumes.
"""
        )

# -- correlation -----------------------------------------------------------
st.markdown("---")
section(
    "Correlation structure",
    "Pearson correlation over the numeric variables. Computed row-wise, so this "
    "runs on a reproducible sample above the configured threshold.",
)

with st.spinner("Computing correlations..."):
    correlation = correlation_matrix(spec)

if correlation.matrix.empty:
    empty_state("Not enough numeric variables are available for a correlation analysis.")
else:
    st.caption(
        f"Computed on: **{correlation.computed_on}**."
        + (f" {correlation.sample_note}" if correlation.sample_note else "")
    )
    st.plotly_chart(
        correlation_heatmap(
            correlation.matrix,
            title="Pearson correlation between numeric variables",
            height=max(420, 60 + 32 * len(correlation.matrix.columns)),
            note=f"Computed on {correlation.computed_on}.",
        ),
        width="stretch",
        config={"displaylogo": False},
    )

    strongest = correlation.strongest_pairs(n=12)
    if not strongest.empty:
        section(
            "Strongest pairwise associations",
            description="Each unordered pair listed once.",
        )
        st.dataframe(strongest, width="stretch", hide_index=True)
        st.caption(
            "`Temperature(F)` and `Wind_Chill(F)` correlate strongly because they "
            "measure nearly the same physical quantity. Correlated features split "
            "importance between themselves in a model, which is why importance "
            "rankings are read as groups rather than exact orderings."
        )
        download_table(strongest, "Download strongest correlations", key="dl_corr",
                       filename_hint="strongest_correlations")

    with st.expander("Explore two variables", expanded=False):
        available = list(correlation.matrix.columns)
        left, right = st.columns(2)
        x_column = left.selectbox("X axis", available, index=0)
        y_column = right.selectbox("Y axis", available, index=min(1, len(available) - 1))
        if x_column == y_column:
            st.info("Choose two different variables.")
        else:
            from smart_traffic.sampling import draw_sample

            draw = draw_sample([x_column, y_column], spec=spec,
                               limit=SETTINGS.sample_scatter_points)
            if draw.frame.empty:
                st.info("No records to plot for this selection.")
            else:
                st.plotly_chart(
                    scatter(
                        draw.frame.dropna(), x_column, y_column,
                        title=f"{y_column} against {x_column}",
                        x_title=x_column, y_title=y_column, opacity=0.3,
                    ),
                    width="stretch",
                    config={"displaylogo": False},
                )
                st.caption(draw.note())

# -- group comparisons -----------------------------------------------------
st.markdown("---")
section(
    "Group comparisons",
    "Non-parametric tests comparing a numeric measure between groups. These "
    "detect a difference in distribution; they do not explain it.",
)

test_left, test_right = st.columns(2)
with test_left:
    group_column = st.selectbox(
        "Grouping variable",
        ["Weather_Condition", "State", "Day_Night", "Is_Weekend", "Is_Rush_Hour",
         "Day_Name", "Timezone", "Civil_Twilight"],
        index=0,
    )
with test_right:
    measure_column = st.selectbox(
        "Measurement to compare",
        ["Temperature(F)", "Visibility(mi)", "Humidity(%)", "Wind_Speed(mph)",
         "Precipitation(in)", "Pressure(in)", "Distance(mi)"],
        index=0,
    )

with st.spinner("Running the Mann-Whitney U test..."):
    comparisons = mann_whitney_comparison(group_column, measure_column, spec)

if comparisons.empty:
    empty_state(
        f"Not enough groups to compare for {group_column}.",
        detail="Try a grouping variable with more categories, or widen the filters.",
    )
else:
    st.dataframe(comparisons, width="stretch", hide_index=True)
    st.caption(
        "Mann-Whitney U is used instead of a t-test because weather and severity "
        "variables are skewed and do not meet the normality assumption."
    )
    st.caption(
        f"Comparisons use a reproducible sample of at most "
        f"{SETTINGS.sample_permutation_importance:,} rows, keeping only groups with "
        f"at least 50 records."
    )
    download_table(comparisons, "Download group comparisons", key="dl_mwu",
                   filename_hint="mann_whitney_comparisons")

# -- chi-square ------------------------------------------------------------
st.markdown("---")
section(
    "Is severity associated with this variable?",
    "Chi-square test of independence on the severity × category contingency table, "
    "with Cramér's V as the effect size.",
)

chi_variable = st.selectbox(
    "Variable to test against severity",
    ["Hour", "Day_Name", "Month", "Day_Night", "Is_Weekend", "Is_Rush_Hour",
     "Civil_Twilight", "Timezone", "Weather_Condition"],
    index=0,
)

with st.spinner("Running the chi-square test..."):
    chi = chi_square_severity_association(chi_variable, spec)

if chi.empty:
    empty_state(f"{chi_variable} has too few categories to test.")
else:
    row = chi.iloc[0]
    left, right = st.columns([1, 1.4])
    with left:
        st.metric("Cramér's V", f"{row['Cramers_V']:.3f}")
        st.metric("P-value", f"{row['P_Value']:.2e}")
    with right:
        st.dataframe(chi, width="stretch", hide_index=True)

    st.markdown(
        f"""
**Interpretation.** Severity and `{chi_variable}` are
{'**statistically associated**' if bool(row['Significant_at_0.05'])
 else '**not statistically associated**'}
at the 0.05 level. Cramér's V of **{row['Cramers_V']:.3f}** indicates a
{row['Association_Strength']} association.

Two cautions. First, with millions of records even a negligible association
reaches statistical significance — the effect size matters more than the p-value
here. Second, association says nothing about direction or cause: a significant
result does not mean changing `{chi_variable}` would change severity.
"""
    )

methodology_notes(include_exposure=False)

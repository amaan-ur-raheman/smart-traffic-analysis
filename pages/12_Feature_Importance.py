"""Page 12 — Feature importance and explainability.

A dedicated look at which variables the trained models relied on, and why
importance must be read carefully in an encoded, correlated feature space.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from smart_traffic.config import SETTINGS
from smart_traffic.evaluation import permutation_importance_scores
from smart_traffic.modeling import (
    MODEL_REGISTRY,
    TrainingConfig,
    available_models,
    feature_importance_table,
    load_model,
    load_training_data,
    split_data,
)
from smart_traffic.ui import (
    download_table,
    empty_state,
    info_box,
    page_header,
    require_dataset,
    section,
    sidebar_header,
)
from smart_traffic.visualizations import bar_chart

st.set_page_config(page_title="Feature Importance", page_icon="🔍", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Feature Importance and Explainability",
    "Which variables the trained models relied on, and what that does and does "
    "not mean.",
    icon="🔍",
)

trained = available_models()
if not trained:
    empty_state(
        "No models have been trained yet.",
        detail="Train them on the **Severity Prediction** page, or run "
               "`uv run sta train` in a terminal.",
    )
    st.stop()

key = st.selectbox(
    "Model", trained,
    format_func=lambda k: MODEL_REGISTRY[k]["label"],
)
pipeline = load_model(key)

if pipeline is None:
    st.error(
        f"The saved artefact for `{key}` could not be loaded. Retrain the models "
        f"to regenerate it."
    )
    st.stop()

# -- importance ------------------------------------------------------------
section(
    "Model importance",
    "Taken directly from the fitted estimator.",
)

importance = feature_importance_table(pipeline, key)
if importance is None:
    empty_state(f"{MODEL_REGISTRY[key]['label']} does not expose an importance attribute.")
    st.stop()

method = importance["Type"].iloc[0]
st.caption(
    f"**Method:** {method}. Feature names come from the fitted pipeline's "
    f"transformer, so they reflect what the model actually saw — including "
    f"one-hot expansion."
)

top = importance.head(25)
st.plotly_chart(
    bar_chart(
        top.sort_values("Importance"), "Feature", "Importance",
        title=f"Top {len(top)} features by importance — {MODEL_REGISTRY[key]['label']}",
        color="#1a9641", horizontal=True, y_title="Importance", text_auto=False,
    ),
    width="stretch", config={"displaylogo": False},
)

# -- grouped view ----------------------------------------------------------
section(
    "Importance grouped back to source variables",
    "One-hot encoding splits a categorical variable across many columns. Summing "
    "them restores the original variable's total contribution.",
)


def group_importance(frame: pd.DataFrame) -> pd.DataFrame:
    """Sum encoded feature importances back to their source variable.

    Args:
        frame: Importance table with a ``Feature`` column.

    Returns:
        DataFrame with one row per source variable and its summed importance.
    """
    def source(feature: str) -> str:
        if feature.startswith("cat__"):
            name = feature[len("cat__"):]
            return name.rsplit("_", 1)[0] if name[-1].isupper() or "_" in name else name
        if feature.startswith("num__"):
            return feature[len("num__"):]
        return feature

    frame = frame.copy()
    frame["Source_Variable"] = frame["Feature"].map(source)
    return (
        frame.groupby("Source_Variable", as_index=False)["Importance"].sum()
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


grouped = group_importance(importance)
st.plotly_chart(
    bar_chart(
        grouped.head(25).sort_values("Importance"), "Source_Variable", "Importance",
        title="Importance summed by original variable",
        color="#2c7bb6", horizontal=True, y_title="Total importance", text_auto=False,
    ),
    width="stretch", config={"displaylogo": False},
)

st.dataframe(grouped, width="stretch", hide_index=True)

with st.expander("Full per-column importance", expanded=False):
    st.dataframe(importance, width="stretch", hide_index=True)
    download_table(importance, "Download feature importance", key="dl_imp",
                   filename_hint=f"{key}_feature_importance")
    download_table(grouped, "Download grouped importance", key="dl_imp_grouped",
                   filename_hint=f"{key}_feature_importance_grouped")

# -- permutation -----------------------------------------------------------
st.markdown("---")
section(
    "Permutation importance",
    "Measures the drop in macro F1 when one feature's values are shuffled. This "
    "catches interactions that split-based importance can miss.",
)

controls = st.columns([1, 1, 1.5])
perm_rows = controls[0].number_input(
    "Rows to score", 500, 20_000, 5_000, step=500,
    help="Cost scales with features × repeats × rows.",
)
perm_repeats = controls[1].slider("Repeats", 2, 20, 5)
controls[2].caption(
    "Uses held-out records, never the training split, so the measured drop is "
    "not inflated by the model having already memorised them."
)

if st.button("Compute permutation importance", type="primary", key="perm_run"):
    with st.spinner("Shuffling features and re-scoring…"):
        try:
            frame = load_training_data(TrainingConfig())
            _x_train, x_test, _y_train, y_test = split_data(frame, TrainingConfig())
            st.session_state["permutation_result"] = permutation_importance_scores(
                pipeline, x_test, y_test,
                n_repeats=perm_repeats,
                max_rows=int(perm_rows),
                random_state=SETTINGS.random_seed,
            )
        except (ValueError, MemoryError, KeyError) as exc:
            st.error(f"Permutation importance could not be computed: {exc}")
            st.session_state.pop("permutation_result", None)
    st.rerun()

permutation = st.session_state.get("permutation_result")
if permutation is None:
    st.info("Press the button above to compute permutation importance.")
elif permutation.empty:
    st.info("No features could be scored.")
else:
    st.caption(
        f"Computed on {permutation['Rows_Scored'].iloc[0]:,} held-out rows with "
        f"{permutation['Repeats'].iloc[0]} repeats per feature. Baseline macro F1 on "
        f"those rows: {permutation['Baseline_Macro_F1'].iloc[0]:.4f}."
    )
    st.plotly_chart(
        bar_chart(
            permutation.head(25).sort_values("Importance_Mean"),
            "Feature", "Importance_Mean",
            title="Permutation importance (mean drop in macro F1 when shuffled)",
            color="#7b3294", horizontal=True,
            y_title="Drop in macro F1", text_auto=False,
        ),
        width="stretch", config={"displaylogo": False},
    )
    with st.expander("Full permutation table", expanded=False):
        st.dataframe(permutation, width="stretch", hide_index=True)
        download_table(permutation, "Download permutation importance", key="dl_perm",
                       filename_hint="permutation_importance")

# -- interpretation --------------------------------------------------------
st.markdown("---")
section("How to read feature importance")

info_box(
    "warning",
    "Importance is not causation",
    "A high score means the model **used** that feature to make its predictions. "
    "It does not mean the feature **causes** severity. Three specific traps "
    "apply to this dataset:",
)

st.markdown(
    """
**1. Correlated features share credit arbitrarily.** `Temperature(F)` and
`Wind_Chill(F)` correlate at roughly 0.99 — they are close to the same physical
measurement. A tree that splits on one leaves the other with little apparent
importance, not because it is uninformative but because the information was
already captured. Read correlated features as a group.

**2. One-hot encoding hides categorical variables.** `State` expands into dozens
of columns. Each individual state column carries a fraction of the variable's
importance, so the ranked list will not show "State" anywhere. The grouped table
above restores it.

**3. Aggregate scores hide per-class behaviour.** A model can look strong overall
while never predicting severity 1 correctly. The per-class table on the Model
Evaluation page is the one that shows this.
"""
)

st.markdown(
    """
**Why permutation importance is included.** Impurity-based importance (what the
tree splits on) has a known bias: it favours features offering many distinct
values or high cardinality. Permutation importance measures the actual
predictive contribution and applies no such preference, at higher computational
cost. Where the two disagree, the permutation result is the more trustworthy one.
"""
)

"""Page 11 — Model evaluation.

Compares the trained models on metrics chosen for an imbalanced target, and
shows the confusion matrix and per-class report behind each aggregate score.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from smart_traffic.config import PATHS, SETTINGS
from smart_traffic.evaluation import (
    MAJORITY_BASELINE_NOTE,
    METRIC_EXPLANATIONS,
    METRIC_ORDER,
    SELECTION_METRICS,
    comparison_table,
    permutation_importance_scores,
)
from smart_traffic.modeling import load_model
from smart_traffic.ui import (
    download_table,
    empty_state,
    info_box,
    page_header,
    require_dataset,
    section,
    severity_regime_warning,
    sidebar_header,
)
from smart_traffic.visualizations import bar_chart, confusion_matrix_heatmap

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Model Evaluation",
    "How the models performed on held-out records, and how to choose between them.",
    icon="📊",
)

# -- load persisted results ------------------------------------------------
summary_path = PATHS.models / "training_summary.json"
reports = {}

if summary_path.exists():
    st.session_state["model_summary"] = json.loads(summary_path.read_text())
    for key in st.session_state["model_summary"]:
        report_path = PATHS.models / f"{key}_classification_report.csv"
        matrix_path = PATHS.models / f"{key}_confusion_matrix.csv"
        if report_path.exists():
            reports[key] = {
                "report": pd.read_csv(report_path, index_col=0),
                "confusion": pd.read_csv(matrix_path, index_col=0),
            }

summary = st.session_state.get("model_summary")

if not summary:
    empty_state(
        "No models have been trained yet.",
        detail="Train them on the **Severity Prediction** page, or run "
               "`uv run sta train` in a terminal.",
    )
    st.stop()

# -- comparison table ------------------------------------------------------
section(
    "Model comparison",
    "All models were evaluated on the same held-out records, so these figures "
    "are directly comparable.",
)

# Macro F1 and balanced accuracy reward opposite ends of the same
# precision/recall dial, and on this data they pick different models. Resolving
# that silently in code would hide the choice, so it is exposed here instead.
selection = st.radio(
    "Select the recommended model on",
    list(SELECTION_METRICS),
    index=0,
    horizontal=True,
)
st.caption(SELECTION_METRICS[selection])

collapsed_labels = {m["label"] for m in summary.values() if m.get("is_degenerate", False)}
metric_values = {m["label"]: m["metrics"] for m in summary.values()}

table = comparison_table(
    metric_values, degenerate=collapsed_labels, selection_metric=selection
)
st.dataframe(
    table.round(4),
    width="stretch",
    hide_index=True,
    column_config={
        "Recommended": st.column_config.CheckboxColumn(
            "Recommended", help=f"Highest {selection.replace('_', ' ').lower()}"
        ),
        "Is_Degenerate": st.column_config.CheckboxColumn(
            "Collapsed",
            help="The model failed to predict at least one severity category at all",
        ),
    },
)

st.caption(
    f"Every value was computed on {table['Samples'].iloc[0]:,} evaluation "
    f"records that were not seen during training."
)

# Surface disagreement between the headline metrics rather than hiding it.
winners = {}
for metric in ("Macro_F1", "Balanced_Accuracy", "Accuracy"):
    probe = comparison_table(
        metric_values, degenerate=collapsed_labels, selection_metric=metric
    )
    winners[metric] = probe.loc[probe["Recommended"], "Model"].tolist()
distinct = {w[0] for w in winners.values() if w}
if len(distinct) > 1:
    st.info(
        "**The metrics disagree about which model is best:** "
        + "; ".join(
            f"{metric.replace('_', ' ').lower()} prefers "
            f"**{w[0] if w else 'none'}**"
            for metric, w in winners.items()
        )
        + ". Macro F1 and balanced accuracy pull in opposite directions on a "
        "target this imbalanced, so no model wins on all three. The choice "
        "above decides which one matters for this analysis.",
        icon="\u2696\ufe0f",
    )

collapsed = table.loc[table["Is_Degenerate"], "Model"].tolist()
if collapsed:
    st.error(
        f"**{', '.join(collapsed)} collapsed onto a single class.** It never "
        f"predicts one or more severity categories, so its scores are an artefact "
        f"of the majority class rather than real discrimination. It is excluded "
        f"from selection.",
        icon="\u26d4",
    )
elif not table["Recommended"].any():
    st.warning(
        "Every model collapsed onto a single class. None is recommended: with no "
        "model able to predict more than one severity category, the features "
        "available at accident onset do not support this task on this data.",
        icon="\u26a0\ufe0f",
    )

# -- selection -------------------------------------------------------------
severity_regime_warning()
section("Model selection")
eligible = {
    key: entry for key, entry in summary.items()
    if not entry.get("is_degenerate", False)
}

# max() over a mapping yields the winning *key*; the metadata is looked up from it.
recommended_key = (
    max(eligible, key=lambda k: eligible[k]["metrics"].get(selection, float("-inf")))
    if eligible
    else max(summary, key=lambda k: summary[k]["metrics"].get(selection, float("-inf")))
)
recommended = summary[recommended_key]
recommended_metrics = recommended["metrics"]

columns = st.columns(5)
columns[0].metric("Selected model", recommended["label"])
columns[1].metric("Categories predicted", f"{recommended.get('classes_predicted', 4)} / 4")
columns[2].metric("Macro F1", f"{recommended_metrics['Macro_F1']:.4f}")
columns[3].metric("Balanced accuracy", f"{recommended_metrics['Balanced_Accuracy']:.4f}")
columns[4].metric("Accuracy", f"{recommended_metrics['Accuracy']:.4f}")

st.markdown(
    f"""
**Why {recommended['label']}.** It attains the highest
{selection.replace('_', ' ').lower()} ({recommended_metrics[selection]:.4f}) of the
models trained. {recommended.get('notes', '')}

**Decision rule.** {recommended.get('prediction_threshold', 'argmax')}.

**Why not accuracy alone.** {MAJORITY_BASELINE_NOTE} It is reported for
completeness, and it is one of the selectable metrics above, but on its own it
mostly measures how often the model says "severity 2".

**What the numbers do not say.** These figures come from a held-out slice of the
same dataset the models were trained on, drawn by the same process. They measure
how well the model reproduces recorded severity patterns in this data \u2014 not how
well it would forecast the severity of a future accident, which would additionally
require validation against incidents reported after the training period.
"""
)

# -- metric reference ------------------------------------------------------
with st.expander("What each metric means", expanded=False):
    for metric in METRIC_ORDER:
        if metric in METRIC_EXPLANATIONS:
            st.markdown(f"- **{metric}** — {METRIC_EXPLANATIONS[metric]}")

# -- per-model detail ------------------------------------------------------
section("Per-model detail")
key = st.selectbox(
    "Select a model",
    list(summary.keys()),
    format_func=lambda k: summary[k]["label"],
)

pipeline = load_model(key)
if pipeline is None:
    st.warning(
        f"The fitted artefact for {summary[key]['label']} is not available, so "
        f"additional analysis cannot be run. Its saved report is still shown below."
    )

model_metrics = summary[key]["metrics"]
columns = st.columns(4)
columns[0].metric("Macro F1", f"{model_metrics['Macro_F1']:.4f}")
columns[1].metric("Balanced accuracy", f"{model_metrics['Balanced_Accuracy']:.4f}")
columns[2].metric("Macro precision", f"{model_metrics['Macro_Precision']:.4f}")
columns[3].metric("Macro recall", f"{model_metrics['Macro_Recall']:.4f}")

st.caption(
    f"Trained on {summary[key]['train_rows']:,} records · "
    f"evaluated on {summary[key]['test_rows']:,} · "
    f"seed {summary[key]['seed']} · "
    f"fitted {summary[key]['trained_at']}"
)

if key in reports:
    report = reports[key]["report"]
    confusion = reports[key]["confusion"]

    st.markdown("#### Per-class metrics")
    st.dataframe(report, width="stretch")

    st.markdown(
        """
Reading this table: **precision** is how many of the records predicted as a
class were truly that class; **recall** is how many of the truly-that-class
records were found. A high precision with low recall means the model is
conservative — it rarely predicts that class and is usually right when it does.
"""
    )

    st.markdown("#### Confusion matrix")
    st.plotly_chart(
        confusion_matrix_heatmap(
            confusion,
            title=f"{summary[key]['label']} — confusion matrix",
            height=470,
            note="Rows are the true category; columns are the prediction.",
        ),
        width="stretch",
        config={"displaylogo": False},
    )

    off_diagonal = confusion.to_numpy(dtype=float)
    correct = sum(
        off_diagonal[i][i] for i in range(min(len(off_diagonal), off_diagonal.shape[1]))
    )
    st.caption(
        f"{correct:,.0f} of {off_diagonal.sum():,.0f} evaluation records "
        f"({correct / off_diagonal.sum() * 100:.1f}%) were classified correctly. "
        f"The per-class table above is the more informative view: the aggregate "
        f"can be dominated by the largest category."
    )

    st.markdown("#### Export")
    left, mid = st.columns(2)
    with left:
        download_table(report, "Download per-class report", key=f"dl_rep_{key}",
                       filename_hint=f"{key}_classification_report")
    with mid:
        download_table(confusion.reset_index(), "Download confusion matrix",
                       key=f"dl_cm_{key}", filename_hint=f"{key}_confusion_matrix")

# -- feature importance ----------------------------------------------------
section(
    "Feature importance",
    "Which encoded features contributed most to the selected model's predictions.",
)

if pipeline is None:
    empty_state(
        "Feature importance requires the fitted model.",
        detail="Retrain the models to regenerate the saved artefacts.",
    )
else:
    from smart_traffic.modeling import feature_importance_table

    importance = feature_importance_table(pipeline, key)

    if importance is None:
        empty_state(
            f"{summary[key]['label']} does not expose a feature importance attribute."
        )
    else:
        importance_type = importance["Type"].iloc[0]
        st.caption(
            f"**Method:** {importance_type}. Computed from the fitted estimator on "
            f"the encoded feature names the pipeline produced, so a one-hot column "
            f"appears as `cat__State_CA` rather than ambiguously as `State`."
        )

        top = importance.head(20)
        st.plotly_chart(
            bar_chart(
                top.sort_values("Importance"),
                "Feature", "Importance",
                title=f"Top {len(top)} features — {summary[key]['label']}",
                color="#1a9641",
                horizontal=True,
                y_title="Importance",
                text_auto=False,
            ),
            width="stretch",
            config={"displaylogo": False},
        )

        with st.expander("Full importance table", expanded=False):
            st.dataframe(importance, width="stretch", hide_index=True)
            download_table(importance, "Download feature importance",
                           key=f"dl_fi_{key}", filename_hint=f"{key}_feature_importance")

        # -- permutation importance ----------------------------------------
        st.markdown("#### Permutation importance")
        st.caption(
            "Shuffles one feature at a time and measures how much macro F1 drops. "
            "This captures interactions that a tree's split-based importance can "
            "miss, and it is more expensive to compute."
        )

        rows_to_score = st.number_input(
            "Evaluation rows to score", 500, 20_000, 5_000, step=500,
            help="Capped because the cost scales with features × repeats × rows.",
        )
        repeats = st.slider("Repeats per feature", 2, 20, 5)

        if st.button("Compute permutation importance", key="run_permutation"):
            with st.spinner("Shuffling features…"):
                try:
                    from smart_traffic.modeling import (
                        TrainingConfig,
                        load_training_data,
                        split_data,
                    )

                    config = TrainingConfig()
                    frame = load_training_data(config, db=None)
                    _x_train, x_test, _y_train, y_test = split_data(frame, config)
                    permutation = permutation_importance_scores(
                        pipeline, x_test, y_test,
                        n_repeats=repeats,
                        max_rows=int(rows_to_score),
                        random_state=SETTINGS.random_seed,
                    )
                    st.session_state["permutation"] = permutation
                except (ValueError, MemoryError, KeyError) as exc:
                    st.error(f"Permutation importance could not be computed: {exc}")
            st.rerun()

        permutation = st.session_state.get("permutation")
        if permutation is not None:
            st.dataframe(permutation.head(25), width="stretch", hide_index=True)
            st.caption(
                f"Computed on {permutation['Rows_Scored'].iloc[0]:,} evaluation rows "
                f"with {permutation['Repeats'].iloc[0]} repeats each."
            )
            download_table(permutation, "Download permutation importance",
                           key="dl_perm", filename_hint="permutation_importance")

        info_box(
            "warning",
            "Importance is not causation",
            "A high importance score means the model *used* that feature, not that "
            "the feature *causes* severity. Three consequences:\n\n"
            "- **Correlated features share credit.** When two variables carry the "
            "same signal, the model splits importance between them arbitrarily, so "
            "a low score may reflect a correlated partner rather than uselessness.\n"
            "- **One-hot encoding hides groups.** `State` becomes dozens of columns, "
            "each with a fraction of the original importance. Sum them before judging "
            "the variable.\n"
            "- **Performance varies by class.** Overall metrics hide that a model may "
            "score well on severity 2 while failing on severity 1. Read the per-class "
            "report, not just the headline number.",
        )

st.markdown("---")
st.markdown(
    "Next: the **Key Findings** page assembles the analytical results and the model "
    "performance into a summary you can export."
)

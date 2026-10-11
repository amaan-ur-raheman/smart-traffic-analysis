"""Model evaluation.

Metrics are chosen for an imbalanced four-class target. Plain accuracy is
reported for completeness but is not the basis of model selection: a model
that predicted severity 2 for every record would score about 80% accuracy here
while being useless, because accuracy is dominated by the majority class.

The selection metrics are macro F1 (equal weight to every severity category)
and balanced accuracy (mean recall across categories).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from smart_traffic import schema
from smart_traffic.logging_config import get_logger

logger = get_logger("evaluation")

#: Metric used to pick the recommended model.
PRIMARY_METRIC: Final[str] = "Macro_F1"

#: Metrics the user may select on, with what each one rewards. The two lead to
#: different winners on this data, which is why the choice is exposed rather
#: than silently resolved in code.
SELECTION_METRICS: Final[dict[str, str]] = {
    "Macro_F1": (
        "Weights all four severity categories equally. Rewards a model that is "
        "decent on every class, and is the default."
    ),
    "Balanced_Accuracy": (
        "Mean recall across the four categories. Favours finding the rare "
        "categories at the cost of precision on the common one."
    ),
    "Accuracy": (
        "Overall share correct. Dominated by severity 2, which holds ~80% of "
        "records -- informative here only alongside the other two."
    ),
}

#: Metrics emitted for every model, in display order.
METRIC_ORDER: Final[tuple[str, ...]] = (
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Weighted_F1",
    "Samples",
)

METRIC_EXPLANATIONS: Final[dict[str, str]] = {
    "Accuracy": "Share of all predictions that are correct. Dominated by the majority class here.",
    "Balanced_Accuracy": "Mean recall across the four severity categories. "
    "Ignores class size when averaging.",
    "Macro_Precision": "Mean precision across categories; each category counts equally.",
    "Macro_Recall": "Mean recall across categories; each category counts equally.",
    "Macro_F1": "Mean F1 across categories. The primary model-selection metric for this target.",
    "Weighted_F1": "F1 averaged by class frequency; inflated by the majority class.",
    "Samples": "Number of evaluation records.",
}

#: The score a constant severity-2 classifier would achieve, for reference.
MAJORITY_BASELINE_NOTE: Final[str] = (
    "A classifier that always predicts the most common severity category would "
    "score high Accuracy and near-zero Macro F1. Compare any model against that "
    "pattern before believing its accuracy."
)


@dataclass(slots=True)
class EvaluationResult:
    """Metrics and matrices for one fitted model.

    Attributes:
        metrics: Scalar metrics keyed by display name.
        report: Per-class precision, recall and F1.
        confusion_matrix: Confusion matrix with a labelled index.
        feature_names: Encoded feature names from the fitted transformer.
        predictions: Predicted labels for the evaluation set.
        probabilities: Per-class predicted probabilities, when available.
        actual: True labels for the evaluation set.
        classes_predicted: How many distinct labels the model actually emitted.
    """

    metrics: dict[str, float]
    report: pd.DataFrame
    confusion_matrix: pd.DataFrame
    feature_names: list[str]
    predictions: np.ndarray | None = None
    probabilities: pd.DataFrame | None = None
    actual: np.ndarray | None = None
    classes_predicted: int = 0

    @property
    def macro_f1(self) -> float:
        """Macro F1 score."""
        return float(self.metrics.get("Macro_F1", float("nan")))

    @property
    def is_degenerate(self) -> bool:
        """Whether the model failed to predict at least one severity category.

        This guard exists because *both* headline metrics can be gamed by a model
        that predicts only the majority class. Such a model scores ~0.80 accuracy,
        0.25 balanced accuracy (one class at 1.0, the rest at 0.0) and -- because
        macro F1 averages four classes, three of them zero -- roughly 0.22 macro
        F1, which can exceed the score of a model genuinely discriminating
        between all four. Counting the distinct labels emitted is the only
        reliable discriminator, so it is checked before any model is recommended.
        """
        return self.classes_predicted < len(schema.SEVERITY_LEVELS)


def evaluate_model(
    pipeline: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    labels: tuple[int, ...] = schema.SEVERITY_LEVELS,
) -> EvaluationResult:
    """Evaluate a fitted pipeline on held-out data.

    Args:
        pipeline: A fitted sklearn-compatible estimator exposing ``predict`` and
            usually ``predict_proba``.
        x_test: Evaluation feature rows.
        y_test: True labels.
        labels: The complete label set. Passed explicitly so that a severity
            category absent from the test split still appears in the confusion
            matrix, rather than silently vanishing from the report.

    Returns:
        An :class:`EvaluationResult`.
    """
    predictions = np.asarray(pipeline.predict(x_test))

    metrics = _scalar_metrics(y_test, predictions)

    rows = []
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=list(labels), zero_division=0
    )
    for index, label in enumerate(labels):
        rows.append(
            {
                "Severity": label,
                "Precision": round(float(precision[index]), 4),
                "Recall": round(float(recall[index]), 4),
                "F1": round(float(f1[index]), 4),
                "Support": int(support[index]),
            }
        )
    report = pd.DataFrame(rows)

    matrix = confusion_matrix(y_test, predictions, labels=list(labels))
    confusion = pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )

    feature_names = _feature_names(pipeline)
    probabilities = _probabilities(pipeline, x_test, labels)

    logger.info(
        "Evaluated: accuracy=%.4f balanced=%.4f macro_f1=%.4f",
        metrics["Accuracy"], metrics["Balanced_Accuracy"], metrics["Macro_F1"],
    )

    return EvaluationResult(
        metrics=metrics,
        report=report,
        confusion_matrix=confusion,
        feature_names=feature_names,
        predictions=predictions,
        probabilities=probabilities,
        actual=np.asarray(y_test),
        classes_predicted=int(len(np.unique(predictions))),
    )


def _scalar_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the scalar metric suite.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of metrics keyed by :data:`METRIC_ORDER` names.
    """
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = float(
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )[2]
    )
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Balanced_Accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "Macro_Precision": round(float(precision), 4),
        "Macro_Recall": round(float(recall), 4),
        "Macro_F1": round(float(f1), 4),
        "Weighted_F1": round(weighted_f1, 4),
        "Samples": int(len(y_true)),
    }


def _feature_names(pipeline: Any) -> list[str]:
    """Return encoded feature names from a fitted pipeline's transformer.

    Accepts either a bare pipeline or a
    :class:`~smart_traffic.tuning.ThresholdTunedClassifier` wrapping one.
    """
    from smart_traffic.modeling import estimator_pipeline

    inner = estimator_pipeline(pipeline)
    try:
        return list(inner.named_steps["preprocess"].get_feature_names_out())
    except (AttributeError, KeyError, ValueError):
        estimator = getattr(inner, "named_steps", {}).get("model")
        count = getattr(estimator, "n_features_in_", 0)
        return [f"feature_{i}" for i in range(int(count or 0))]


def _probabilities(
    pipeline: Any, x_test: pd.DataFrame, labels: tuple[int, ...]
) -> pd.DataFrame | None:
    """Return per-class predicted probabilities, when the model exposes them."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        raw = np.asarray(pipeline.predict_proba(x_test))
    except (ValueError, AttributeError):
        return None

    from smart_traffic.modeling import classes_of

    classes = classes_of(pipeline)

    frame = pd.DataFrame(
        raw,
        columns=[f"P(Severity={label})" for label in classes],
    )
    frame.index = x_test.index
    return frame


def comparison_table(
    results: dict[str, dict[str, float]],
    *,
    degenerate: set[str] | None = None,
    selection_metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Build the model comparison table and mark the recommended model.

    Recommendation is the highest ``PRIMARY_METRIC`` among models that actually
    predict every severity category. A degenerate model -- one that collapses
    onto the majority class -- is excluded from selection even if it scores well
    on macro F1, which it can do simply by scoring 1.0 on one class and 0.0 on
    the other three. See :attr:`EvaluationResult.is_degenerate`.

    Args:
        results: Mapping of model label to its metric dictionary.
        degenerate: Labels of models that failed to predict all categories.
        selection_metric: Metric the recommendation is based on.

    Returns:
        DataFrame with one row per model, ordered by ``PRIMARY_METRIC``
        descending, with degeneracy and the recommendation marked.
    """
    if not results:
        return pd.DataFrame(
            columns=["Model", *METRIC_ORDER, "Is_Degenerate", "Recommended"]
        )

    frame = pd.DataFrame.from_dict(results, orient="index")
    frame.index.name = "Model"
    ordered = [m for m in METRIC_ORDER if m in frame.columns]
    sort_metric = selection_metric if selection_metric in ordered else PRIMARY_METRIC
    frame = frame[ordered].sort_values(sort_metric, ascending=False).reset_index()

    collapsed = degenerate or set()
    frame["Is_Degenerate"] = frame["Model"].isin(collapsed)
    frame["Recommended"] = False

    metric = selection_metric if selection_metric in frame.columns else PRIMARY_METRIC
    eligible = frame[~frame["Is_Degenerate"]]
    if not eligible.empty:
        best = eligible[metric].idxmax()
        frame.loc[best, "Recommended"] = True
    elif not frame.empty:
        # Every model collapsed. Recommending one would be misleading, so none
        # is; the caller surfaces this rather than silently picking a bad model.
        pass
    return frame


def permutation_importance_scores(
    pipeline: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    max_rows: int = 5_000,
) -> pd.DataFrame:
    """Compute permutation importance on a bounded slice of the test set.

    Permutation importance measures how much a model's score degrades when one
    feature's values are shuffled. It captures interactions a tree's split-based
    importance can miss, and it is expensive -- so this caps the rows it uses
    and reports how many.

    Note that the returned rows correspond to the *input* columns of
    ``x_test``, not to the pipeline's encoded features. Permutation works by
    shuffling a column of the input, so one encoded one-hot column can never be
    attributed on its own. Names therefore come from ``x_test.columns``; use
    :func:`smart_traffic.modeling.feature_importance_table` for per-encoded-
    column importances.

    Args:
        pipeline: A fitted pipeline.
        x_test: Evaluation features.
        y_test: True labels.
        n_repeats: Shuffles per feature.
        random_state: Seed.
        max_rows: Row cap.

    Returns:
        DataFrame with ``Feature``, ``Importance_Mean``, ``Importance_Std``,
        ``Baseline_Macro_F1``, ``Rows_Scored`` and ``Repeats``.
    """
    from sklearn.inspection import permutation_importance

    # Cap rows to keep the O(features x repeats x rows) cost bounded.
    subset = x_test
    if len(x_test) > max_rows:
        subset = x_test.sample(n=max_rows, random_state=random_state)

    labels = y_test.loc[subset.index] if hasattr(y_test, "loc") else y_test

    result = permutation_importance(
        pipeline,
        subset,
        labels,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="f1_macro",
        n_jobs=-1,
    )

    input_names = list(x_test.columns)
    if len(result.importances_mean) != len(input_names):
        raise ValueError(
            f"Permutation importance returned {len(result.importances_mean)} values "
            f"for {len(input_names)} input columns; the frame and pipeline disagree."
        )

    frame = pd.DataFrame(
        {
            "Feature": input_names,
            "Importance_Mean": np.round(result.importances_mean, 6),
            "Importance_Std": np.round(result.importances_std, 6),
        }
    )
    frame["Baseline_Macro_F1"] = round(_baseline_score(pipeline, subset, labels), 4)
    frame["Rows_Scored"] = len(subset)
    frame["Repeats"] = n_repeats
    return frame.sort_values("Importance_Mean", ascending=False).reset_index(drop=True)


def _baseline_score(pipeline: Any, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Return the model's macro F1 on the rows used for permutation testing."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_test, np.asarray(pipeline.predict(x_test)), average="macro",
        zero_division=0))


def classification_report_text(report: pd.DataFrame) -> str:
    """Render a per-class metrics table as plain text for a download.

    Args:
        report: Per-class metrics DataFrame.

    Returns:
        A CSV-formatted string.
    """
    return report.to_csv(index=False)

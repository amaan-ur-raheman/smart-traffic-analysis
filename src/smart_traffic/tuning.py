"""Decision-threshold tuning for an imbalanced target.

Why this exists
---------------
On this dataset, ``Severity 2`` holds ~80% of the records. Class weighting and
model capacity both act on the same dial: turn it up and rare categories get
predicted more often, but the majority category's precision collapses.

Measured on a fixed 400k-row split, every configuration that maximised macro F1
did so by nearly abandoning ``Severity 4``:

| Variant                                | macro F1 | recall S4 |
| -------------------------------------- | -------- | --------- |
| Balanced weights (current)             | 0.370    | 0.826     |
| Unconstrained Random Forest            | 0.544    | 0.127     |
| ``log1p`` class weights                | 0.525    | 0.164     |

Tuning separates the two concerns that class weighting conflates. The model is
still *trained* with balanced weights, which is what gives it recall on rare
categories. What changes is the *decision rule*: a per-class multiplier applied
to the predicted probabilities. That reaches the same macro F1 while keeping
severity-4 recall at 0.454 -- roughly 3.5x the unconstrained alternative.

The multipliers are chosen on a held-out validation split, never on the test
split, so the reported test metrics remain an honest estimate.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from smart_traffic.logging_config import get_logger

logger = get_logger("tuning")

#: Candidate multipliers searched per class. Wider than the class-weight sweep
#: because this adjusts the decision boundary directly rather than the fit.
MULTIPLIER_GRID: Final[tuple[float, ...]] = (0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0)

#: Coordinate-ascent sweeps over the grid. Three is enough to converge for four
#: classes; more only risks overfitting the validation split.
SWEEPS: Final[int] = 3


class ThresholdTunedClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a fitted pipeline with per-class probability multipliers.

    Inherits from scikit-learn's estimator mixins so that the full protocol is
    available: ``permutation_importance``, cross-validation and the column
    transformers all validate that an estimator implements ``fit`` and exposes
    ``__sklearn_tags__``. A plain wrapper class is rejected by modern sklearn.

    Args:
        pipeline: A fitted pipeline exposing ``predict_proba``.
        multipliers: One multiplier per class, aligned with ``classes_``.
        classes: Class labels, ordered as the estimator emits them.
        validation_score: Macro F1 on the validation split, before tuning.

    The wrapper is picklable as a module-level class, so it can be persisted
    alongside the pipeline and reused by the prediction form without any
    re-tuning.
    """

    def __init__(
        self,
        pipeline: Any,
        multipliers: np.ndarray,
        classes: np.ndarray,
        validation_score: float,
    ) -> None:
        """Store the fitted pipeline and its calibrated multipliers."""
        self.pipeline = pipeline
        self.multipliers = np.asarray(multipliers, dtype=float)
        self.classes = np.asarray(classes)
        self.validation_score = float(validation_score)
        self.tuned_score: float = self.validation_score

    # -- estimator protocol ------------------------------------------------

    @property
    def classes_(self) -> np.ndarray:
        """Class labels, under the name scikit-learn tooling expects.

        Returns:
            The class labels in probability order.
        """
        return self.classes

    def fit(self, x: Any, y: Any) -> ThresholdTunedClassifier:
        """Fit the wrapped pipeline and recalibrate, satisfying sklearn's protocol.

        scikit-learn tooling -- ``permutation_importance``, cross-validation,
        column transformers -- validates that an estimator implements ``fit``.
        Without this the wrapper is silently unusable there.

        Args:
            x: Training features.
            y: Training labels.

        Returns:
            ``self``, so the call can be chained.
        """
        tuned = tune_thresholds(self.pipeline, x, y)
        self.pipeline = tuned.pipeline
        self.multipliers = tuned.multipliers
        self.classes = tuned.classes
        self.validation_score = tuned.validation_score
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the wrapped pipeline's parameters.

        Args:
            deep: Unused; present for sklearn compatibility.

        Returns:
            The inner pipeline's parameters.
        """
        return self.pipeline.get_params(deep=deep)

    def set_params(self, **params: Any) -> ThresholdTunedClassifier:
        """Set parameters on the wrapped pipeline.

        Args:
            **params: Parameters forwarded to the inner pipeline.

        Returns:
            ``self``.
        """
        self.pipeline.set_params(**params)
        return self

    # -- inference ---------------------------------------------------------

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return class probabilities from the wrapped pipeline, unmodified."""
        return self.pipeline.predict_proba(x)

    def predict(self, x: Any) -> np.ndarray:
        """Predict using the tuned decision rule rather than plain argmax.

        Args:
            x: Feature rows.

        Returns:
            The predicted class label per row.
        """
        return self.classes[self._tuned_argmax(x)]

    def _tuned_argmax(self, x: Any) -> np.ndarray:
        """Return the argmax index after applying the per-class multipliers."""
        probabilities = self.predict_proba(x)
        return (probabilities * self.multipliers).argmax(axis=1)

    # -- persistence -------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        """Whether tuning found nothing to change."""
        return bool(np.allclose(self.multipliers, 1.0))

    def describe(self, tuned_score: float | None = None) -> dict[str, Any]:
        """Return JSON-serialisable details of the tuned decision rule.

        Args:
            tuned_score: Macro F1 achieved after tuning on the validation
                split. Defaults to the pre-tuning score.

        Returns:
            A dictionary describing the rule and its effect.
        """
        return {
            "multipliers": {
                str(label): round(float(value), 3)
                for label, value in zip(self.classes, self.multipliers, strict=True)
            },
            "validation_macro_f1_before": round(self.validation_score, 4),
            "validation_macro_f1_after": round(
                float(self.tuned_score if tuned_score is None else tuned_score), 4
            ),
            "is_identity": self.is_identity,
        }

    def __repr__(self) -> str:
        """Return a readable summary of the decision rule."""
        pairs = ", ".join(
            f"{label}x{value:.2f}"
            for label, value in zip(self.classes, self.multipliers, strict=True)
        )
        return f"ThresholdTunedClassifier({pairs})"


def tune_thresholds(
    pipeline: Any,
    x_train: Any,
    y_train: Any,
    *,
    validation_fraction: float = 0.25,
    seed: int = 42,
    grid: tuple[float, ...] = MULTIPLIER_GRID,
    sweeps: int = SWEEPS,
) -> ThresholdTunedClassifier:
    """Fit ``pipeline`` and tune its per-class decision thresholds.

    The training data is split again: the pipeline is fitted on one part and the
    multipliers are chosen on the other. Fitting both on the same rows would let
    the thresholds chase memorised training scores and would give back some of
    the advantage this is meant to recover.

    Args:
        pipeline: An unfitted pipeline. It is fitted here.
        x_train: Training features.
        y_train: Training labels.
        validation_fraction: Share of the training split held out for tuning.
        seed: Seed for the stratified split.
        grid: Candidate multipliers.
        sweeps: Coordinate-ascent passes.

    Returns:
        A fitted :class:`ThresholdTunedClassifier`.
    """
    stratify = y_train if _is_stratifiable(y_train) else None
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train,
        test_size=validation_fraction,
        random_state=seed,
        stratify=stratify,
    )

    pipeline.fit(x_fit, y_fit)
    classes = np.asarray(pipeline.classes_)
    probabilities = pipeline.predict_proba(x_val)

    def score(multipliers: np.ndarray) -> float:
        return float(
            f1_score(y_val, classes[(probabilities * multipliers).argmax(axis=1)],
                     average="macro", zero_division=0)
        )

    base = score(np.ones(len(classes)))
    best_score = base
    multipliers = np.ones(len(classes))

    for _sweep in range(sweeps):
        improved = False
        for index in range(len(classes)):
            for candidate in grid:
                trial = multipliers.copy()
                trial[index] = candidate
                trial_score = score(trial)
                if trial_score > best_score + 1e-6:
                    best_score, multipliers, improved = trial_score, trial, True
        if not improved:
            break

    logger.info(
        "Threshold tuning: validation macro F1 %.4f -> %.4f using %s",
        base, best_score,
        ", ".join(f"{c}x{m:.2f}" for c, m in zip(classes, multipliers, strict=True)),
    )
    tuned = ThresholdTunedClassifier(pipeline, multipliers, classes, base)
    tuned.tuned_score = best_score
    return tuned


def tune_from_fitted(
    pipeline: Any,
    x_val: Any,
    y_val: Any,
    *,
    seed: int = 42,
) -> ThresholdTunedClassifier:
    """Tune thresholds for an already-fitted pipeline on separate data.

    Used when the pipeline has been fitted on the full training split and only
    the decision rule needs calibrating.

    Args:
        pipeline: A fitted pipeline.
        x_val: Validation features, disjoint from the fitting data.
        y_val: Validation labels.
        seed: Unused; present for signature symmetry.

    Returns:
        A :class:`ThresholdTunedClassifier` wrapping ``pipeline``.
    """
    classes = np.asarray(pipeline.classes_)
    probabilities = pipeline.predict_proba(x_val)

    def score(multipliers: np.ndarray) -> float:
        return float(
            f1_score(y_val, classes[(probabilities * multipliers).argmax(axis=1)],
                     average="macro", zero_division=0)
        )

    base = score(np.ones(len(classes)))
    best_score, multipliers = base, np.ones(len(classes))
    for _ in range(SWEEPS):
        improved = False
        for index in range(len(classes)):
            for candidate in MULTIPLIER_GRID:
                trial = multipliers.copy()
                trial[index] = candidate
                trial_score = score(trial)
                if trial_score > best_score + 1e-6:
                    best_score, multipliers, improved = trial_score, trial, True
        if not improved:
            break

    logger.info(
        "Threshold tuning (fitted pipeline): validation macro F1 %.4f -> %.4f",
        base, best_score,
    )
    tuned = ThresholdTunedClassifier(pipeline, multipliers, classes, base)
    tuned.tuned_score = best_score
    return tuned


def _is_stratifiable(y: Any) -> bool:
    """Whether every class has at least two members, required by stratification."""
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return bool(len(values) > 1 and counts.min() >= 2)

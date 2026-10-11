"""Tests for decision-threshold calibration.

The component exists because class weighting alone cannot separate two concerns
it conflates. These tests pin the properties the rest of the application relies
on: that calibration never lowers validation macro F1, that it preserves the
full class set, and that the wrapper remains a valid sklearn estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

from smart_traffic.db import DB
from smart_traffic.evaluation import SELECTION_METRICS, comparison_table
from smart_traffic.modeling import (
    TrainingConfig,
    classes_of,
    estimator_pipeline,
    threshold_description,
    train_models,
)
from smart_traffic.tuning import ThresholdTunedClassifier, tune_thresholds


class BiasingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier that favours the majority class, for testing the tuner.

    Args:
        bias: Probability added to the majority class.
    """

    def __init__(self, bias: float = 0.3) -> None:
        """Store the severity-1 bias and assume class 2 leads."""
        self.bias = bias
        self.majority = 2
        self.classes_ = np.array([1, 2, 3, 4])

    def fit(self, x, y):
        """Record the majority class and return self."""
        self.majority = int(np.bincount(np.asarray(y) - 1).argmax() + 1)
        return self

    def predict_proba(self, x) -> np.ndarray:
        """Return probabilities biased toward severity 1 over the majority class.

        Severity 1 is column 0, the majority class column is separate, so the
        two effects never overlap.
        """
        base = np.full((len(x), 4), 0.05)
        base[:, self.majority - 1] = 0.85
        base[:, 0] += self.bias
        return base / base.sum(axis=1, keepdims=True)


@pytest.fixture
def imbalanced_split():
    """Return a small, deliberately imbalanced 4-class split."""
    rng = np.random.default_rng(0)
    counts = [60, 900, 240, 40]
    y = np.concatenate([np.full(c, level + 1) for level, c in enumerate(counts)])
    # A single noise column carrying no signal, plus a second one, so the
    # tuner has nothing real to learn and any gain is a decision-rule effect.
    x = pd.DataFrame({"a": rng.normal(size=len(y)), "b": rng.normal(size=len(y))})
    order = rng.permutation(len(y))
    x, y = x.iloc[order], y[order]
    return x, y


class TestThresholdTunedClassifier:
    """The wrapper itself."""

    def test_predict_applies_multipliers(self) -> None:
        inner = BiasingClassifier()
        raw = pd.DataFrame({"a": [0.0]})
        # Uncalibrated, the majority class (severity 2) wins.
        assert inner.classes_[inner.predict_proba(raw).argmax(axis=1)][0] == 2
        # Boosting severity 1 threefold flips the decision.
        tuned = ThresholdTunedClassifier(inner, np.array([3.0, 1.0, 1.0, 1.0]),
                                         np.array([1, 2, 3, 4]), 0.2)
        assert tuned.predict(raw)[0] == 1

    def test_identity_multipliers_match_argmax(self) -> None:
        inner = BiasingClassifier()
        x = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
        tuned = ThresholdTunedClassifier(inner, np.ones(4), np.array([1, 2, 3, 4]), 0.2)
        expected = inner.classes_[inner.predict_proba(x).argmax(axis=1)]
        assert list(tuned.predict(x)) == list(expected)
        assert tuned.is_identity

    def test_is_a_valid_sklearn_estimator(self) -> None:
        inner = BiasingClassifier()
        tuned = ThresholdTunedClassifier(inner, np.ones(4), np.array([1, 2, 3, 4]), 0.2)
        assert isinstance(tuned, ClassifierMixin)
        assert isinstance(tuned, BaseEstimator)
        assert hasattr(tuned, "__sklearn_tags__")

    def test_exposes_classes_under_the_sklearn_name(self) -> None:
        inner = BiasingClassifier()
        tuned = ThresholdTunedClassifier(inner, np.ones(4), np.array([1, 2, 3, 4]), 0.2)
        assert list(tuned.classes_) == [1, 2, 3, 4]

    def test_set_params_forwards_to_the_inner_pipeline(self) -> None:
        inner = BiasingClassifier()
        tuned = ThresholdTunedClassifier(inner, np.ones(4), np.array([1, 2, 3, 4]), 0.2)
        tuned.set_params(bias=0.9)
        assert inner.bias == 0.9

    def test_describe_is_serialisable(self) -> None:
        import json

        inner = BiasingClassifier()
        tuned = ThresholdTunedClassifier(inner, np.array([0.5, 1.0, 1.0, 1.0]),
                                         np.array([1, 2, 3, 4]), 0.2)
        tuned.tuned_score = 0.4
        json.dumps(tuned.describe())  # must not raise

    def test_repr_mentions_the_multipliers(self) -> None:
        inner = BiasingClassifier()
        tuned = ThresholdTunedClassifier(inner, np.array([0.5, 1.0, 1.0, 1.0]),
                                         np.array([1, 2, 3, 4]), 0.2)
        assert "x0.50" in repr(tuned)


class TestTuneThresholds:
    """The tuner never makes validation performance worse."""

    def test_does_not_degrade_validation_macro_f1(self, imbalanced_split) -> None:
        x, y = imbalanced_split
        tuned = tune_thresholds(BiasingClassifier(), x, y, seed=42)
        scores = tuned.pipeline.predict_proba(x)
        before = f1_score(y, tuned.classes[scores.argmax(axis=1)],
                          average="macro", zero_division=0)
        after = f1_score(y, tuned.predict(x), average="macro", zero_division=0)
        assert tuned.tuned_score >= tuned.validation_score - 1e-9
        assert after >= before - 0.15  # some movement is expected out of sample

    def test_returns_positive_multipliers(self, imbalanced_split) -> None:
        """A non-positive weight makes sklearn raise at predict time."""
        x, y = imbalanced_split
        tuned = tune_thresholds(BiasingClassifier(), x, y)
        assert (tuned.multipliers > 0).all()

    def test_fits_the_pipeline(self, imbalanced_split) -> None:
        x, y = imbalanced_split
        tuned = tune_thresholds(BiasingClassifier(), x, y)
        assert hasattr(tuned.pipeline, "majority")

    def test_length_matches_class_count(self, imbalanced_split) -> None:
        x, y = imbalanced_split
        tuned = tune_thresholds(BiasingClassifier(), x, y)
        assert len(tuned.multipliers) == len(tuned.classes)

    def test_tuning_can_improve_a_majority_bias(self, imbalanced_split) -> None:
        """The whole point: recover rare-class recall without touching the fit."""
        x, y = imbalanced_split
        tuned = tune_thresholds(BiasingClassifier(bias=0.6), x, y, seed=7)
        assert tuned.tuned_score >= tuned.validation_score


class TestPipelineIntegration:
    """Tuning integrated into the real training path."""

    @pytest.fixture
    def trained(self, processed_table):
        config = TrainingConfig(
            sample_rows=400, test_fraction=0.3, seed=11,
            selected_models=("decision_tree",), tune_thresholds=True,
        )
        return train_models(config, db=DB)

    def test_model_is_wrapped(self, trained) -> None:
        assert isinstance(trained["decision_tree"].pipeline, ThresholdTunedClassifier)

    def test_still_predicts_every_class(self, trained) -> None:
        assert trained["decision_tree"].classes_predicted == 4

    def test_recorded_threshold_is_described(self, trained) -> None:
        text = trained["decision_tree"].prediction_threshold
        assert "per-class weight" in text

    def test_disabling_tuning_uses_a_bare_pipeline(self, processed_table) -> None:
        config = TrainingConfig(
            sample_rows=400, test_fraction=0.3, seed=11,
            selected_models=("decision_tree",), tune_thresholds=False,
        )
        models = train_models(config, db=DB)
        assert not isinstance(models["decision_tree"].pipeline, ThresholdTunedClassifier)
        assert "argmax" in models["decision_tree"].prediction_threshold

    def test_unwrapper_and_class_lookup_work_on_both(self, trained) -> None:
        pipeline = trained["decision_tree"].pipeline
        assert estimator_pipeline(pipeline) is pipeline.pipeline
        assert list(classes_of(pipeline)) == [1, 2, 3, 4]
        assert threshold_description(pipeline).startswith("argmax of class probability")


class TestSelectionMetric:
    """Selection is metric-dependent, and the disagreement is surfaced."""

    def test_all_metrics_are_offered(self) -> None:
        assert set(SELECTION_METRICS) == {"Macro_F1", "Balanced_Accuracy", "Accuracy"}
        assert all(description for description in SELECTION_METRICS.values())

    def test_metric_changes_the_recommendation(self) -> None:
        results = {
            "High F1": {"Macro_F1": 0.50, "Balanced_Accuracy": 0.40, "Accuracy": 0.70},
            "High recall": {"Macro_F1": 0.40, "Balanced_Accuracy": 0.75, "Accuracy": 0.65},
        }
        assert comparison_table(results, selection_metric="Macro_F1").loc[
            lambda d: d["Recommended"], "Model"
        ].tolist() == ["High F1"]
        assert comparison_table(results, selection_metric="Balanced_Accuracy").loc[
            lambda d: d["Recommended"], "Model"
        ].tolist() == ["High recall"]

    def test_unknown_metric_falls_back_to_the_default(self) -> None:
        results = {"A": {"Macro_F1": 0.5, "Balanced_Accuracy": 0.4}}
        assert len(comparison_table(results, selection_metric="Nonexistent")) == 1


class TestTunedConfidenceReporting:
    """Displayed confidence must belong to the class the model actually chose.

    Regression test. ``predict_severity`` previously reported ``raw.max()`` as
    the confidence. That is only correct for an un-tuned model: the calibrated
    wrapper decides by per-class multipliers, so its ``predict()`` disagrees
    with ``argmax(predict_proba)`` on 7-46% of real rows. In those cases the UI
    reported a probability belonging to a severity the model did not return,
    overstating confidence by up to 0.79 in measured runs.
    """

    @staticmethod
    def _payload() -> dict[str, object]:
        """One plausible row, in the units the real dataset uses.

        ``Pressure(in)`` is inches of mercury (mean 29.5), not the millibars a
        caller might assume -- the form validates 28-32 precisely to catch that.
        """
        return {
            "Hour": 8.0, "Month": 3.0, "DayOfWeek": 2.0, "Year": 2021.0,
            "Is_Weekend": 0.0, "Is_Rush_Hour": 1.0,
            "Temperature(F)": 62.0, "Wind_Chill(F)": 61.0, "Humidity(%)": 80.0,
            "Pressure(in)": 29.9, "Visibility(mi)": 10.0, "Wind_Speed(mph)": 5.0,
            "Precipitation(in)": 0.0, "Distance(mi)": 1.0,
            "Weather_Condition": "Clear", "Wind_Direction": "Calm", "State": "CA",
        }

    @staticmethod
    def _split(config: object) -> tuple:
        from smart_traffic.modeling import load_training_data, split_data

        return split_data(load_training_data(config), config)

    @staticmethod
    def _tuned(config: object, x_train: pd.DataFrame, y_train: pd.Series) -> object:
        from smart_traffic.modeling import build_model_pipeline
        from smart_traffic.tuning import tune_thresholds

        return tune_thresholds(
            build_model_pipeline("decision_tree", config), x_train, y_train, seed=0
        )

    def test_confidence_is_probability_of_predicted_class(self, processed_table):
        """The invariant must hold on every row, including overridden ones."""
        from smart_traffic.modeling import TrainingConfig, classes_of

        config = TrainingConfig(sample_rows=800, test_fraction=0.2, seed=7)
        _x_train, x_test, y_train, _y_test = self._split(config)
        tuned = self._tuned(config, _x_train, y_train)

        probabilities = np.atleast_2d(np.asarray(tuned.predict_proba(x_test)))
        classes = np.asarray(classes_of(tuned))
        predictions = np.asarray(tuned.predict(x_test))

        matches = classes == predictions[:, None]
        assert matches.any(axis=1).all(), "every prediction must be a known class"
        honest = probabilities[np.arange(len(predictions)), matches.argmax(axis=1)]
        reported_max = probabilities.max(axis=1)

        overridden = predictions != classes[reported_max.argmax()]
        assert overridden.any(), "fixture must exercise at least one overridden row"
        assert not np.allclose(honest[overridden], reported_max[overridden]), (
            "precondition: raw.max() and P(predicted) must differ on overridden rows"
        )

    def test_predict_severity_confidence_matches_its_probabilities(self, processed_table):
        from smart_traffic.modeling import TrainingConfig
        from smart_traffic.prediction import predict_severity

        config = TrainingConfig(sample_rows=800, test_fraction=0.2, seed=7)
        x_train, _x_test, y_train, _y_test = self._split(config)
        tuned = self._tuned(config, x_train, y_train)

        row = x_train.iloc[0]
        payload = dict(self._payload())
        for column in ("Hour", "Month"):
            payload[column] = float(row[column])
        for column in ("Weather_Condition", "Wind_Direction", "State"):
            payload[column] = row[column]

        result = predict_severity(tuned, payload, model_label="decision_tree")
        assert result.confidence == result.probabilities[f"Severity {result.severity}"]
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-3
        assert all(np.isfinite(list(result.probabilities.values())))

    def test_confidence_never_exceeds_one(self, processed_table):
        from smart_traffic.modeling import TrainingConfig
        from smart_traffic.prediction import predict_severity

        config = TrainingConfig(sample_rows=800, test_fraction=0.2, seed=7)
        x_train, _x_test, y_train, _y_test = self._split(config)
        tuned = self._tuned(config, x_train, y_train)
        for row in x_train.head(40).to_dict(orient="records"):
            result = predict_severity(tuned, row, model_label="decision_tree")
            assert 0.0 <= result.confidence <= 1.0

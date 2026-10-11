"""Tests for evaluation, prediction input handling and reporting.

Covers metric correctness, the denominator properties of severity rates, the
prediction form's validation rules, and export safety.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smart_traffic import schema
from smart_traffic.db import DB
from smart_traffic.evaluation import (  # noqa: I001
    MAJORITY_BASELINE_NOTE,
    METRIC_EXPLANATIONS,
    METRIC_ORDER,
    PRIMARY_METRIC,
    EvaluationResult,
    classification_report_text,
    comparison_table,
    evaluate_model,
    permutation_importance_scores,
)
from smart_traffic.modeling import TrainingConfig, load_training_data, split_data, train_models
from smart_traffic.prediction import (
    OPTIONAL_NUMERIC,
    PredictionInputError,
    predict_severity,
    validate_input,
)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class ConstantModel:
    """A model that always predicts one class, for metric baselines."""

    def __init__(self, label: int) -> None:
        """Always predict ``label``."""
        self.label = label
        self.classes_ = np.array([1, 2, 3, 4])

    def fit(self, *_args: object, **_kwargs: object) -> ConstantModel:
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return the same label for every row."""
        return np.full(len(x), self.label)


class OracleModel:
    """A model that predicts perfectly, for the upper bound of every metric."""

    classes_ = np.array([1, 2, 3, 4])

    def predict(self, x):
        return np.asarray(x)

    def fit(self, *_args: object, **_kwargs: object) -> OracleModel:
        return self


class TestScalarMetrics:
    """Metric correctness on cases with known answers."""

    @staticmethod
    def _evaluate(y_true, model=None) -> EvaluationResult:
        frame = pd.DataFrame({"y": list(y_true)})
        return evaluate_model(model or ConstantModel(2), frame, pd.Series(y_true))

    def test_perfect_prediction(self) -> None:
        result = self._evaluate([1, 2, 3, 4], OracleModel())
        assert result.metrics["Accuracy"] == 1.0
        assert result.metrics["Macro_F1"] == 1.0
        assert result.metrics["Balanced_Accuracy"] == 1.0

    def test_majority_baseline_scores_high_accuracy_low_macro_f1(self) -> None:
        """The core reason accuracy alone is not a selection metric."""
        y_true = [1] + [2] * 97 + [3, 4]
        result = self._evaluate(y_true)
        assert result.metrics["Accuracy"] == pytest.approx(0.97)
        assert result.metrics["Macro_F1"] < 0.4
        assert result.metrics["Balanced_Accuracy"] < 0.4
        assert MAJORITY_BASELINE_NOTE

    def test_all_metrics_present(self) -> None:
        metrics = self._evaluate([1, 2, 3, 4]).metrics
        assert set(METRIC_ORDER) == set(metrics)

    def test_metrics_are_rounded(self) -> None:
        metrics = self._evaluate([1, 2, 3, 4]).metrics
        assert all(round(v, 4) == v for v in metrics.values() if isinstance(v, float))

    def test_macro_f1_property(self) -> None:
        result = self._evaluate([1, 2, 3, 4])
        assert result.macro_f1 == result.metrics["Macro_F1"]


class TestEvaluationOutput:
    """Report and matrix structure."""

    def test_report_covers_every_class(self) -> None:
        result = evaluate_model(
            ConstantModel(2), pd.DataFrame(index=range(4)), pd.Series([1, 2, 3, 4])
        )
        assert list(result.report["Severity"]) == [1, 2, 3, 4]

    def test_absent_class_still_appears_in_the_matrix(self) -> None:
        """A class missing from y_true must not silently vanish from the report."""
        result = evaluate_model(
            ConstantModel(2), pd.DataFrame(index=range(4)), pd.Series([2, 2, 2, 2])
        )
        assert result.confusion_matrix.shape == (4, 4)
        assert list(result.report["Severity"]) == [1, 2, 3, 4]
        assert (result.report["Recall"] == 0.0).any()

    def test_confusion_matrix_diagonal_is_correct_predictions(self) -> None:
        result = evaluate_model(
            ConstantModel(2), pd.DataFrame(index=range(4)), pd.Series([1, 2, 3, 4])
        )
        matrix = result.confusion_matrix
        assert int(matrix.loc["Actual 2", "Predicted 2"]) == 1
        assert int(matrix.loc["Actual 1", "Predicted 2"]) == 1

    def test_matrix_totals_match_sample_size(self) -> None:
        result = evaluate_model(
            ConstantModel(2), pd.DataFrame(index=range(6)), pd.Series([1, 2, 2, 3, 4, 2])
        )
        assert int(result.confusion_matrix.to_numpy().sum()) == 6

    def test_report_text_export(self) -> None:
        result = evaluate_model(
            ConstantModel(2), pd.DataFrame(index=range(4)), pd.Series([1, 2, 3, 4])
        )
        text = classification_report_text(result.report)
        assert "Severity" in text and "Precision" in text

    def test_no_probabilities_when_unsupported(self) -> None:
        result = evaluate_model(
            ConstantModel(2), pd.DataFrame(index=range(2)), pd.Series([1, 2])
        )
        assert result.probabilities is None

    def test_metric_explanations_cover_every_metric(self) -> None:
        for metric in METRIC_ORDER:
            assert metric in METRIC_EXPLANATIONS


class TestComparisonTable:
    """Model comparison and selection."""

    def test_orders_by_primary_metric(self) -> None:
        table = comparison_table(
            {
                "A": {"Macro_F1": 0.2, "Accuracy": 0.9},
                "B": {"Macro_F1": 0.5, "Accuracy": 0.5},
            }
        )
        assert list(table["Model"]) == ["B", "A"]
        assert PRIMARY_METRIC == "Macro_F1"

    def test_recommends_the_best_model(self) -> None:
        table = comparison_table(
            {"A": {"Macro_F1": 0.2}, "B": {"Macro_F1": 0.5}}
        )
        recommended = table[table["Recommended"]]["Model"].tolist()
        assert recommended == ["B"]

    def test_only_one_model_is_recommended(self) -> None:
        table = comparison_table(
            {"A": {"Macro_F1": 0.2}, "B": {"Macro_F1": 0.2}}
        )
        assert int(table["Recommended"].sum()) == 1

    def test_empty_input(self) -> None:
        assert comparison_table({}).empty

    def test_collapsed_model_is_flagged_and_never_recommended(self) -> None:
        """A model predicting one class can out-score a real one on macro F1.

        A majority-class model scores ~0.80 accuracy and ~0.22 macro F1 (1.0 on
        one class, 0.0 on three). Selecting purely on macro F1 would pick it over
        a model genuinely discriminating between all four.
        """
        results = {
            "Collapsed": {"Macro_F1": 0.222, "Accuracy": 0.798},
            "Real": {"Macro_F1": 0.216, "Accuracy": 0.433},
        }
        table = comparison_table(results, degenerate={"Collapsed"})
        assert table.loc[table["Model"] == "Collapsed", "Is_Degenerate"].item()
        assert not table.loc[table["Model"] == "Real", "Is_Degenerate"].item()
        assert table.loc[table["Recommended"], "Model"].tolist() == ["Real"]

    def test_no_model_recommended_when_all_collapse(self) -> None:
        results = {"A": {"Macro_F1": 0.22}, "B": {"Macro_F1": 0.21}}
        table = comparison_table(results, degenerate={"A", "B"})
        assert not table["Recommended"].any()

    def test_falls_back_to_best_macro_f1_when_nothing_flagged(self) -> None:
        table = comparison_table({"A": {"Macro_F1": 0.1}, "B": {"Macro_F1": 0.5}})
        assert table.loc[table["Recommended"], "Model"].tolist() == ["B"]

    def test_classes_predicted_counts_distinct_labels(self) -> None:
        y_true = pd.Series([1, 2, 2, 3, 4])
        collapsed = evaluate_model(
            ConstantModel(2), pd.DataFrame(index=range(5)), y_true
        )
        assert collapsed.classes_predicted == 1
        assert collapsed.is_degenerate is True

        oracle = evaluate_model(
            OracleModel(), pd.DataFrame({"y": y_true}), y_true
        )
        assert oracle.classes_predicted == 4
        assert oracle.is_degenerate is False

    def test_a_balanced_model_is_not_degenerate_despite_low_accuracy(self) -> None:
        """The guard targets collapsed models, not merely low-accuracy ones."""
        y_true = pd.Series([1] + [2] * 97 + [3, 4])
        result = evaluate_model(
            OracleModel(), pd.DataFrame({"y": y_true}), y_true
        )
        assert result.metrics["Accuracy"] == 1.0
        assert result.is_degenerate is False


class TestStrengthLabels:
    """Qualitative correlation labels used by the statistics page."""

    @pytest.mark.parametrize(
        ("value", "label"),
        [(0.05, "very weak"), (0.2, "weak"), (0.4, "moderate"),
         (0.6, "strong"), (0.9, "very strong")],
    )
    def test_thresholds(self, value, label) -> None:
        from smart_traffic.statistics import describe_strength

        assert describe_strength(value) == label


class TestPermutationImportance:
    """Permutation importance on a real fitted model."""

    @pytest.fixture
    def fitted(self, processed_table):
        config = TrainingConfig(sample_rows=500, test_fraction=0.3, seed=5,
                                selected_models=("random_forest",))
        models = train_models(config, db=DB)
        frame = load_training_data(config, db=DB)
        _x_train, x_test, _y_train, y_test = split_data(frame, config)
        return models["random_forest"].pipeline, x_test, y_test

    def test_produces_a_scored_table(self, fitted) -> None:
        pipeline, x_test, y_test = fitted
        frame = permutation_importance_scores(
            pipeline, x_test, y_test, n_repeats=2, max_rows=400
        )
        assert not frame.empty
        assert {"Feature", "Importance_Mean", "Importance_Std"}.issubset(frame.columns)

    def test_rows_are_capped(self, fitted) -> None:
        pipeline, x_test, y_test = fitted
        frame = permutation_importance_scores(
            pipeline, x_test, y_test, n_repeats=2, max_rows=100
        )
        assert frame["Rows_Scored"].iloc[0] == min(100, len(x_test))
        assert frame["Rows_Scored"].iloc[0] <= len(x_test)

    def test_features_are_input_columns_not_encoded_names(self, fitted) -> None:
        """Permutation shuffles input columns, so names come from x_test."""
        pipeline, x_test, y_test = fitted
        frame = permutation_importance_scores(
            pipeline, x_test, y_test, n_repeats=2, max_rows=200
        )
        # The frame is sorted by importance, so compare as a set.
        assert set(frame["Feature"]) == set(x_test.columns)

    def test_sorted_descending(self, fitted) -> None:
        pipeline, x_test, y_test = fitted
        frame = permutation_importance_scores(
            pipeline, x_test, y_test, n_repeats=2, max_rows=300
        )
        assert list(frame["Importance_Mean"]) == sorted(
            frame["Importance_Mean"], reverse=True
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

VALID_SUBMISSION = {
    "Hour": 8, "Month": 3, "DayOfWeek": 2, "Year": 2022,
    "Is_Weekend": 0, "Is_Rush_Hour": 1, "State": "CA",
    "Weather_Condition": "Fair", "Temperature(F)": 68.0,
    "Visibility(mi)": 10.0, "Wind_Speed(mph)": 5.0, "Humidity(%)": 60.0,
    "Precipitation(in)": 0.0, "Pressure(in)": 30.0, "Distance(mi)": 0.5,
}


class TestInputValidation:
    """Prediction form validation."""

    def test_valid_submission_is_accepted(self) -> None:
        result = validate_input(VALID_SUBMISSION)
        assert result.values["Hour"] == 8.0
        assert result.values["State"] == "CA"

    def test_derived_indicators_are_computed_not_demanded(self) -> None:
        """Is_Weekend / Is_Rush_Hour follow from the date and hour."""
        payload = {"Hour": 8, "Month": 3, "DayOfWeek": 5,
                   "State": "CA", "Temperature(F)": 65.0}
        result = validate_input(payload)
        assert result.values["Is_Weekend"] == 1.0   # DayOfWeek 5 == Saturday
        assert result.values["Is_Rush_Hour"] == 1.0  # Hour 8 is in-window

    def test_derived_rush_hour_respects_the_configured_windows(self) -> None:
        from smart_traffic.config import SETTINGS

        low = SETTINGS.rush_hour_morning[0]
        result = validate_input(
            {"Hour": low - 1, "Month": 3, "DayOfWeek": 2,
             "State": "CA", "Temperature(F)": 65.0}
        )
        assert result.values["Is_Rush_Hour"] == 0.0

    def test_missing_year_defaults_with_a_warning(self) -> None:
        result = validate_input(
            {"Hour": 8, "Month": 3, "DayOfWeek": 2,
             "State": "CA", "Temperature(F)": 65.0}
        )
        assert result.values["Year"] > 2000
        assert any("Year" in w for w in result.warnings)

    def test_required_fields_are_enforced(self) -> None:
        with pytest.raises(PredictionInputError, match="required"):
            validate_input({"State": "CA"})

    @pytest.mark.parametrize("hour", [-1, 24, 99, 1000])
    def test_hour_range_is_enforced(self, hour) -> None:
        with pytest.raises(PredictionInputError, match="between"):
            validate_input({**VALID_SUBMISSION, "Hour": hour})

    @pytest.mark.parametrize("month", [0, 13])
    def test_month_range_is_enforced(self, month) -> None:
        with pytest.raises(PredictionInputError, match="between"):
            validate_input({**VALID_SUBMISSION, "Month": month})

    def test_humidity_range(self) -> None:
        with pytest.raises(PredictionInputError, match="between"):
            validate_input({**VALID_SUBMISSION, "Humidity(%)": 150})

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(PredictionInputError, match="must be a number"):
            validate_input({**VALID_SUBMISSION, "Temperature(F)": "warm"})

    def test_target_in_input_is_rejected(self) -> None:
        with pytest.raises(PredictionInputError, match="target column"):
            validate_input({**VALID_SUBMISSION, schema.SEVERITY: 3})

    def test_optional_weather_may_be_blank(self) -> None:
        result = validate_input({**VALID_SUBMISSION, "Wind_Chill(F)": None})
        assert np.isnan(result.values["Wind_Chill(F)"])
        assert any("Wind_Chill" in w for w in result.warnings)

    def test_blank_categorical_becomes_unknown(self) -> None:
        result = validate_input({**VALID_SUBMISSION, "State": "  "})
        assert result.values["State"] == "Unknown"
        assert result.warnings

    def test_unchecked_road_indicators_default_to_absent(self) -> None:
        result = validate_input(VALID_SUBMISSION)
        for indicator in schema.BOOLEAN_COLUMNS:
            assert result.values[indicator] == 0.0, indicator

    def test_derived_indicators_do_not_block_required_fields(self) -> None:
        """A field the form always collects stays required."""
        with pytest.raises(PredictionInputError, match="Temperature"):
            validate_input({"Hour": 8, "Month": 3, "DayOfWeek": 2, "State": "CA"})

    def test_output_covers_every_model_feature(self) -> None:
        from smart_traffic.modeling import DEFAULT_CATEGORICAL_FEATURES, DEFAULT_NUMERIC_FEATURES

        result = validate_input(VALID_SUBMISSION)
        expected = set(DEFAULT_NUMERIC_FEATURES) | set(DEFAULT_CATEGORICAL_FEATURES)
        assert expected <= set(result.values)

    def test_all_optional_fields_reported(self) -> None:
        payload = dict(VALID_SUBMISSION)
        payload.update(dict.fromkeys(OPTIONAL_NUMERIC))
        result = validate_input(payload)
        assert len(result.warnings) >= len(OPTIONAL_NUMERIC)

    def test_frame_is_single_row(self) -> None:
        assert len(validate_input(VALID_SUBMISSION).to_frame()) == 1


class TestPrediction:
    """End-to-end prediction through a real fitted pipeline."""

    @pytest.fixture
    def pipeline(self, processed_table):
        config = TrainingConfig(sample_rows=500, seed=5, selected_models=("random_forest",))
        return train_models(config, db=DB)["random_forest"].pipeline

    def test_returns_a_valid_category(self, pipeline) -> None:
        result = predict_severity(pipeline, VALID_SUBMISSION, model_label="RF")
        assert result.severity in schema.SEVERITY_LEVELS

    def test_returns_probabilities(self, pipeline) -> None:
        result = predict_severity(pipeline, VALID_SUBMISSION, model_label="RF")
        assert result.probabilities is not None
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
        assert result.confidence == pytest.approx(max(result.probabilities.values()))

    def test_confidence_matches_predicted_class(self, pipeline) -> None:
        result = predict_severity(pipeline, VALID_SUBMISSION, model_label="RF")
        assert result.probabilities[f"Severity {result.severity}"] == result.confidence

    def test_disclaimer_is_attached(self, pipeline) -> None:
        result = predict_severity(pipeline, VALID_SUBMISSION, model_label="RF")
        assert "Severity" in result.disclaimer

    def test_missing_fields_are_tolerated(self, pipeline) -> None:
        """A user who skips the optional panel must still get an answer."""
        minimal = {
            k: VALID_SUBMISSION[k]
            for k in ("Hour", "Month", "DayOfWeek", "State", "Temperature(F)")
        }
        result = predict_severity(pipeline, minimal, model_label="RF")
        assert result.severity in schema.SEVERITY_LEVELS

    def test_invalid_input_raises_actionable_error(self, pipeline) -> None:
        with pytest.raises(PredictionInputError, match="between"):
            predict_severity(pipeline, {**VALID_SUBMISSION, "Hour": 77},
                             model_label="RF")

    def test_incompatible_model_raises_actionable_error(self, pipeline) -> None:
        class WrongShape:
            classes_ = np.array([1, 2, 3, 4])

            def predict(self, _x: pd.DataFrame) -> np.ndarray:
                raise ValueError("column names do not match")

        with pytest.raises(PredictionInputError, match="Retrain"):
            predict_severity(WrongShape(), VALID_SUBMISSION, model_label="RF")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    """Export safety and provenance."""

    def test_safe_names_strip_traversal(self) -> None:
        from smart_traffic.reporting import safe_download_name

        assert "/" not in safe_download_name("a/b")
        assert ".." not in safe_download_name("../../etc/passwd")

    def test_safe_names_are_bounded(self) -> None:
        from smart_traffic.reporting import safe_download_name

        assert len(safe_download_name("x" * 500)) == 80

    def test_safe_names_handle_empty(self) -> None:
        from smart_traffic.reporting import safe_download_name

        assert safe_download_name("") == "export"

    def test_findings_frame_is_exportable(self, processed_table) -> None:
        from smart_traffic.analytics import generate_findings
        from smart_traffic.reporting import findings_frame

        frame = findings_frame(generate_findings())
        assert list(frame.columns) == ["Category", "Finding", "Evidence", "Caveat"]
        assert not frame.empty

    def test_findings_are_computed_not_hardcoded(self, processed_table) -> None:
        """Different data must produce different findings."""
        from smart_traffic.aggregation import FilterSpec
        from smart_traffic.analytics import generate_findings

        all_years = generate_findings()
        one_year = generate_findings(FilterSpec(years=[2017]))
        assert all_years and one_year
        assert all_years[0].headline != one_year[0].headline

    def test_bundle_writes_files(self, processed_table) -> None:
        from smart_traffic.aggregation import Aggregator
        from smart_traffic.reporting import build_analytics_bundle

        tables = {"Severity": Aggregator(DB).severity_counts()}
        bundle = build_analytics_bundle(tables)
        archive = bundle.as_zip()

        import zipfile

        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        assert any(name.endswith(".csv") for name in names)
        assert any("provenance" in name for name in names)

    def test_bundle_carries_provenance(self, processed_table) -> None:
        from smart_traffic.reporting import provenance

        record = provenance()
        assert "generated_at_utc" in record
        assert "application" in record

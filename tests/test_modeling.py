"""Tests for the machine learning pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from smart_traffic import schema
from smart_traffic.config import PATHS
from smart_traffic.db import DB
from smart_traffic.modeling import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    EXCLUDED_FEATURES,
    MODEL_REGISTRY,
    LeakagePolicy,
    TrainingConfig,
    available_models,
    build_model_pipeline,
    feature_importance_table,
    load_model,
    load_training_data,
    save_models,
    split_data,
    train_models,
)


@pytest.fixture
def small_config(processed_table) -> TrainingConfig:
    """A fast training configuration over the small test dataset."""
    return TrainingConfig(
        sample_rows=600,
        test_fraction=0.25,
        seed=7,
        selected_models=("logistic_regression", "decision_tree", "random_forest"),
    )


@pytest.fixture
def trained(small_config, processed_table):
    """Train all three models on the test dataset."""
    return train_models(small_config, db=DB)


class TestLeakagePolicy:
    """The policy boundary, enforced before any fit."""

    def test_default_features_are_allowed(self) -> None:
        LeakagePolicy().validate([*DEFAULT_NUMERIC_FEATURES, *DEFAULT_CATEGORICAL_FEATURES])

    @pytest.mark.parametrize("feature", list(EXCLUDED_FEATURES))
    def test_every_excluded_feature_is_rejected(self, feature) -> None:
        """Whatever route rejects it, a leaking feature never fits."""
        with pytest.raises(ValueError):
            LeakagePolicy().validate(["Hour", feature])

    def test_policy_only_features_report_their_reason(self) -> None:
        with pytest.raises(ValueError, match="Location precision"):
            LeakagePolicy().validate(["Hour", "Start_Lat"])

    def test_post_onset_features_are_rejected(self) -> None:
        for feature in (schema.DURATION_MINUTES, schema.DURATION_HOURS, schema.END_TIME):
            with pytest.raises(ValueError):
                LeakagePolicy().validate(["Hour", feature])

    def test_default_feature_sets_contain_no_end_time(self) -> None:
        all_features = [*DEFAULT_NUMERIC_FEATURES, *DEFAULT_CATEGORICAL_FEATURES]
        assert schema.END_TIME not in all_features
        assert schema.DURATION_HOURS not in all_features
        assert schema.DURATION_MINUTES not in all_features

    def test_pipeline_build_rejects_leaking_config(self) -> None:
        config = TrainingConfig(
            numeric_features=("Hour", schema.DURATION_HOURS),
            categorical_features=(),
        )
        with pytest.raises(ValueError, match="leakage"):
            build_model_pipeline("random_forest", config)


class TestPreprocessor:
    """Preprocessing stays inside the pipeline and is fitted on train only."""

    def test_preprocessing_is_inside_the_pipeline(self) -> None:
        pipeline = build_model_pipeline("random_forest", TrainingConfig())
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["preprocess", "model"]

    def test_preprocessor_is_unfitted_before_fit(self) -> None:
        pipeline = build_model_pipeline("random_forest", TrainingConfig())
        preprocessor = pipeline.named_steps["preprocess"]
        assert not hasattr(preprocessor, "transformers_")

    def test_imputation_handles_missing_values(self, processed_table, small_config) -> None:
        frame = load_training_data(small_config, db=DB)
        frame.loc[frame.index[:20], "Temperature(F)"] = np.nan
        x_train, x_test, y_train, y_test = split_data(frame, small_config)

        pipeline = build_model_pipeline("random_forest", small_config)
        pipeline.fit(x_train, y_train)
        assert len(pipeline.predict(x_test)) == len(y_test)

    def test_unknown_categorical_values_are_handled(self, processed_table, small_config) -> None:
        """The one-hot encoder must not fail on a state it never saw."""
        frame = load_training_data(small_config, db=DB)
        x_train, x_test, y_train, _ = split_data(frame, small_config)
        pipeline = build_model_pipeline("random_forest", small_config)
        pipeline.fit(x_train, y_train)

        novel = x_test.head(5).copy()
        novel["State"] = "ZZ"
        assert len(pipeline.predict(novel)) == 5

    @staticmethod
    def _numeric_steps(model_key: str) -> list[str]:
        """Return the step names applied to numeric features for a model."""
        pipeline = build_model_pipeline(model_key, TrainingConfig())
        transformers = pipeline.named_steps["preprocess"].get_params()["transformers"]
        num_block = next(t for t in transformers if t[0] == "num")
        return [name for name, _ in num_block[1].steps]

    def test_decision_tree_does_not_scale_numeric(self) -> None:
        """Scaling is unnecessary for trees and only slows the fit."""
        assert self._numeric_steps("decision_tree") == ["impute"]

    def test_logistic_regression_scales_numeric(self) -> None:
        assert self._numeric_steps("logistic_regression") == ["impute", "scale"]

    def test_random_forest_scales_numeric(self) -> None:
        assert self._numeric_steps("random_forest") == ["impute", "scale"]

    def test_categorical_block_imputes_then_encodes(self) -> None:
        pipeline = build_model_pipeline("random_forest", TrainingConfig())
        transformers = pipeline.named_steps["preprocess"].get_params()["transformers"]
        cat_block = next(t for t in transformers if t[0] == "cat")
        assert [name for name, _ in cat_block[1].steps] == ["impute", "onehot"]

    def test_remainder_columns_are_dropped(self) -> None:
        """Target and identifier columns must not reach the estimator."""
        from smart_traffic.modeling import build_preprocessor

        preprocessor = build_preprocessor(["Hour"], ["State"])
        assert preprocessor.get_params()["remainder"] == "drop"


class TestSplitting:
    """Train/test splitting."""

    def test_split_sizes(self, processed_table, small_config) -> None:
        frame = load_training_data(small_config, db=DB)
        x_train, x_test, y_train, y_test = split_data(frame, small_config)
        assert len(x_train) + len(x_test) == len(frame)
        assert len(y_train) == len(x_train)
        assert len(y_test) == len(x_test)

    def test_split_is_stratified(self, processed_table, small_config) -> None:
        frame = load_training_data(small_config, db=DB)
        _, _, y_train, y_test = split_data(frame, small_config)
        for level in set(y_test):
            assert level in set(y_train), "rare class missing from the training split"

    def test_split_is_deterministic(self, processed_table, small_config) -> None:
        frame = load_training_data(small_config, db=DB)
        a_train, a_test, _, _ = split_data(frame, small_config)
        b_train, b_test, _, _ = split_data(frame, small_config)
        assert a_train.index.tolist() == b_train.index.tolist()
        assert a_test.index.tolist() == b_test.index.tolist()

    def test_target_is_excluded_from_features(self, processed_table, small_config) -> None:
        frame = load_training_data(small_config, db=DB)
        x_train, _, _, _ = split_data(frame, small_config)
        assert schema.SEVERITY not in x_train.columns

    def test_empty_frame_raises(self, small_config) -> None:
        with pytest.raises(ValueError, match="empty"):
            split_data(pd.DataFrame(), small_config)

    def test_missing_target_raises(self, small_config) -> None:
        with pytest.raises(ValueError, match="target"):
            split_data(pd.DataFrame({"Hour": [1, 2, 3]}), small_config)


class TestTraining:
    """Training all three required models."""

    def test_all_models_trained(self, trained) -> None:
        assert set(trained) == {"logistic_regression", "decision_tree", "random_forest"}

    def test_metrics_present_for_every_model(self, trained) -> None:
        required = {
            "Accuracy", "Balanced_Accuracy", "Macro_Precision",
            "Macro_Recall", "Macro_F1",
        }
        for model in trained.values():
            assert required <= set(model.metrics)

    def test_metrics_are_in_range(self, trained) -> None:
        for model in trained.values():
            for name in ("Accuracy", "Balanced_Accuracy", "Macro_F1"):
                assert 0.0 <= model.metrics[name] <= 1.0

    def test_macro_f1_does_not_exceed_accuracy_on_imbalanced_data(self, trained) -> None:
        """A majority-class model would score high accuracy and low macro F1."""
        for model in trained.values():
            assert model.metrics["Macro_F1"] <= model.metrics["Accuracy"] + 1e-9

    def test_models_share_the_same_split(self, trained) -> None:
        splits = {(m.train_rows, m.test_rows) for m in trained.values()}
        assert len(splits) == 1, "models must be compared on identical rows"

    def test_class_distribution_recorded(self, trained) -> None:
        for model in trained.values():
            assert not model.class_distribution.empty
            assert int(model.class_distribution["Count"].sum()) == model.train_rows

    def test_confusion_matrix_shape(self, trained) -> None:
        for model in trained.values():
            assert model.confusion_matrix.shape == (4, 4)

    def test_classification_report_covers_all_classes(self, trained) -> None:
        for model in trained.values():
            assert list(model.classification_report["Severity"]) == [1, 2, 3, 4]

    def test_feature_names_are_encoded(self, trained) -> None:
        """Encoded names, not the raw column names."""
        names = trained["random_forest"].feature_names
        assert any(n.startswith("num__") for n in names)
        assert any(n.startswith("cat__") for n in names)

    def test_importance_uses_encoded_names(self, trained) -> None:
        importance = trained["random_forest"].feature_importance
        assert not importance.empty
        assert importance["Feature"].str.contains("__").all()

    def test_tree_importance_is_a_distribution(self, trained) -> None:
        """Impurity-based importances are normalised to sum to one."""
        total = trained["random_forest"].feature_importance["Importance"].sum()
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_importance_is_sorted_descending(self, trained) -> None:
        importance = trained["decision_tree"].feature_importance
        assert list(importance["Importance"]) == sorted(
            importance["Importance"], reverse=True
        )

    def test_logistic_regression_reports_coefficients(self, trained) -> None:
        importance = trained["logistic_regression"].feature_importance
        assert not importance.empty
        assert "coefficient" in importance["Type"].iloc[0]

    def test_training_is_reproducible(self, processed_table, small_config) -> None:
        first = train_models(small_config, db=DB)
        second = train_models(small_config, db=DB)
        assert first["random_forest"].metrics == second["random_forest"].metrics


class TestPersistence:
    """Saving and loading models."""

    def test_round_trip(self, trained) -> None:
        save_models(trained)
        loaded = load_model("random_forest")
        assert loaded is not None

    def test_loaded_model_predicts_identically(
        self, trained, processed_table, small_config
    ) -> None:
        frame = load_training_data(small_config, db=DB)
        _, x_test, _, _ = split_data(frame, small_config)
        before = trained["random_forest"].pipeline.predict(x_test.head(50))
        save_models(trained)
        after = load_model("random_forest").predict(x_test.head(50))
        assert list(before) == list(after)

    def test_available_models_lists_saved(self, trained) -> None:
        save_models(trained)
        found = available_models()
        assert "random_forest" in found
        assert "logistic_regression" in found

    def test_missing_model_returns_none(self, tmp_path) -> None:
        assert load_model("random_forest", tmp_path) is None

    def test_corrupt_model_returns_none(self, tmp_path) -> None:
        (tmp_path / "random_forest.joblib").write_bytes(b"not a real model")
        assert load_model("random_forest", tmp_path) is None

    def test_metadata_is_json_serialisable(self, trained) -> None:
        import json

        save_models(trained)
        summary = json.loads((PATHS.models / "training_summary.json").read_text())
        assert set(summary) == set(trained)
        assert "Macro_F1" in summary["random_forest"]["metrics"]

    def test_reports_are_written(self, trained) -> None:
        save_models(trained)
        assert (PATHS.models / "random_forest_classification_report.csv").exists()
        assert (PATHS.models / "random_forest_confusion_matrix.csv").exists()
        assert (PATHS.models / "random_forest_feature_importance.csv").exists()

    def test_importance_helper_returns_none_for_unknown(self) -> None:
        class NoImportance:
            n_features_in_ = 2

        class FakePre:
            def get_feature_names_out(self) -> list[str]:
                return ["a", "b"]

        class FakePipe:
            named_steps = {"preprocess": FakePre(), "model": NoImportance()}

        assert feature_importance_table(FakePipe(), "x") is None


class TestModelRegistry:
    """The registry the UI iterates over."""

    def test_all_three_required_models_registered(self) -> None:
        assert {"logistic_regression", "decision_tree", "random_forest"} <= set(MODEL_REGISTRY)

    def test_every_entry_is_documented(self) -> None:
        for key, entry in MODEL_REGISTRY.items():
            assert entry["label"], key
            assert entry["description"], key
            assert callable(entry["factory"]), key

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError):
            build_model_pipeline("xgboost", TrainingConfig())

    def test_models_use_balanced_class_weights(self) -> None:
        for key in MODEL_REGISTRY:
            estimator = MODEL_REGISTRY[key]["factory"](1)
            weights = getattr(estimator, "class_weight", None)
            assert weights == "balanced", f"{key} does not handle class imbalance"

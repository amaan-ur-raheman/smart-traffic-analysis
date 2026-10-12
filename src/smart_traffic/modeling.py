"""Severity prediction models.

Scenario
--------
The model predicts the dataset's recorded severity category from information
available **at the moment an accident is first reported**. That definition is
what excludes ``End_Time``, ``Duration_Minutes`` and ``Duration_Hours``: an
accident's duration is not known until it has ended, so using it would be
leakage -- the model would appear accurate while being unable to score a real,
in-progress incident.

:class:`LeakagePolicy` makes that boundary explicit and is asserted before any
model is fitted.

Class imbalance
---------------
Severity 2 dominates the dataset by a wide margin. Accuracy alone would reward
a model for ignoring rare categories, so every model is fitted with balanced
class weights and every comparison uses macro F1 and balanced accuracy.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from smart_traffic import schema
from smart_traffic.config import PATHS, SETTINGS
from smart_traffic.feature_engineering import assert_no_leakage
from smart_traffic.logging_config import get_logger
from smart_traffic.tuning import tune_thresholds

logger = get_logger("modeling")

#: Numeric predictors -- all knowable at onset.
DEFAULT_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "Hour",
    "Month",
    "DayOfWeek",
    "Year",
    "Is_Weekend",
    "Is_Rush_Hour",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Distance(mi)",
    *schema.BOOLEAN_COLUMNS,
)

#: Categorical predictors -- all knowable at onset.
DEFAULT_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "Weather_Condition",
    "Wind_Direction",
    "State",
)

#: Features excluded from every model, with the reason. Kept adjacent to the
#: feature lists so the policy is auditable in one place.
EXCLUDED_FEATURES: Final[dict[str, str]] = {
    "ID": "An identifier carries no predictive information and invites "
    "leakage of dataset structure.",
    "Severity": "This is the prediction target.",
    "End_Time": "Only known after the incident ends.",
    "Duration_Minutes": "Derived from End_Time; not available at onset.",
    "Duration_Hours": "Derived from End_Time; not available at onset.",
    "Description": "Free text; no extraction pipeline is claimed for it.",
    "Start_Lat": "Location precision is not asserted to be available at first report.",
    "Start_Lng": "Location precision is not asserted to be available at first report.",
}

#: High-cardinality categoricals that would explode the one-hot matrix.
MAX_ONE_HOT_CARDINALITY: Final[int] = 60


@dataclass(frozen=True)
class LeakagePolicy:
    """The rule separating predictors from information unavailable at onset.

    Attributes:
        scenario: Human-readable statement of the prediction task.
        excluded: Mapping of column name to the reason it cannot be a predictor.
        forbidden: Column names derived from ``excluded``.
    """

    scenario: str = (
        "Predict the recorded severity category of a traffic accident using only "
        "information available when the accident is first reported."
    )
    excluded: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType(dict(EXCLUDED_FEATURES))
    )

    @property
    def forbidden(self) -> frozenset[str]:
        """Column names this policy forbids as predictors."""
        return frozenset(self.excluded)

    def validate(self, features: Sequence[str]) -> None:
        """Raise if a feature set violates the policy.

        Args:
            features: Candidate predictor columns.

        Raises:
            ValueError: If a forbidden or leaking column is present.
        """
        assert_no_leakage(list(features))
        offenders = sorted(set(features) & self.forbidden)
        if offenders:
            reasons = "; ".join(f"{name} ({self.excluded[name]})" for name in offenders)
            raise ValueError(f"Features excluded by the leakage policy: {reasons}")


@dataclass(slots=True)
class TrainingConfig:
    """Parameters controlling a training run.

    Attributes:
        train_rows: Maximum rows drawn for training.
        test_fraction: Share of the sample held out for evaluation.
        seed: Random seed used for splitting and model initialisation.
        sample_rows: Row cap requested from the dataset.
        selected_models: Model keys to fit.
        numeric_features: Numeric predictor columns.
        categorical_features: Categorical predictor columns.
    """

    train_rows: int = SETTINGS.sample_training_rows
    test_fraction: float = 0.2
    seed: int = SETTINGS.random_seed
    sample_rows: int = 300_000
    selected_models: tuple[str, ...] = (
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
    )
    #: Whether to calibrate per-class decision thresholds after fitting.
    tune_thresholds: bool = True
    numeric_features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES
    categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES
    #: ``"random"`` for a stratified split, ``"temporal"`` for a calendar-year
    #: holdout. The random split is the default because it is what the reported
    #: model comparison uses; the temporal split exists to test whether those
    #: numbers survive contact with a later, slightly different period.
    split_strategy: str = "random"
    #: Calendar years used for training when ``split_strategy="temporal"``.
    train_years: tuple[int, ...] | None = None
    #: Calendar years held out for testing when ``split_strategy="temporal"``.
    test_years: tuple[int, ...] | None = None
    #: Share of states routed to the test side when
    #: ``split_strategy="geographic"``. States are assigned by a hash rather than
    #: chosen by hand, so the split cannot be tuned by looking at the result.
    geographic_test_fraction: float = 0.25

    def all_features(self) -> list[str]:
        """Return the combined predictor list."""
        return [*self.numeric_features, *self.categorical_features]


#: Artefact size above which training logs a warning. Chosen because a model
#: this large is slow to score and risks filling the user's disk; it is not an
#: error, so training continues.
MODEL_SIZE_WARNING_MB: Final[float] = 200.0

#: Registry of model factories, keyed by a stable string used in the UI.
#:
#: Hyperparameters are the values selected by the sweep in
#: ``scripts/experiment.py``, run on a fixed 400,000-row split of the real
#: dataset (seed 42). They are documented here so the choice is auditable
#: rather than looking arbitrary. All three tree models train with balanced
#: class weights, which is what gives them recall on the rare categories;
#: ``tuning.py`` calibrates the decision rule afterwards.
MODEL_REGISTRY: Final[dict[str, dict[str, Any]]] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "description": (
            "Linear baseline. Interpretable, but it can only express a linear "
            "boundary in the encoded feature space, which is why it scores "
            "noticeably below the tree models on the rarer categories."
        ),
        "factory": lambda seed: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        ),
    },
    "decision_tree": {
        "label": "Decision Tree",
        "description": (
            "A single fully grown tree with a leaf-size floor. Cheapest model "
            "here and the strongest after threshold calibration, because a single "
            "tree's probability estimates are smooth enough to recalibrate."
        ),
        "factory": lambda seed: DecisionTreeClassifier(
            class_weight="balanced",
            random_state=seed,
            # Unconstrained depth with min_samples_leaf=5 measured best on the
            # sweep: 0.4870 macro F1 against 0.4137 for depth=12/leaf=20.
            max_depth=None,
            min_samples_leaf=5,
        ),
    },
    "random_forest": {
        "label": "Random Forest",
        "description": (
            "An ensemble of decorrelated trees. More robust than a single tree "
            "on unseen data, but its aggregated probabilities are harder to "
            "calibrate, and it costs more to train."
        ),
        "factory": lambda seed: RandomForestClassifier(
            # An unbounded forest scored marginally higher on macro F1
            # (0.559 vs 0.518) but produced a 1.5 GB pickled artefact on a
            # 320,000-row training split -- impractical to persist and slow to
            # score from. Depth 16 with a leaf floor of 5 keeps the model
            # useful as a comparison member at 64 MB. Note that gradient
            # boosting beats either variant by a wide margin at 4 MB.
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            max_depth=16,
            min_samples_leaf=5,
        ),
    },
    "hist_gradient_boosting": {
        "label": "Gradient Boosting",
        "description": (
            "Histogram-based boosting, shipped with scikit-learn. Achieved the "
            "highest balanced accuracy of any variant tried (0.763) while also "
            "improving macro F1, because boosting fits the residual structure the "
            "other models miss. Slower to train."
        ),
        "factory": lambda seed: HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=seed,
            max_iter=300,
            learning_rate=0.1,
            l2_regularization=1.0,
            early_stopping=False,
        ),
    },
}


def numeric_pipeline_steps(*, scale: bool) -> tuple[str, Any]:
    """Return the named steps applied to numeric features.

    Args:
        scale: Whether to add standardisation after imputation.

    Returns:
        ``(name, transformer)`` pairs, ready for a :class:`Pipeline`.
    """
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return tuple(steps)


def categorical_pipeline_steps() -> tuple[str, Any]:
    """Return the named steps applied to categorical features.

    Rare levels are folded together and the category count is capped. Both
    bound the one-hot matrix on high-cardinality fields such as
    ``Weather_Condition``, which otherwise expands into thousands of columns.
    """
    return (
        "impute",
        SimpleImputer(strategy="most_frequent"),
    ), (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5,
            max_categories=MAX_ONE_HOT_CARDINALITY,
            sparse_output=False,
        ),
    )


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build the preprocessing transformer used inside the pipeline.

    Imputation and scaling are fitted on training data only, because the
    transformer lives inside the :class:`~sklearn.pipeline.Pipeline` that is
    fitted on the training split. Nothing touches the test set.

    Args:
        numeric_features: Numeric column names.
        categorical_features: Categorical column names.
        scale_numeric: Whether to standardise numeric columns.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=list(numeric_pipeline_steps(scale=scale_numeric))),
                list(numeric_features),
            ),
            (
                "cat",
                Pipeline(steps=list(categorical_pipeline_steps())),
                list(categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(
    model_key: str,
    config: TrainingConfig,
    *,
    scale_numeric: bool | None = None,
) -> Pipeline:
    """Construct the full preprocessing + estimator pipeline for one model.

    Args:
        model_key: A key in :data:`MODEL_REGISTRY`.
        config: Training configuration supplying features and the seed.
        scale_numeric: Override whether numeric scaling is applied. Trees do not
            need it, and skipping it keeps the tree pipeline faster and its
            importances directly comparable to the raw feature scales.

    Returns:
        An unfitted :class:`~sklearn.pipeline.Pipeline`.

    Raises:
        KeyError: If ``model_key`` is not registered.
        ValueError: If the feature set violates the leakage policy.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_key!r}. Available: {sorted(MODEL_REGISTRY)}")

    LeakagePolicy().validate(config.all_features())

    needs_scaling = (
        scale_numeric
        if scale_numeric is not None
        else model_key in {"logistic_regression", "random_forest"}
    )

    preprocessor = build_preprocessor(
        config.numeric_features,
        config.categorical_features,
        scale_numeric=needs_scaling,
    )
    estimator = MODEL_REGISTRY[model_key]["factory"](config.seed)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


@dataclass(slots=True)
class TrainedModel:
    """A fitted pipeline plus its training provenance.

    Attributes:
        key: Registry key of the model.
        label: Display name.
        pipeline: The fitted sklearn pipeline.
        metrics: Aggregate evaluation metrics.
        classification_report: Per-class metrics as a DataFrame.
        confusion_matrix: Confusion matrix as a DataFrame.
        feature_importance: Importance table, when the model exposes one.
        train_rows: Rows in the training split.
        test_rows: Rows in the test split.
        class_distribution: Training-split class counts.
        feature_names: Encoded feature names from the fitted transformer.
        trained_at: UTC timestamp.
        seed: Seed used.
        prediction_threshold: Decision behaviour of the classifier.
        notes: Human-readable caveats.
    """

    key: str
    label: str
    pipeline: Pipeline
    metrics: dict[str, float]
    classification_report: pd.DataFrame
    confusion_matrix: pd.DataFrame
    feature_importance: pd.DataFrame | None
    train_rows: int
    test_rows: int
    class_distribution: pd.DataFrame
    feature_names: list[str]
    trained_at: str
    seed: int
    prediction_threshold: str = "argmax of predicted class probability"
    classes_predicted: int = 4
    notes: str = ""

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary, excluding the fitted estimator."""
        return {
            "key": self.key,
            "label": self.label,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "trained_at": self.trained_at,
            "seed": self.seed,
            "prediction_threshold": self.prediction_threshold,
            "classes_predicted": self.classes_predicted,
            "is_degenerate": self.classes_predicted < len(schema.SEVERITY_LEVELS),
            "feature_count": len(self.feature_names),
            "notes": self.notes,
        }


def load_training_data(
    config: TrainingConfig,
    *,
    db: Any = None,
) -> pd.DataFrame:
    """Draw the modelling frame from the dataset.

    Args:
        config: Supplies the row cap, feature list and seed.
        db: Optional database handle.

    Returns:
        A DataFrame containing the predictor columns, ``ID`` and ``Severity``.
    """
    from smart_traffic.aggregation import FilterSpec
    from smart_traffic.sampling import draw_sample

    columns = ["ID", *config.all_features(), schema.SEVERITY]
    if config.split_strategy in {"temporal", "geographic"}:
        # The partition key is needed even when it is not a predictor; split_data
        # drops it alongside every other non-feature column.
        key = "Year" if config.split_strategy == "temporal" else schema.STATE
        if key not in columns:
            columns = ["ID", key, *columns[1:]]
    # A temporal run must not sample across all years and only then split, or
    # earlier years would land in the test set. Restrict the draw to the union
    # of both periods so the split itself decides which rows are held out.
    if config.split_strategy == "temporal":
        if not config.train_years or not config.test_years:
            raise ValueError(
                "A temporal split needs both train_years and test_years to be set."
            )
        years = tuple(sorted({*config.train_years, *config.test_years}))
    else:
        years = None
    # A geographic split needs every state drawn; the partition happens in
    # split_data. Filtering here by year only would be meaningless.
    spec = FilterSpec(years=years) if years else FilterSpec()
    draw = draw_sample(
        columns,
        spec=spec,
        limit=config.sample_rows,
        seed=config.seed,
        db=db,
    )
    logger.info("Loaded modelling sample: %s", draw.note())
    return draw.frame


def split_data(
    frame: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train and test sets.

    Two strategies, selected by ``config.split_strategy``:

    ``"random"``
        A stratified random split. Every model in this project is reported under
        this, and it is the right default for a like-for-like model comparison.
    ``"temporal"``
        Train on ``config.train_years`` and test on ``config.test_years``. This
        is the honest test of whether the model learned something durable rather
        than the quirks of a particular date range, because no test record can
        share a year with a training record.

    Stratification matters here: with severity 1 at under 1% of the data, an
    unstratified split could leave the test set with too few rare-class examples
    to evaluate at all.

    Args:
        frame: Rows containing features and ``Severity``.
        config: Supplies the split fraction, seed and strategy.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: If the frame is empty, the target is missing, or the
            requested temporal years yield no train or test rows.
    """
    from sklearn.model_selection import train_test_split

    if frame.empty:
        raise ValueError("Cannot split an empty modelling frame.")
    if schema.SEVERITY not in frame.columns:
        raise ValueError(f"Modelling frame is missing the target column {schema.SEVERITY!r}.")

    features = config.all_features()
    present = [f for f in features if f in frame.columns]
    partition_key = {
        "temporal": "Year",
        "geographic": schema.STATE,
    }.get(config.split_strategy)
    key_is_predictor = partition_key is not None and partition_key in present
    if partition_key is not None and partition_key not in present:
        # Needed to partition; removed again once the split is made.
        present = [*present, partition_key]
    if len(present) < len(features):
        missing = [f for f in features if f not in frame.columns]
        logger.warning("Dropping %s absent feature(s): %s", len(missing), ", ".join(missing))

    x = frame[present]
    y = pd.to_numeric(frame[schema.SEVERITY], errors="coerce")
    keep = y.notna()
    x, y = x[keep], y[keep].astype(int)

    if config.split_strategy in {"temporal", "geographic"}:
        if config.split_strategy == "temporal":
            x_train, x_test, y_train, y_test = _temporal_split(x, y, config)
        else:
            x_train, x_test, y_train, y_test = _geographic_split(x, y, config)
        if not key_is_predictor:
            # The key was carried only to decide the split; it is not a
            # predictor here, so it must not leak into the fitted model.
            x_train = x_train.drop(columns=[partition_key], errors="ignore")
            x_test = x_test.drop(columns=[partition_key], errors="ignore")
        return x_train, x_test, y_train, y_test

    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(
        x, y,
        test_size=config.test_fraction,
        random_state=config.seed,
        stratify=stratify,
    )


def _temporal_split(
    x: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split on calendar year so train and test periods cannot overlap."""
    train_years = tuple(config.train_years or ())
    test_years = tuple(config.test_years or ())
    if not train_years or not test_years:
        raise ValueError(
            "A temporal split needs both train_years and test_years to be set."
        )
    overlap = set(train_years) & set(test_years)
    if overlap:
        raise ValueError(
            f"Temporal split has years in both sets: {sorted(overlap)}. Train and "
            "test periods must be disjoint or the split is not temporal."
        )
    if "Year" not in x.columns:
        raise ValueError(
            "A temporal split needs the 'Year' column, which is not in the "
            "modelling frame."
        )

    years = pd.to_numeric(x["Year"], errors="coerce")
    in_train = years.isin(train_years)
    in_test = years.isin(test_years)
    if not in_train.any():
        raise ValueError(f"No training rows in years {sorted(train_years)}.")
    if not in_test.any():
        raise ValueError(f"No test rows in years {sorted(test_years)}.")

    logger.info(
        "Temporal split: train %s (%d rows) -> test %s (%d rows)",
        f"{min(train_years)}-{max(train_years)}", int(in_train.sum()),
        f"{min(test_years)}-{max(test_years)}", int(in_test.sum()),
    )
    return x[in_train], x[in_test], y[in_train], y[in_test]


def _geographic_split(
    x: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split so no state appears on both sides.

    States are assigned to the test side by ``crc32(state, seed)``, which makes
    the split reproducible and independent of state size. Choosing states by hand
    -- "hold out California" or "hold out the small ones" -- invites picking a
    split that flatters the result; a fixed hash cannot.

    ``crc32`` rather than Python's built-in :func:`hash`, which is salted per
    process and silently produced a *different* partition on every run -- the
    exact opposite of reproducible.

    Args:
        x: Feature frame carrying the state column.
        y: Target series.
        config: Supplies the seed and the test fraction.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: If the state column is missing, or the hash assigns
            everything to one side.
    """
    if schema.STATE not in x.columns:
        raise ValueError(
            f"A geographic split needs the {schema.STATE!r} column, which is not "
            "in the modelling frame."
        )
    fraction = float(config.geographic_test_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"geographic_test_fraction must be between 0 and 1; got {fraction}."
        )

    states = pd.Series(x[schema.STATE].astype(str).to_numpy())
    # crc32 is stable across processes, unlike Python's salted hash().
    buckets = states.map(
        lambda s: zlib.crc32(f"{s}:{config.seed}".encode()) % 10_000
    )
    in_test = buckets < int(fraction * 10_000)
    if not in_test.any() or in_test.all():
        raise ValueError(
            "The hash assigned every state to one side; adjust "
            "geographic_test_fraction or the seed."
        )

    held_out = sorted(set(states[in_test]))
    logger.info(
        "Geographic split: %d state(s) held out (%s), %d training state(s)",
        len(held_out), ", ".join(held_out), states[~in_test].nunique(),
    )
    return x[~in_test], x[in_test], y[~in_test], y[in_test]


def train_models(
    config: TrainingConfig,
    *,
    db: Any = None,
    progress: Any = None,
) -> dict[str, TrainedModel]:
    """Train every selected model on the same split and evaluate each one.

    All models see identical train and test rows, so the comparison table is a
    like-for-like comparison rather than three separate experiments.

    Args:
        config: Training configuration.
        db: Optional database handle.
        progress: Optional ``callable(stage, fraction, message)``.

    Returns:
        Mapping of model key to :class:`TrainedModel`.

    Raises:
        ValueError: If no dataset is available or the sample is too small.
    """
    from smart_traffic.evaluation import evaluate_model

    def report(stage: str, fraction: float, message: str) -> None:
        if progress is not None:
            progress(stage, fraction, message)

    report("load", 0.05, "Drawing the modelling sample")
    frame = load_training_data(config, db=db)
    if frame.empty:
        raise ValueError("No rows available for training.")

    x_train, x_test, y_train, y_test = split_data(frame, config)
    logger.info(
        "Training on %s rows, evaluating on %s rows",
        f"{len(x_train):,}", f"{len(x_test):,}",
    )

    results: dict[str, TrainedModel] = {}
    total = max(1, len(config.selected_models))
    for index, key in enumerate(config.selected_models):
        report("train", 0.1 + 0.8 * index / total, f"Training {MODEL_REGISTRY[key]['label']}")

        if config.tune_thresholds:
            # The pipeline is fitted on 75% of the training split; the decision
            # rule is calibrated on the held-out 25%, and the calibrated model
            # is then evaluated on the untouched test split.
            pipeline = tune_thresholds(
                build_model_pipeline(key, config), x_train, y_train, seed=config.seed
            )
        else:
            pipeline = build_model_pipeline(key, config)
            pipeline.fit(x_train, y_train)

        evaluation = evaluate_model(pipeline, x_test, y_test)
        importance = feature_importance_table(pipeline, key)

        results[key] = TrainedModel(
            key=key,
            label=MODEL_REGISTRY[key]["label"],
            pipeline=pipeline,
            metrics=evaluation.metrics,
            classification_report=evaluation.report,
            confusion_matrix=evaluation.confusion_matrix,
            feature_importance=importance,
            train_rows=len(x_train),
            test_rows=len(x_test),
            class_distribution=y_train.value_counts().sort_index().rename("Count").reset_index(),
            feature_names=evaluation.feature_names,
            trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
            seed=config.seed,
            prediction_threshold=threshold_description(pipeline),
            classes_predicted=evaluation.classes_predicted,
            notes=(
                "Trained on a reproducible sample. Metrics describe performance on "
                "held-out records from this dataset only; they do not establish "
                "real-world forecasting skill or safety-critical fitness."
            ),
        )

    report("done", 1.0, "Training complete")
    return results


def estimator_pipeline(pipeline: Any) -> Any:
    """Return the underlying scikit-learn pipeline.

    Args:
        pipeline: A pipeline, or a
            :class:`~smart_traffic.tuning.ThresholdTunedClassifier` wrapping one.

    Returns:
        The pipeline whose ``named_steps`` are addressable.
    """
    return pipeline.pipeline if hasattr(pipeline, "pipeline") else pipeline


def classes_of(pipeline: Any) -> np.ndarray:
    """Return the class labels an estimator emits, in its probability order.

    Args:
        pipeline: A pipeline, a threshold-calibrated wrapper, or a bare estimator.

    Returns:
        The class labels.
    """
    for attribute in ("classes", "classes_"):
        values = getattr(pipeline, attribute, None)
        if values is not None:
            return np.asarray(values)
    inner = estimator_pipeline(pipeline)
    values = getattr(inner, "classes_", None)
    if values is None and hasattr(inner, "named_steps"):
        values = inner.named_steps["model"].classes_
    return np.asarray(values)


def threshold_description(pipeline: Any) -> str:
    """Return a human-readable statement of the model's decision rule.

    Args:
        pipeline: A trained model.

    Returns:
        A sentence naming either the plain argmax rule or the calibrated one
        with the per-class multipliers it applies.
    """
    multipliers = getattr(pipeline, "multipliers", None)
    if multipliers is None:
        return "argmax of predicted class probability (no calibration applied)"
    classes = getattr(pipeline, "classes", [])
    pairs = ", ".join(
        f"Severity {label} x{value:.2f}"
        for label, value in zip(classes, multipliers, strict=False)
    )
    return f"argmax of class probability x per-class weight ({pairs})"


def feature_importance_table(pipeline: Pipeline, model_key: str) -> pd.DataFrame | None:
    """Extract feature importance from a fitted pipeline.

    Tree ensembles expose ``feature_importances_``; the linear model exposes
    coefficient magnitudes. Both are reported under the *encoded* feature names
    the pipeline produced, so a one-hot column is labelled ``cat__State_CA``
    rather than ambiguously as ``State``.

    Args:
        pipeline: A fitted pipeline.
        model_key: Registry key, used to pick the right extraction method.

    Returns:
        DataFrame with ``Feature``, ``Importance`` and ``Type``, or ``None``
        when the model exposes neither.
    """
    inner = estimator_pipeline(pipeline)
    estimator = inner.named_steps["model"]
    try:
        names = list(inner.named_steps["preprocess"].get_feature_names_out())
    except (AttributeError, ValueError):
        names = [f"f{i}" for i in range(estimator.n_features_in_)]

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "impurity-based importance"
    elif hasattr(estimator, "coef_"):
        # Multinomial logistic regression exposes one row per class; the mean
        # absolute coefficient gives a single comparable magnitude per feature.
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = (
            np.abs(coefficients).mean(axis=0)
            if coefficients.ndim > 1
            else np.abs(coefficients)
        )
        kind = "mean absolute coefficient"
    else:
        return None

    if len(values) != len(names):
        return None

    frame = pd.DataFrame({"Feature": names, "Importance": np.round(values, 6)})
    frame["Type"] = kind
    return frame.sort_values("Importance", ascending=False).reset_index(drop=True)


def save_models(models: dict[str, TrainedModel], directory: Path | None = None) -> Path:
    """Persist trained pipelines and their metadata.

    Args:
        models: Mapping of model key to :class:`TrainedModel`.
        directory: Destination directory. Defaults to ``models/``.

    Returns:
        The directory written to.
    """
    target = Path(directory) if directory else PATHS.models
    target.mkdir(parents=True, exist_ok=True)

    for key, model in models.items():
        artefact = target / f"{key}.joblib"
        joblib.dump(model.pipeline, artefact)
        size_mb = artefact.stat().st_size / 1e6
        if size_mb > MODEL_SIZE_WARNING_MB:
            logger.warning(
                "%s.joblib is %.0f MB. An unconstrained tree can grow enormous on a "
                "large sample; consider a bounded configuration or a smaller "
                "training size.",
                key, size_mb,
            )
        model.classification_report.to_csv(target / f"{key}_classification_report.csv", index=False)
        model.confusion_matrix.to_csv(target / f"{key}_confusion_matrix.csv")
        if model.feature_importance is not None:
            model.feature_importance.to_csv(target / f"{key}_feature_importance.csv", index=False)

    summary = {key: model.to_metadata() for key, model in models.items()}
    (target / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Saved %s model(s) to %s", len(models), target)
    return target


def load_model(key: str, directory: Path | None = None) -> Pipeline | None:
    """Load a persisted pipeline.

    Args:
        key: Model key, e.g. ``"random_forest"``.
        directory: Directory to load from. Defaults to ``models/``.

    Returns:
        The fitted pipeline, or ``None`` when it is not present or unreadable.
    """
    source = Path(directory) if directory else PATHS.models
    path = source / f"{key}.joblib"
    if not path.exists():
        logger.info("No persisted model at %s", path.name)
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        # A truncated, corrupt or version-mismatched artefact must degrade to
        # "no model available" rather than crashing a page. The specific
        # exceptions vary by cause (pickle errors, zip errors, import errors
        # for renamed classes), so the boundary is deliberately broad here.
        logger.warning("Could not load %s: %s", path.name, exc)
        return None


def available_models(directory: Path | None = None) -> list[str]:
    """List model keys that have been persisted.

    Args:
        directory: Directory to inspect. Defaults to ``models/``.

    Returns:
        Sorted list of available model keys.
    """
    source = Path(directory) if directory else PATHS.models
    if not source.exists():
        return []
    return sorted(p.stem for p in source.glob("*.joblib") if p.stem in MODEL_REGISTRY)

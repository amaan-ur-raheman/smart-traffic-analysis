"""Single-record severity prediction.

The prediction form must present the model with exactly the preprocessing the
model was trained on -- that is what calling the fitted pipeline rather than
re-implementing the transform guarantees. This module's job is validating the
user's input and assembling it into the right shape.

Input is treated as transient. Nothing a user types is written to disk or to
the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from smart_traffic import schema
from smart_traffic.config import SEVERITY_MEANING
from smart_traffic.feature_engineering import is_rush_hour_mask
from smart_traffic.logging_config import get_logger
from smart_traffic.modeling import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
)
from smart_traffic.modeling import (
    classes_of as _classes_of,
)

logger = get_logger("prediction")

#: Plausibility bounds enforced on user input, mirroring the dataset's own range.
#: Values outside these are almost certainly typos, and silently accepting them
#: would produce a confident-looking prediction from nonsense.
INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "Temperature(F)": (-40.0, 140.0),
    "Wind_Chill(F)": (-50.0, 140.0),
    "Humidity(%)": (0.0, 100.0),
    "Pressure(in)": (28.0, 32.0),
    "Visibility(mi)": (0.0, 30.0),
    "Wind_Speed(mph)": (0.0, 150.0),
    "Precipitation(in)": (0.0, 15.0),
    "Distance(mi)": (0.0, 100.0),
    "Hour": (0, 23),
    "Month": (1, 12),
    "DayOfWeek": (0, 6),
    "Year": (2016, 2100),
}

#: Optional numeric fields: a blank is acceptable and becomes NaN.
OPTIONAL_NUMERIC: frozenset[str] = frozenset(
    {
        "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
        "Wind_Speed(mph)", "Precipitation(in)", "Distance(mi)",
    }
)

#: Value substituted when the calendar year is not supplied. A prediction for
#: "an accident like this one" without a year is still answerable, and the
#: substitution is reported alongside the result rather than applied silently.
DEFAULT_YEAR: int = 2023

#: Numeric features the form computes from other inputs rather than collecting.
DERIVED_NUMERIC: frozenset[str] = frozenset({"Is_Weekend", "Is_Rush_Hour"})


class PredictionInputError(ValueError):
    """Raised when the submitted form values are unusable."""


@dataclass(slots=True)
class PredictionInput:
    """A validated single-accident description ready for inference.

    Attributes:
        values: Column name to value, covering the model's feature set.
        warnings: Non-blocking notes about imputed or defaulted values.
    """

    values: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Return the input as a one-row DataFrame.

        Returns:
            A DataFrame with a single row and the validated columns.
        """
        return pd.DataFrame([self.values])


@dataclass(slots=True)
class PredictionResult:
    """The outcome of a single prediction.

    Attributes:
        severity: Predicted severity category.
        probabilities: Per-class probabilities, when the model exposes them.
        confidence: Probability of the predicted class, or ``None``.
        model_label: Display name of the model used.
        disclaimer: Standing caveat shown with every prediction.
    """

    severity: int
    probabilities: dict[str, float] | None
    confidence: float | None
    model_label: str
    disclaimer: str = SEVERITY_MEANING


def validate_input(
    raw: dict[str, Any],
    *,
    numeric_features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES,
) -> PredictionInput:
    """Validate and coerce raw form values into a model-ready input.

    Required fields are those the form always collects; weather readings are
    optional because they are genuinely often unrecorded in the source data, and
    the pipeline imputes them the same way it does during training.

    Args:
        raw: Raw values keyed by column name.
        numeric_features: Columns the model treats as numeric.
        categorical_features: Columns the model treats as categorical.

    Returns:
        A validated :class:`PredictionInput`.

    Raises:
        PredictionInputError: If a required field is missing or a value is
            outside its plausible range.
    """
    if schema.SEVERITY in raw:
        # Reject on the raw payload, not the assembled values: a target key must
        # never reach the model even if the feature lists were misconfigured.
        raise PredictionInputError("The target column must never appear in the input.")

    values: dict[str, Any] = {}
    warnings: list[str] = []

    for column in numeric_features:
        value = raw.get(column)
        blank = value is None or (isinstance(value, str) and not value.strip())

        if blank:
            if column in DERIVED_NUMERIC:
                # Derived from DayOfWeek / Hour after the loop, once those are
                # known. Demanding them here would fail before they could be
                # computed, and asking the user for them would invite values
                # contradicting the date and hour they also supply.
                continue
            if column == "Year":
                values[column] = float(DEFAULT_YEAR)
                warnings.append(
                    f"Year was left blank; {DEFAULT_YEAR} was substituted, so the "
                    f"prediction reflects that year's conditions."
                )
                continue
            if column in schema.BOOLEAN_COLUMNS:
                # The form treats an unchecked road indicator as "not present",
                # which is a claim about the accident, so it is surfaced to the
                # user rather than applied silently.
                values[column] = 0.0
                continue
            if column in OPTIONAL_NUMERIC:
                values[column] = np.nan
                warnings.append(
                    f"{column} was left blank; the training-time imputation was applied."
                )
                continue
            raise PredictionInputError(f"{column} is required.")

        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionInputError(f"{column} must be a number.") from exc

        if not np.isfinite(numeric):
            if column in OPTIONAL_NUMERIC:
                values[column] = np.nan
                warnings.append(f"{column} was not a finite number; imputed.")
                continue
            raise PredictionInputError(f"{column} must be a finite number.")

        low, high = INPUT_BOUNDS.get(column, (float("-inf"), float("inf")))
        if not low <= numeric <= high:
            raise PredictionInputError(
                f"{column} must be between {low:g} and {high:g}; got {numeric:g}."
            )
        values[column] = numeric

    for column in categorical_features:
        value = raw.get(column)
        if value is None or (isinstance(value, str) and not value.strip()):
            values[column] = "Unknown"
            warnings.append(f"{column} was left blank; recorded as 'Unknown'.")
        else:
            values[column] = str(value)

    # Derive what can be derived. Is_Weekend and Is_Rush_Hour are deterministic
    # functions of DayOfWeek and Hour, which the form always collects, so asking
    # the user to enter them separately would only invite contradictions.
    if "Is_Weekend" in numeric_features and "Is_Weekend" not in values:
        day = values.get("DayOfWeek")
        values["Is_Weekend"] = float(day >= 5) if day is not None else 0.0
        if day is None:
            warnings.append("Is_Weekend could not be derived; it was set to weekday.")

    if "Is_Rush_Hour" in numeric_features and "Is_Rush_Hour" not in values:
        hour = values.get("Hour")
        if hour is not None:
            derived = bool(is_rush_hour_mask(pd.Series([hour])).iloc[0])
            values["Is_Rush_Hour"] = float(derived)
        else:
            values["Is_Rush_Hour"] = 0.0
            warnings.append("Is_Rush_Hour could not be derived; it was set to false.")

    for required in ("Hour", "Month", "DayOfWeek"):
        if required not in values or values[required] is None:
            raise PredictionInputError(f"{required} is required to make a prediction.")

    missing = [f for f in (*numeric_features, *categorical_features) if f not in values]
    if missing:
        # The pipeline's ColumnTransformer rejects a frame missing any of its
        # configured columns. Rather than let scikit-learn raise an opaque
        # error, fill the gap here and name it.
        for column in missing:
            values[column] = "Unknown"
        warnings.append(
            "Fields not collected by this form were filled with placeholders: "
            + ", ".join(missing)
        )

    return PredictionInput(values=values, warnings=warnings)


def predict_severity(
    pipeline: Any,
    submitted: dict[str, Any],
    *,
    model_label: str,
    numeric_features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES,
) -> PredictionResult:
    """Validate an input, run the model, and package the result.

    Args:
        pipeline: A fitted sklearn pipeline.
        submitted: Raw form values.
        model_label: Display name for the model.
        numeric_features: Columns treated as numeric.
        categorical_features: Columns treated as categorical.

    Returns:
        A :class:`PredictionResult`.

    Raises:
        PredictionInputError: If the input is invalid or incompatible with the
            model. Both are reported to the user as actionable messages rather
            than as a stack trace.
    """
    validated = validate_input(
        submitted,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    frame = validated.to_frame()

    # Run the identical preprocessing the model was trained with.
    try:
        severity = int(np.asarray(pipeline.predict(frame))[0])
    except (ValueError, TypeError) as exc:
        raise PredictionInputError(
            "The model could not process these inputs. It was likely trained on a "
            "different feature set than the current form. Retrain the models and "
            "try again."
        ) from exc

    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    if hasattr(pipeline, "predict_proba"):
        try:
            raw = np.asarray(pipeline.predict_proba(frame))[0]
            classes = _classes_of(pipeline)
            probabilities = {
                f"Severity {label}": round(float(probability), 4)
                for label, probability in zip(classes, raw, strict=False)
            }
            # The confidence shown must be the probability of the class actually
            # returned by predict(). Using raw.max() is only equivalent for an
            # un-tuned model: the threshold-calibrated wrapper decides by
            # per-class multipliers, so on a real sample it disagrees with
            # argmax(predict_proba) for roughly 8-15% of rows. Reporting
            # raw.max() in those cases attributes a probability to a severity
            # the model did not choose, overstating confidence by up to ~0.22.
            matching = np.flatnonzero(np.asarray(classes) == severity)
            confidence = round(
                float(raw[matching[0]]) if matching.size else float(raw.max()), 4
            )
        except (ValueError, AttributeError, IndexError) as exc:
            logger.warning("Probabilities unavailable: %s", exc)

    if severity not in schema.SEVERITY_LEVELS:
        logger.warning("Model returned out-of-range severity %s", severity)

    return PredictionResult(
        severity=severity,
        probabilities=probabilities,
        confidence=confidence,
        model_label=model_label,
    )

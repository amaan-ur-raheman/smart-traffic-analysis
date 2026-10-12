"""Page 10 — Severity prediction.

Model training and the prediction form. Training is never automatic: it is an
explicit action, because it is the most expensive operation in the application
and because silently fitting models on launch would make the page slow and
non-reproducible.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from smart_traffic.config import PATHS, SETTINGS
from smart_traffic.db import DB
from smart_traffic.modeling import (
    MODEL_REGISTRY,
    TrainingConfig,
    available_models,
    load_model,
    save_models,
    train_models,
)
from smart_traffic.prediction import PredictionInputError, predict_severity
from smart_traffic.ui import (
    empty_state,
    info_box,
    page_header,
    require_dataset,
    section,
    severity_regime_warning,
    sidebar_header,
)
from smart_traffic.visualizations import probability_chart

st.set_page_config(page_title="Severity Prediction", page_icon="🤖", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Severity Prediction",
    "Train classification models for the recorded severity category, and apply "
    "a trained model to a single hypothetical accident.",
    icon="🤖",
)
severity_regime_warning()

# ---------------------------------------------------------------------------
# Scenario and leakage policy
# ---------------------------------------------------------------------------
info_box(
    "info",
    "Prediction scenario",
    "**Predict the recorded severity category of an accident using only "
    "information available when it is first reported.**\n\n"
    "This definition is what excludes `End_Time` and anything derived from it. "
    "An accident's duration is not known until it has ended, so a model using "
    "duration would score well while being unable to handle a real, still-"
    "in-progress incident. The exclusion is enforced by code, not by convention.",
)

from smart_traffic.modeling import EXCLUDED_FEATURES  # noqa: E402

with st.expander("Leakage policy — what is excluded and why", expanded=False):
    st.dataframe(
        pd.DataFrame(
            [{"Feature": name, "Reason": reason}
             for name, reason in EXCLUDED_FEATURES.items()]
        ),
        width="stretch", hide_index=True,
    )

tab_train, tab_predict = st.tabs(["Train models", "Predict a single accident"])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
with tab_train:
    section(
        "Training configuration",
        "Models train on a reproducible sample rather than the whole dataset, "
        "because fitting a Random Forest on 7.7 million rows is neither fast "
        "nor necessary for a comparison.",
    )

    controls = st.columns([1, 1, 1.4])
    train_rows = controls[0].number_input(
        # Default from SETTINGS so the UI, the CLI and the documented protocol
        # all agree. It was hardcoded to 300_000 here, so a user training from
        # the UI silently ran a different protocol than the README reports and
        # overwrote the canonical artefacts with worse numbers.
        "Training sample size", 5_000, 1_000_000,
        SETTINGS.sample_training_rows, step=5_000,
        help="Rows drawn for training. Larger is more stable but slower.",
    )
    test_fraction = controls[1].slider(
        "Test fraction", 0.10, 0.40, 0.20, 0.05,
    )
    seed = controls[2].number_input("Random seed", 0, 10_000, SETTINGS.random_seed, step=1)

    tune = st.toggle(
        "Calibrate decision thresholds",
        value=True,
        help="Fits each model on 75% of the training split, then tunes per-class "
             "decision thresholds on the remaining 25% to maximise macro F1. "
             "This raises macro F1 substantially without abandoning the rare "
             "severity categories the way softer class weights do.",
    )
    chosen = st.multiselect(
        "Models to train",
        list(MODEL_REGISTRY),
        default=list(MODEL_REGISTRY),
        format_func=lambda key: MODEL_REGISTRY[key]["label"],
    )
    for key in chosen:
        st.caption(f"**{MODEL_REGISTRY[key]['label']}** — {MODEL_REGISTRY[key]['description']}")

    if not chosen:
        st.warning("Select at least one model to train.", icon="⚠️")
    else:
        estimated = int(train_rows * (1 - test_fraction))
        st.caption(
            f"Approximately **{estimated:,}** training rows and "
            f"**{int(train_rows * test_fraction):,}** evaluation rows per model, "
            f"identical across models so the comparison is like-for-like."
        )

        if st.button("Train models", type="primary", key="train_models"):
            config = TrainingConfig(
                sample_rows=train_rows,
                test_fraction=test_fraction,
                seed=seed,
                selected_models=tuple(chosen),
                tune_thresholds=tune,
            )
            progress_bar = st.progress(0.0, text="Preparing…")
            status = st.empty()

            def update(_stage: str, fraction: float, message: str) -> None:
                """Render training progress into the bar and the status line."""
                progress_bar.progress(min(fraction, 1.0), text=message)
                status.text(message)

            try:
                with st.spinner("Training…"):
                    DB.reset()
                    trained = train_models(config, db=DB, progress=update)
                    save_models(trained)
                    st.session_state["trained"] = {
                        k: m.to_metadata() for k, m in trained.items()
                    }
                progress_bar.progress(1.0, text="Training complete")
                st.success(
                    f"Trained {len(trained)} model(s). Results are on the "
                    f"**Model Evaluation** page."
                )
            except (ValueError, MemoryError) as exc:
                progress_bar.empty()
                st.error(f"Training could not complete: {exc}")
            finally:
                status.empty()

    trained_keys = available_models()
    if trained_keys:
        section("Persisted models")
        st.markdown(
            f"""
Models on disk in `{PATHS.models}`:

| Model | File |
| --- | --- |
""" + "\n".join(
                f"| {MODEL_REGISTRY[k]['label']} | `{k}.joblib` |" for k in trained_keys
            )
        )
        st.caption(
            "Each `.joblib` file contains the complete fitted pipeline: "
            "imputation, scaling, one-hot encoding and the estimator. Applying it "
            "guarantees the prediction form uses the same preprocessing as training."
        )

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
with tab_predict:
    if not available_models():
        empty_state(
            "No model has been trained yet.",
            detail="Use the **Train models** tab above, or run `uv run sta train` "
                   "in a terminal, then return to this page.",
        )
    else:
        model_key = st.selectbox(
            "Model to use",
            available_models(),
            format_func=lambda k: MODEL_REGISTRY[k]["label"],
        )
        pipeline = load_model(model_key)
        if pipeline is None:
            st.error(
                f"The saved artefact for `{model_key}` could not be loaded. It may "
                f"be truncated or built with a different library version. Retrain "
                f"the models to regenerate it."
            )
        else:
            st.success(f"Loaded **{MODEL_REGISTRY[model_key]['label']}**.")

            st.caption(
                "Fields marked with * are optional. A blank optional field is "
                "imputed exactly as the training pipeline imputes it, and any "
                "imputation is reported with the result."
            )

            with st.form("prediction_form"):
                st.markdown("**When**")
                time_columns = st.columns(4)
                hour = time_columns[0].selectbox("Hour", list(range(24)), 8)
                month = time_columns[1].selectbox(
                    "Month", list(range(1, 13)),
                    format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"), index=2,
                )
                day_of_week = time_columns[2].selectbox(
                    "Day of week", list(range(7)),
                    format_func=lambda d: pd.Timestamp(2024, 3, 4 + d).strftime("%A"), index=0,
                )
                year = time_columns[3].number_input(
                    "Year", 2016, 2100, 2023, step=1
                )
                is_weekend = time_columns[0].checkbox("Weekend", value=False)
                is_rush = time_columns[1].checkbox("Rush hour", value=False)

                st.markdown("**Where**")
                place_columns = st.columns(2)
                states_present = DB.sql(
                    "SELECT DISTINCT State FROM accidents WHERE State IS NOT NULL ORDER BY State"
                )["State"].tolist()
                state = place_columns[0].selectbox("State", states_present or ["CA"])
                weather_options = DB.sql(
                    "SELECT DISTINCT Weather_Condition FROM accidents "
                    "WHERE Weather_Condition IS NOT NULL ORDER BY Weather_Condition LIMIT 400"
                )["Weather_Condition"].tolist()
                weather = place_columns[1].selectbox(
                    "Weather condition", weather_options or ["Fair"]
                )

                st.markdown("**Recorded conditions** *(optional — leave blank if unrecorded)*")
                weather_columns = st.columns(4)
                temperature = weather_columns[0].number_input(
                    "Temperature (°F)", -40.0, 140.0, 70.0, 1.0
                )
                visibility = weather_columns[1].number_input(
                    "Visibility (miles)", 0.0, 30.0, 10.0, 0.5
                )
                wind_speed = weather_columns[2].number_input(
                    "Wind speed (mph)", 0.0, 150.0, 5.0, 1.0
                )
                humidity = weather_columns[3].number_input(
                    "Humidity (%)", 0.0, 100.0, 60.0, 1.0
                )
                extra_columns = st.columns(3)
                precipitation = extra_columns[0].number_input(
                    "Precipitation (in)", 0.0, 15.0, 0.0, 0.01, format="%.2f"
                )
                pressure = extra_columns[1].number_input(
                    "Pressure (in)", 28.0, 32.0, 30.0, 0.01, format="%.2f"
                )
                distance = extra_columns[2].number_input(
                    "Distance (mi)", 0.0, 100.0, 0.5, 0.1
                )

                st.markdown("**Road indicators**")
                indicator_names = [
                    ("Traffic_Signal", "Traffic signal"),
                    ("Crossing", "Crossing"),
                    ("Junction", "Junction"),
                    ("Stop", "Stop sign"),
                    ("Railway", "Railway"),
                    ("Roundabout", "Roundabout"),
                    ("Bump", "Bump"),
                    ("Give_Way", "Give way"),
                    ("No_Exit", "No exit"),
                    ("Amenity", "Amenity"),
                    ("Station", "Station"),
                    ("Traffic_Calming", "Traffic calming"),
                    ("Turning_Loop", "Turning loop"),
                ]
                indicator_values: dict[str, float] = {}
                indicator_columns = st.columns(5)
                for index, (key, label) in enumerate(indicator_names):
                    with indicator_columns[index % 5]:
                        indicator_values[key] = float(
                            indicator_columns[index % 5].checkbox(label, value=False,
                                key=f"ind_{key}")
                        )

                submitted = st.form_submit_button(
                    "Predict severity", type="primary", width="stretch"
                )

            if submitted:
                payload = {
                    "Hour": hour, "Month": month, "DayOfWeek": day_of_week,
                    "Year": year, "Is_Weekend": int(is_weekend),
                    "Is_Rush_Hour": int(is_rush), "State": state,
                    "Weather_Condition": weather, "Temperature(F)": temperature,
                    "Visibility(mi)": visibility, "Wind_Speed(mph)": wind_speed,
                    "Humidity(%)": humidity, "Precipitation(in)": precipitation,
                    "Pressure(in)": pressure, "Distance(mi)": distance,
                    **indicator_values,
                }
                try:
                    result = predict_severity(
                        pipeline, payload,
                        model_label=MODEL_REGISTRY[model_key]["label"],
                    )
                except PredictionInputError as exc:
                    st.error(str(exc))
                else:
                    st.markdown("---")
                    columns = st.columns([1, 1.4])
                    columns[0].metric(
                        "Predicted severity", str(result.severity), delta=None
                    )
                    if result.confidence is not None:
                        columns[1].metric(
                            "Model confidence",
                            f"{result.confidence:.1%}",
                            help="Probability assigned to the predicted class. It is "
                                 "the model's confidence, not a guarantee of correctness.",
                        )

                    if result.probabilities:
                        st.plotly_chart(
                            probability_chart(
                                result.probabilities, result.severity,
                                title="Predicted class probabilities",
                            ),
                            width="stretch",
                            config={"displaylogo": False},
                        )

                    warnings = getattr(result, "warnings", None)
                    if warnings:
                        for warning in warnings:
                            st.caption(f"ℹ️ {warning}")

                    info_box(
                        "warning",
                        "This is a model estimate, not a guarantee",
                        f"{result.disclaimer}\n\n"
                        f"The model was trained on a reproducible sample of records "
                        f"from this dataset only. It has not been validated against "
                        f"real-time reporting, and its output should not be used as "
                        f"the sole basis of any operational or safety decision. Your "
                        f"input was used for this prediction only and was not stored.",
                    )

"""Experiment harness for improving severity prediction.

Every variant is evaluated on one fixed, stratified train/test split so the
numbers are directly comparable. Results are appended to /tmp/results.json as
they complete, so a long run can be inspected part-way through.

Run:  uv run python scripts/experiment.py [stage ...]
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from smart_traffic.db import DB  # noqa: E402
from smart_traffic.modeling import (  # noqa: E402
    TrainingConfig,
    build_model_pipeline,
    load_training_data,
    split_data,
)

RESULTS = Path("/tmp/results.json")
SEED = 42
SAMPLE = 400_000


def load_split():
    """Return the fixed (x_train, x_test, y_train, y_test) split."""
    DB.reset()
    cfg = TrainingConfig(sample_rows=SAMPLE, test_fraction=0.2, seed=SEED)
    frame = load_training_data(cfg, db=DB)
    return (*split_data(frame, cfg), cfg)


def add_cyclic(frame: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for the cyclical calendar fields.

    A tree splits hour at a threshold: `hour <= 9` then `hour <= 18`. It has no
    way to express that 23:00 and 00:00 are neighbours, so a rush-hour pattern
    straddling midnight gets cut in half. Encoding the angle directly makes the
    wrap-around visible to a linear threshold.
    """
    out = frame.copy()
    for column, period in (("Hour", 24), ("Month", 12), ("DayOfWeek", 7)):
        if column not in out.columns:
            continue
        angle = 2 * np.pi * pd.to_numeric(out[column], errors="coerce") / period
        out[f"{column}_sin"] = np.sin(angle).round(6)
        out[f"{column}_cos"] = np.cos(angle).round(6)
    return out


def record(name: str, y_true, y_pred, seconds: float, note: str = "") -> dict:
    """Score, print and persist one result row."""
    metrics = {
        "name": name,
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "bal_acc": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "recall": {
            int(k): round(float(v), 4)
            for k, v in zip(
                [1, 2, 3, 4],
                recall_score(y_true, y_pred, labels=[1, 2, 3, 4],
                             average=None, zero_division=0),
                strict=True,
            )
        },
        "seconds": round(seconds, 1),
        "note": note,
    }
    existing = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    existing = [r for r in existing if r["name"] != name]
    existing.append(metrics)
    RESULTS.write_text(json.dumps(existing, indent=2))

    r = metrics["recall"]
    print(
        f"  {name:<44} macroF1={metrics['macro_f1']:.4f} balAcc={metrics['bal_acc']:.4f} "
        f"acc={metrics['accuracy']:.4f}  R1={r[1]:.3f} R2={r[2]:.3f} R3={r[3]:.3f} R4={r[4]:.3f}"
        f"  [{seconds:.0f}s]"
    )
    return metrics


def table():
    """Print every recorded result, best macro F1 first."""
    if not RESULTS.exists():
        print("no results yet")
        return
    rows = sorted(json.loads(RESULTS.read_text()), key=lambda r: -r["macro_f1"])
    print(f"\n{'variant':<44}{'macroF1':>9}{'balAcc':>9}{'acc':>8}{'R1':>7}{'R4':>7}")
    print("-" * 84)
    for row in rows:
        print(
            f"{row['name']:<44}{row['macro_f1']:>9.4f}{row['bal_acc']:>9.4f}"
            f"{row['accuracy']:>8.4f}{row['recall']['1']:>7.3f}{row['recall']['4']:>7.3f}"
        )


# ---------------------------------------------------------------------------


def stage_class_weights(x_train, x_test, y_train, y_test, cfg):
    """Test alternative imbalance strategies against the current balanced weights."""
    print("\n=== STAGE 1: class-weight strategies (Random Forest) ===")
    counts = y_train.value_counts()
    n, k = len(y_train), y_train.nunique()

    # Every scheme below must stay strictly positive: sklearn rejects a
    # negative sample weight with an opaque "probabilities are not
    # non-negative" error at predict time. `log(1+x)` is used rather than
    # `1+log(x)` because the latter goes negative for the majority class.
    strategies = {
        "balanced (current)": "balanced",
        "sqrt-balanced": {int(c): float(np.sqrt(n / (k * counts[c]))) for c in counts.index},
        "quarter-balanced": {
            int(c): float((n / (k * counts[c])) ** 0.25) for c in counts.index
        },
        "moderate (1+0.5*bal)": {
            int(c): float(1 + 0.5 * (n / (k * counts[c]))) for c in counts.index
        },
        "log1p-balanced": {
            int(c): float(np.log1p(n / (k * counts[c]))) for c in counts.index
        },
        "none": None,
    }

    for label, weight in strategies.items():
        pipe = build_model_pipeline("random_forest", cfg)
        pipe.named_steps["model"].set_params(class_weight=weight)
        start = time.perf_counter()
        pipe.fit(x_train, y_train)
        record(f"RF / {label}", y_test, pipe.predict(x_test),
               time.perf_counter() - start, note=label)


def stage_hyperparameters(x_train, x_test, y_train, y_test, cfg):
    """Grid-search the Decision Tree and Random Forest depth/leaf settings."""
    print("\n=== STAGE 2: hyperparameters ===")
    for depth, leaf in [(8, 50), (12, 20), (None, 5), (None, 1), (20, 10)]:
        pipe = build_model_pipeline("decision_tree", cfg)
        pipe.named_steps["model"].set_params(max_depth=depth, min_samples_leaf=leaf)
        label = f"depth={depth},leaf={leaf}"
        start = time.perf_counter()
        pipe.fit(x_train, y_train)
        record(f"DT / {label}", y_test, pipe.predict(x_test),
               time.perf_counter() - start, note="hyperparameter sweep")

    for depth, leaf, trees in [(18, 20, 200), (None, 1, 300), (24, 5, 300), (None, 2, 500)]:
        pipe = build_model_pipeline("random_forest", cfg)
        pipe.named_steps["model"].set_params(
            max_depth=depth, min_samples_leaf=leaf, n_estimators=trees
        )
        label = f"depth={depth},leaf={leaf},n={trees}"
        start = time.perf_counter()
        pipe.fit(x_train, y_train)
        record(f"RF / {label}", y_test, pipe.predict(x_test),
               time.perf_counter() - start, note="hyperparameter sweep")


def stage_cyclic(x_train, x_test, y_train, y_test, cfg):
    """Add sin/cos encodings of the cyclical calendar fields."""
    print("\n=== STAGE 3: cyclical encoding ===")
    tr, te = add_cyclic(x_train), add_cyclic(x_test)

    for key, label in (("decision_tree", "DT"), ("random_forest", "RF")):
        pipe = build_model_pipeline(key, cfg)
        start = time.perf_counter()
        pipe.fit(tr, y_train)
        record(f"{label} / +cyclic", y_test, pipe.predict(te),
               time.perf_counter() - start, note="sin/cos of Hour, Month, DayOfWeek")


def stage_threshold(x_train, x_test, y_train, y_test, cfg):
    """Tune per-class decision thresholds instead of relying on argmax.

    Class weights change what the model *learns*; thresholds change what it
    *decides*. Tuning the two separately is the principled alternative to
    weakening the weights: the model keeps the recall that balanced weighting
    bought, and the decision rule is then chosen for the metric we actually
    care about.
    """
    print("\n=== STAGE 5: per-class threshold tuning ===")
    from sklearn.metrics import f1_score as macro_f1

    x_train_v, x_val, y_train_v, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
    )

    for key, label in (("decision_tree", "DT"), ("random_forest", "RF")):
        pipe = build_model_pipeline(key, cfg)
        start = time.perf_counter()
        pipe.fit(x_train_v, y_train_v)
        proba = pipe.named_steps["model"].predict_proba(
            pipe.named_steps["preprocess"].transform(x_val)
        )
        classes = pipe.named_steps["model"].classes_
        base = macro_f1(y_val, classes[proba.argmax(axis=1)], average="macro", zero_division=0)

        # Coordinate ascent over per-class probability multipliers.
        multipliers = np.ones(len(classes))
        best = base
        for _ in range(4):
            for index in range(len(classes)):
                for factor in (0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0):
                    trial = multipliers.copy()
                    trial[index] = factor
                    score = macro_f1(
                        y_val, classes[(proba * trial).argmax(axis=1)],
                        average="macro", zero_division=0,
                    )
                    if score > best:
                        best, multipliers = score, trial
        final = pipe.fit(x_train, y_train)
        _ = final
        test_proba = pipe.named_steps["model"].predict_proba(
            pipe.named_steps["preprocess"].transform(x_test)
        )
        record(f"{label} / balanced + tuned thresholds", y_test,
               classes[(test_proba * multipliers).argmax(axis=1)],
               time.perf_counter() - start,
               note=(
                   f"multipliers={np.round(multipliers, 2).tolist()}, "
                   f"val macroF1 {base:.4f}->{best:.4f}"
               ))


def stage_boosting(x_train, x_test, y_train, y_test, cfg):
    """Add gradient boosting, which sklearn ships without a new dependency."""
    print("\n=== STAGE 4: HistGradientBoostingClassifier ===")
    from sklearn.ensemble import HistGradientBoostingClassifier

    from smart_traffic.modeling import build_preprocessor

    numeric = [c for c in cfg.numeric_features if c in x_train.columns]
    categorical = [c for c in cfg.categorical_features if c in x_train.columns]
    pre = build_preprocessor(numeric, categorical, scale_numeric=False)
    tr = pre.fit_transform(x_train)
    te = pre.transform(x_test)
    print(f"  encoded matrix: {tr.shape} ({tr.nbytes / 1e9:.2f} GB)")

    for label, params in [
        ("default", {}),
        ("balanced+iters300", {"class_weight": "balanced", "max_iter": 300}),
        ("balanced+l2+iters500", {"class_weight": "balanced", "max_iter": 500,
                                 "learning_rate": 0.08, "l2_regularization": 1.0}),
    ]:
        model = HistGradientBoostingClassifier(random_state=SEED, **params)
        start = time.perf_counter()
        model.fit(tr, y_train)
        record(f"HistGB / {label}", y_test, model.predict(te),
               time.perf_counter() - start, note="gradient boosting")


STAGES = {
    "weights": stage_class_weights,
    "hyper": stage_hyperparameters,
    "cyclic": stage_cyclic,
    "threshold": stage_threshold,
    "boost": stage_boosting,
}


def main() -> int:
    """Run the requested stages."""
    requested = sys.argv[1:] or list(STAGES)
    x_train, x_test, y_train, y_test, cfg = load_split()
    print(f"split: train={len(x_train):,} test={len(x_test):,} seed={SEED}\n")

    for name in requested:
        if name not in STAGES:
            print(f"unknown stage: {name}. Available: {list(STAGES)}")
            continue
        STAGES[name](x_train, x_test, y_train, y_test, cfg)

    table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

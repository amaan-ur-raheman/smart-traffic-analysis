# Smart Traffic Analysis and Accident Severity Prediction

An interactive analytics application for the **US Accidents (2016–2023)** dataset: pattern
discovery, statistical analysis, dimensionality reduction, clustering, and machine-learning
prediction of the recorded severity category.

Built with Python, Streamlit, DuckDB, Pandas, Plotly and scikit-learn, managed with
[`uv`](https://docs.astral.sh/uv/).

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Quick start](#3-quick-start)
4. [Prerequisites (macOS)](#4-prerequisites-macos)
5. [Installation](#5-installation)
6. [Obtaining the dataset](#6-obtaining-the-dataset)
7. [Processing the dataset](#7-processing-the-dataset)
8. [Running the application](#8-running-the-application)
9. [Training models](#9-training-models)
10. [Command-line reference](#10-command-line-reference)
11. [Running tests and lint checks](#11-running-tests-and-lint-checks)
12. [Architecture](#12-architecture)
13. [Project structure](#13-project-structure)
14. [Methodology](#14-methodology)
15. [Limitations](#15-limitations)
16. [Troubleshooting](#16-troubleshooting)
17. [Performance considerations](#17-performance-considerations)
18. [Exporting results](#18-exporting-results)

---

## 1. Problem statement

This project asks two questions about a large-scale record of US traffic accidents:

1. **Descriptive** — when, where, and under what recorded conditions do accidents appear most
   often, and how does the recorded severity category vary across those groups?
2. **Predictive** — can the recorded severity category be anticipated from the circumstances
   known at the moment an accident is first reported?

The second question is the harder one, and its answer is qualified by
[§15 Limitations](#15-limitations). In short: the models are exploratory, they are evaluated on
a held-out slice of the same dataset they were trained on, and they are not validated for
operational traffic-safety use.

---

## 2. Dataset

| Property | Value |
| --- | --- |
| Name | US Accidents (2016–2023) |
| Source | [Kaggle — sobhanmoosavi/us-accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) |
| File | `US_Accidents_March23.csv` |
| Size | ~3 GB, ~7.7 million rows, 46 columns |
| Coverage | February 2016 – March 2023, 49 US states |
| Access | Free Kaggle account required |

### What `Severity` means

`Severity` is a 1–4 scale describing the **reported traffic impact / disruption** of the
accident. It is **not** a measurement of injuries, fatalities, or human harm. Every page in the
application carries this caveat alongside any severity figure.

### Reference figures

The published totals for the original dataset are recorded in `src/smart_traffic/schema.py` as
`REFERENCE_TOTALS`. They are used for exactly one purpose: to let the Data Quality page detect
that a *different* file has been supplied. They are never displayed as this project's results,
and they never substitute for a calculation.

---

## 3. Quick start

If you already have `uv` and Python 3.11+:

```bash
# 1. Sync the environment
uv sync

# 2. Put the dataset here (see §6)
#    data/raw/US_Accidents_March23.csv

# 3. Build the processed dataset (takes a few minutes on the full 3 GB file)
uv run sta build

# 4. Launch the app
uv run streamlit run app.py
```

The app opens at <http://localhost:8501>.

**No dataset yet?** Generate a schema-faithful synthetic file and explore the full application
immediately — no Kaggle account, no 3 GB download:

```bash
uv run sta demo
uv run sta build --source data/synthetic/synthetic_accidents.csv
uv run streamlit run app.py
```

The app detects synthetic data (by its `SYNTH-` ID prefix) and shows a persistent warning on
every page, so its figures can never be mistaken for real results.

---

## 4. Prerequisites (macOS)

| Requirement | Notes |
| --- | --- |
| macOS | Intel or Apple Silicon; tested on arm64 |
| Python 3.11 | `uv` installs it for you — no manual Python install needed |
| `uv` | See below |
| ~6 GB free disk | Room for the 3 GB CSV plus the Parquet output |
| Xcode Command Line Tools | Optional, only for faster file watching: `xcode-select --install` |

### Installing `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell, or add the installer to your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Verify:

```bash
uv --version
```

---

## 5. Installation

```bash
cd smart-traffic-analysis
uv sync --locked     # reproducible install from uv.lock
```

`uv sync` also works, but `--locked` fails if `uv.lock` and `pyproject.toml` have
drifted apart instead of silently re-resolving, so it is the command to use in CI or
when reproducing someone else's environment.

`uv` creates `.venv/`, installs every dependency from `uv.lock`, and installs the project itself
in editable mode. **No virtual environment activation is required** — prefix commands with
`uv run`.

To verify the environment rebuilds cleanly from scratch:

```bash
rm -rf .venv && uv sync --locked
```

### Configuration

Every path can be overridden with an environment variable. Nothing is hardcoded to a specific
machine:

| Variable | Default | Purpose |
| --- | --- | --- |
| `STA_DATA_DIR` | `./data` | Root of the data tree |
| `STA_MODELS_DIR` | `./models` | Persisted model artefacts |
| `STA_ARTIFACTS_DIR` | `./artifacts` | Intermediate artefacts |
| `STA_LOG_DIR` | `./logs` | Rotating log files |
| `STA_PROJECT_ROOT` | auto-detected | Overrides root detection |

---

## 6. Obtaining the dataset

The CSV is **not** bundled with this project — it is too large to distribute in a repository,
and it must not be committed (see `.gitignore`).

1. Sign in to [Kaggle](https://www.kaggle.com).
2. Open the dataset page:
   <https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents>
3. Download `US_Accidents_March23.csv` (~3 GB).
4. Place it at exactly:

```
smart-traffic-analysis/data/raw/US_Accidents_March23.csv
```

A free Kaggle account is sufficient. No API key is needed — the core application never contacts
an external service, and the dataset never leaves your machine.

To confirm placement:

```bash
uv run sta status
```

---

## 7. Processing the dataset

```bash
uv run sta build
```

This converts the CSV into a compressed Parquet table and builds a reproducibility sample. It
reports row counts, invalid values and elapsed time.

### Two interchangeable ingestion strategies

| Method | Command | Use when |
| --- | --- | --- |
| `chunked` *(default)* | `uv run sta build` | You want exact progress and minimal memory |
| `duckdb` | `uv run sta build --method duckdb` | You want the fastest possible conversion |

Both are covered by a test that asserts they produce **byte-identical output**, so choosing one
over the other never changes a result.

### Records are never silently dropped

Invalid values are recorded as flags, not deletions:

| Flag | Meaning |
| --- | --- |
| `Is_Valid_Severity` | `Severity` is one of 1, 2, 3, 4 |
| `Is_Valid_Coordinate` | Coordinates fall inside the USA bounding box |
| `Is_Valid_Record` | Both of the above |

Every record stays in the table. Analytical pages exclude invalid records; the Data Quality
page reports exactly how many that is, and the Data Dictionary documents why.

### Skipping the sample

```bash
uv run sta build --no-sample
```

Useful when you only want descriptive analytics and never intend to train a model.

---

## 8. Running the application

```bash
uv run streamlit run app.py
```

### Pages

| # | Page | Contents |
| --- | --- | --- |
| 1 | Project Overview | Dataset facts, workflow, capabilities, limitations |
| 2 | Data Ingestion & Quality | Validation checks, reference comparison, missing-value profile |
| 3 | Traffic Overview | KPI cards, severity mix, yearly/monthly/hourly counts, top states and cities |
| 4 | Temporal Analysis | Hour/day/month/year patterns, rush hour, weekday, heatmaps |
| 5 | Geographic Analysis | Grid-aggregated density map, hotspots, coordinate validation |
| 6 | Weather & Road Conditions | Coverage warnings, weather distributions, severity bands, road indicators |
| 7 | Severity Analysis | Distributions and within-group rates across nine dimensions |
| 8 | Statistical Analysis | Descriptive statistics, correlation, Mann-Whitney U, chi-square |
| 9 | PCA & Clustering | Principal components, K-Means, agglomerative hierarchical |
| 10 | Severity Prediction | Model training and the single-accident prediction form |
| 11 | Model Evaluation | Comparison table, confusion matrices, per-class reports |
| 12 | Feature Importance | Impurity-based, grouped and permutation importance |
| 13 | Key Findings | Computed findings, executive summary, exports |
| 14 | Methodology & Limitations | How everything works and what it cannot show |

### Global filters

The sidebar filters apply to every analytical page: **Year**, **State**, **City**, **Severity**,
**Month**, and **Day / time category** (Weekday, Weekend, Rush Hour, Non-Rush Hour, Day,
Night). Filters persist as you navigate and can be reset with one click.

### Configuration

```bash
uv run streamlit run app.py --server.port 8502   # different port
uv run streamlit run app.py --server.address 0.0.0.0   # expose on the network
```

The app binds to `127.0.0.1` by default (see `.streamlit/config.toml`); it reads a local
dataset and trains local models, so it has no reason to be network-accessible.

---

## 9. Training models

Models are **never** trained automatically. Training is an explicit action, from the UI or the
CLI:

```bash
uv run sta train                       # all three models, 400k rows
uv run sta train --rows 200000         # smaller sample
uv run sta train --models random_forest
```

Or use the **Severity Prediction → Train models** tab in the app, where you can also choose the
test fraction and seed.

Artefacts are written to `models/`:

```
models/
├── logistic_regression.joblib
├── decision_tree.joblib
├── random_forest.joblib
├── training_summary.json
├── <model>_classification_report.csv
├── <model>_confusion_matrix.csv
└── <model>_feature_importance.csv
```

Each `.joblib` holds the **complete fitted pipeline** — imputation, scaling, one-hot encoding
and the estimator. Calling that pipeline is what guarantees the prediction form uses exactly the
same preprocessing as training.

### Models compared

| Model | Rationale |
| --- | --- |
| Logistic Regression | Interpretable linear baseline |
| Decision Tree | Cheapest model; a single tree's probabilities calibrate well |
| Random Forest | Robust ensemble. Depth-bounded: an unbounded forest scored marginally
  higher on macro F1 but pickled to a 1.5 GB artefact, so depth is capped at 16 |
| Gradient Boosting | `HistGradientBoostingClassifier`, shipped with scikit-learn |

All four use `class_weight="balanced"` and have their per-class decision thresholds
calibrated afterwards. See [§14 Methodology](#14-methodology).

### Measured results

400,000-row sample, seed 42, 320,000 train / 80,000 test, all models on the identical split.

| Model | Macro F1 | Balanced Acc | Accuracy |
| --- | --- | --- | --- |
| **Gradient Boosting** | **0.5791** | 0.6168 | 0.7992 |
| Random Forest | 0.5179 | 0.5532 | 0.7452 |
| Decision Tree | 0.4870 | 0.5872 | 0.7296 |
| Logistic Regression | 0.4011 | 0.4648 | 0.7079 |

Per-class, for the recommended model:

| Severity | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| 1 | 0.509 | 0.609 | 0.555 | 691 |
| 2 | 0.916 | 0.842 | 0.877 | 63,630 |
| 3 | 0.511 | 0.682 | 0.585 | 13,568 |
| 4 | 0.272 | 0.334 | 0.300 | 2,111 |

### These numbers are optimistic, and here is why

**Severity is not recorded consistently across the years.** The share of records at
each level moves far more between years than any weather or road factor does:

| Year | Records | Sev 1 | Sev 2 | Sev 3 | Sev 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2016 | 410,821 | 0.05% | 65.73% | 30.72% | 3.49% |
| 2018 | 893,426 | 0.03% | 64.32% | 32.76% | 2.89% |
| 2020 | 1,178,913 | 2.41% | 79.24% | 15.72% | 2.64% |
| 2021 | 1,563,753 | 0.01% | 88.55% | 9.47% | 1.98% |
| 2022 | 1,762,452 | 2.15% | 90.97% | 4.42% | 2.46% |
| 2023 | 246,633 | 0.00% | 97.14% | **0.00%** | 2.86% |

Severity 3 falls from about a third of all records to **none at all** in 2023, and
severity 1 appears in bulk only in 2020 and 2022. That is a change in reporting
convention, not a change in how severe accidents became.

Two consequences:

1. **A random split flatters the models.** `Year` is a feature, so a model can score
   partly by recognising which regime a record was logged under. The table above is
   therefore the right basis for *comparing* models — all four saw identical rows —
   but it is not evidence of real-world skill.
2. **A temporal holdout is far harsher.** Training on 2016–2021 and testing on
   2022–2023:

   | Model | Macro F1 (random) | Macro F1 (temporal) |
   | --- | ---: | ---: |
   | Gradient Boosting | 0.5791 | **0.3976** |
   | Decision Tree | 0.5076 | **0.3618** |

   Accuracy *rises* to 0.918 on the temporal split while macro F1 halves, because
   2022–2023 is 91.7% severity 2. That gap is the clearest possible illustration of
   why accuracy is the wrong headline metric here.

Reproduce the holdout yourself:

```bash
uv run sta train --temporal --train-years 2016-2021 --test-years 2022-2023
```

What was investigated, and what it rules out:

- **Threshold calibration is not the cause.** Disabling it changes macro F1 from
  0.3618 to 0.3571 — within noise.
- **Dropping `Year` does not help.** It makes matters *worse* (0.3088), so `Year`
  carries genuine signal rather than being a pure regime shortcut.
- **The operating point is the lever.** Thresholds fitted with knowledge of the
  target period reach 0.7235. That is an oracle a deployed model cannot use, but
  calibrating on the most recent *available* period is legitimate and recovers part
  of the gap (0.2152 → 0.2608 on a 2016–2019 → 2021–2023 protocol).

The honest summary: these models predict **how severity was recorded in a given
reporting era**, not how severe a crash physically was. Neither figure should be
presented as a forecast.

**The three headline metrics disagree about which model is best.** Macro F1 and balanced
accuracy reward opposite ends of the same precision/recall dial on a target this imbalanced,
so no model wins on all three. The Model Evaluation page lets you choose which metric selects
the recommendation and states plainly when they disagree, rather than resolving the choice
silently in code.

These figures come from a held-out slice of the *same dataset* the models were trained on. They
measure how well the models reproduce recorded severity patterns in this data — not forecasting
skill against future incidents, which would require temporal validation this project does not
perform.

---

## 10. Command-line reference

```
sta build      Convert the raw CSV into processed Parquet and build the sample
sta sample     Rebuild only the reproducibility sample
sta train      Train the severity models and persist them
sta validate   Print the data-quality report
sta status     Show what is currently loaded
sta demo       Generate a synthetic development dataset
```

Common options:

```bash
uv run sta build --source path/to/other.csv
uv run sta build --method duckdb
uv run sta build --no-sample
uv run sta sample --rows 100000 --seed 7
uv run sta train --rows 500000
uv run sta validate --missing
uv run sta demo --rows 200000 --seed 123
```

Every command writes a detailed trace to `logs/sta.log` while printing an actionable message to
the terminal.

---

## 11. Running tests and lint checks

```bash
uv run pytest            # 352 tests
uv run pytest -v         # verbose
uv run pytest tests/test_modeling.py -k leakage
uv run ruff check .
uv run ruff check . --fix
```

### What the test suite covers

| File | Focus |
| --- | --- |
| `test_data_validation.py` | Required columns, value domains, invalid vs missing, aggregate report |
| `test_preprocessing.py` | Ingestion errors, record retention, **strategy equivalence**, sampling |
| `test_feature_engineering.py` | Mixed timestamps, timezone offsets, rush hour, day/night, duration, leakage |
| `test_aggregation.py` | Filters, counting, **severity-rate denominators**, heatmaps, ordering |
| `test_modeling.py` | Leakage policy, preprocessing, splitting, training, persistence |
| `test_evaluation.py` | Metric correctness, permutation importance, prediction validation, exports |
| `test_tuning.py` | Threshold calibration: never degrades validation score, valid estimator |
| `test_streamlit_app.py` | Every page renders, with and without a dataset |

Tests run against small synthetic fixtures in a temporary directory. **The 3 GB CSV is never
required and never downloaded.** Every test is independent of the others and of the real
`data/` folder.

---

## 12. Architecture

```
        US_Accidents_March23.csv  (~3 GB, 46 columns)
                     │
                     ▼
        ┌────────────────────────┐
        │  ingestion.py          │  chunked Pandas reads OR a single DuckDB
        │  (two strategies)      │  push-down pipeline — identical output
        └───────────┬────────────┘
                    ▼
        accidents.parquet  (56 columns, one row per record)
        + Is_Valid_Severity / Is_Valid_Coordinate / Is_Valid_Record
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  db.py (DuckDB)         sample parquet
  aggregates run in SQL    hash(ID, seed) — reproducible
        │                        │
        ▼                        ▼
  aggregation.py          PCA · clustering · correlation
  analytics.py            modelling · permutation importance
  geospatial.py
        │
        ▼
  visualizations.py ──► Streamlit pages
```

### Design decisions worth knowing

**Everything aggregates in SQL.** A chart costs one scan against the Parquet file rather than a
7.7M-row Pandas materialisation. This is the single largest reason the app stays responsive.

**Maps are grid-aggregated.** 7.7 million markers cannot be rendered in a browser. Coordinates
are binned into cells in SQL, returning a few thousand rows that communicate the same density
pattern.

**Sampling is deterministic.** Expensive work selects rows with `hash(ID, seed)` rather than
`RANDOM()`. The same seed always picks the same rows, so reported metrics are comparable across
runs.

**Timestamps keep local wall-clock time.** The source mixes formats and a minority of rows carry
a UTC offset. Offsets are *stripped*, not converted — converting would silently shift those
rows' hour of day, and hour-of-day is a primary analytical axis here.

**Leakage is enforced in code.** `LeakagePolicy.validate()` raises before any model is fitted if
a feature list contains `End_Time`, duration, the target or the identifier.

**Findings are computed, never hardcoded.** Every claim on the Key Findings page comes from a
live query, so a different dataset produces different findings rather than stale ones.

---

## 13. Project structure

```
smart-traffic-analysis/
├── app.py                     Entry point and landing page
├── pyproject.toml             Dependencies, Ruff and Pytest configuration
├── uv.lock                    Locked dependency graph (do not edit by hand)
├── .python-version            3.11
├── .streamlit/config.toml     Theme and server defaults
│
├── pages/                     Streamlit multipage app
│   ├── 1_…?  →  2_Data_Quality.py … 14_Methodology.py
│
├── src/smart_traffic/         Application package
│   ├── config.py              Paths and tunable analysis settings
│   ├── schema.py              Column knowledge and reference figures
│   ├── logging_config.py      Structured logging with path redaction
│   ├── db.py                  DuckDB query layer
│   ├── ingestion.py           CSV → Parquet, two strategies
│   ├── validation.py          Data quality checks
│   ├── feature_engineering.py Derived features and the leakage guard
│   ├── aggregation.py         Filtered SQL aggregates and severity rates
│   ├── analytics.py           KPIs and computed findings
│   ├── statistics.py          Correlation and hypothesis tests
│   ├── dimensionality_reduction.py  PCA
│   ├── clustering.py          K-Means and agglomerative hierarchical
│   ├── sampling.py            Reproducible sampling
│   ├── modeling.py            Pipelines, training, persistence
│   ├── evaluation.py          Metrics, confusion matrices, permutation importance
│   ├── tuning.py             Decision-threshold calibration
│   ├── prediction.py          Prediction input validation and inference
│   ├── geospatial.py          Density grids and hotspots
│   ├── visualizations.py      Plotly figure builders
│   ├── reporting.py           Exports and provenance
│   ├── synthetic.py           Schema-faithful development data
│   ├── ui.py                  Shared Streamlit components
│   └── cli.py                 Command-line interface
│
├── scripts/
│   └── experiment.py          Measured model-improvement harness
├── tests/                     338 tests
├── data/
│   ├── raw/                   ← place US_Accidents_March23.csv here
│   ├── processed/             Generated Parquet + DuckDB catalogue
│   ├── analytics/             Exported reports and bundles
│   └── synthetic/             Generated development data
├── models/                    Trained pipelines and reports
├── artifacts/                 Intermediate outputs
└── logs/                      Rotating log files
```

---

## 14. Methodology

### Derived features

| Feature | Definition |
| --- | --- |
| `Year`, `Month`, `Month_Name`, `Day` | Calendar components of `Start_Time` |
| `Hour` | Local clock hour, 0–23 |
| `DayOfWeek` (0 = Monday), `Day_Name` | Weekday index and name |
| `Is_Weekend` | True for Saturday and Sunday |
| `Is_Rush_Hour` | **07:00–09:59 and 16:00–18:59**, both endpoints inclusive |
| `Day_Night` | From `Sunrise_Sunset`: Day, Night, or **Unknown** |
| `Is_Night` | 1 for Night, 0 for Day, **missing for Unknown** |
| `Duration_Minutes`, `Duration_Hours` | `End_Time − Start_Time`; negative values become missing |

`Unknown` is preserved as its own category rather than folded into Day or Night — imputing it
would invent data where the source was ambiguous.

Rush hour is configurable via `AnalysisSettings.rush_hour_morning` / `rush_hour_evening`. The
Pandas and SQL ingestion paths both read the same setting, so they cannot disagree.

### Prediction scenario

> **Predict the recorded severity category of an accident using only information available when
> the accident is first reported.**

This definition is what excludes `End_Time` and everything derived from it. An accident's
duration is not known until it has ended, so a model using duration would score well while being
unable to handle a real, still-in-progress incident.

| Excluded | Reason |
| --- | --- |
| `ID` | An identifier carries no predictive information and invites leakage |
| `Severity` | The prediction target |
| `End_Time` | Only known after the incident ends |
| `Duration_Minutes` / `Duration_Hours` | Derived from `End_Time` |
| `Description`, `Street` | Free text; no extraction pipeline is claimed |
| `Start_Lat`, `Start_Lng` | Location precision is not asserted to be available at first report |

The exclusion is enforced by `LeakagePolicy.validate()`, which raises before any model is fitted.

### Preprocessing

Imputation and scaling live **inside** the scikit-learn `Pipeline`, so they are fitted on the
training split only and never see the test set.

- Numeric: median imputation, then standardisation (skipped for the Decision Tree, which does
  not need it and only runs slower with it).
- Categorical: most-frequent imputation, then one-hot encoding with rare levels folded together
  and a cap of 60 categories per variable — this bounds memory on high-cardinality fields like
  `Weather_Condition`.

### Class imbalance

Severity 2 holds 79.7% of the dataset. A classifier that always predicted severity 2 would score
79.5% accuracy and macro F1 of 0.22 — competitive-looking numbers for a model that is completely
useless. Three mechanisms guard against that:

**1. Balanced class weights.** Every model trains with `class_weight="balanced"`, which is what
gives it any recall on the rare categories.

**2. Decision-threshold calibration.** Class weights change what a model *learns*; thresholds
change what it *decides*. Conflating them is what makes the tuning problem hard, so this project
separates them. The pipeline is fitted on 75% of the training split, and a per-class multiplier
applied to the predicted probabilities is chosen on the held-out 25% by coordinate ascent on
macro F1. The test split is never touched by the tuner.

Measured on a fixed 400,000-row split, every configuration that maximised macro F1 did so by
nearly abandoning severity 4:

| Variant                                | macro F1 | recall S4 |
| -------------------------------------- | -------- | --------- |
| Balanced weights, plain argmax        | 0.370    | 0.826     |
| Unconstrained Random Forest           | 0.544    | 0.127     |
| `log1p` class weights                 | 0.525    | 0.164     |
| **Balanced + calibrated thresholds**  | **0.543**| **0.454** |

(The threshold-tuning rows above were measured on the Decision Tree during the sweep; the
final pipeline applies the same calibration to all four models.)

Threshold calibration reaches the same macro F1 while keeping severity-4 recall at roughly
3.5× the unconstrained alternative. The multipliers are persisted with the model and shown on
the Model Evaluation page.

**3. A degeneracy guard.** A model that fails to predict one severity category at all is
excluded from selection, however well it scores. This is checked because macro F1 *and*
balanced accuracy can both be satisfied by a collapsed model: predicting only the majority class
yields ~0.80 accuracy, exactly 0.25 balanced accuracy (one class at 1.0, three at 0.0), and
~0.22 macro F1 — which can beat a model genuinely discriminating between all four. Counting the
distinct labels emitted is the only reliable discriminator.

### What did not help

Two techniques were implemented and measured, and are **not** in the final pipeline:

- **Cyclical sin/cos encoding** of hour, month and day-of-week. Zero measurable change
  (macro F1 0.3701 → 0.3701). The reason is visible in the importance table: those features
  carry 0.015, 0.010 and 0.100 respectively, while `Year` carries 0.373. Trees cannot exploit
  the wrap-around, but these features barely matter here, so there is nothing to exploit.

- **Softer class weights** (sqrt, log1p, quarter). They raise macro F1 but destroy recall on
  the rarest category — see the table above. Threshold calibration dominates them.

The experiment harness that produced these measurements is kept at
`scripts/experiment.py`, so the numbers can be reproduced:

```bash
uv run python scripts/experiment.py weights hyper cyclic threshold boost
```

### Reproducibility

| Setting | Default |
| --- | --- |
| Random seed | 42 |
| Decision-threshold calibration | enabled by default |
| Rush hour | 07:00–09:59, 16:00–18:59 |
| Training sample cap | 400,000 rows |
| Clustering sample cap | 25,000 rows |
| PCA sample cap | 40,000 rows |
| Hierarchical hard cap | 5,000 rows |
| Valid latitude | 18.0 to 72.0 |
| Valid longitude | −180.0 to −40.0 |

Sampling uses `hash(ID, seed)` rather than `RANDOM()`, so rerunning the pipeline on unchanged
data reproduces the same metrics exactly.

---

## 15. Limitations

**Severity is traffic impact, not harm.** `Severity` records the reported disruption level. No
conclusion about human consequence can be drawn from this dataset at all.

**Counts are frequency, not risk.** These are counts of *recorded accidents*. The dataset
contains no traffic-volume or exposure data, so frequency cannot be converted into a rate of
accidents per unit of driving. A state with more recorded accidents may simply have more drivers,
more reporting coverage, or both.

**No causal claims.** Every relationship shown is an observational association in recorded data.
Unmeasured confounding is not controlled for.

**Reporting bias.** The dataset reflects what was reported to the source. Reporting practice
varies by jurisdiction and over time, which is a plausible contributor to the year-on-year
increase and is not separable from genuine change.

**The final year is incomplete.** The last calendar year present covers only part of the year.
It is excluded from full-year comparisons, and every per-year chart marks it.

**Class imbalance sets a low ceiling.** With under 1% of records at severity 1, per-class recall
for the rarest category will be low for every model tried. This is a property of the data, not
a fixable modelling defect.

**Sampled analysis carries sampling error.** PCA, clustering, correlation and training use
bounded samples. Figures computed on a sample are labelled as such wherever they appear.

**Models are exploratory.** They are trained and evaluated on one historical dataset. They are
not validated for real-world traffic-safety use, they are not calibrated, and they should not
inform operational decisions.

**The reported metrics do not survive a temporal holdout.** Training on 2016-2021 and testing
on 2022-2023 drops macro F1 from 0.5791 to 0.3976 for the gradient-boosting model. The cause is
that severity prevalence shifts drastically across the years — severity 3 goes from about a
third of all records to none at all in 2023 — so a model partly learns the reporting convention
of a period rather than the severity of an incident. This project now supports the check
(`uv run sta train --temporal --train-years 2016-2021 --test-years 2022-2023`); the full
investigation, including what was ruled out, is in
[These numbers are optimistic](#these-numbers-are-optimistic-and-here-is-why).

**Feature importance is not causation.** A high importance score means the model *used* a
feature, not that the feature *causes* severity. Correlated features split importance between
them arbitrarily, and one-hot encoding spreads a single categorical variable across dozens of
columns that each look unimportant alone.

---

## 16. Troubleshooting

**"No processed dataset is loaded"**
Run `uv run sta build`. If the CSV is not present, follow [§6](#6-obtaining-the-dataset), or
generate synthetic data with `uv run sta demo`.

**`Source file not found`**
The path in the error is the expected location. Confirm with `uv run sta status`.

**`... is missing required column(s)`**
The CSV is not the original schema. Required columns are `ID`, `Severity` and `Start_Time`.

**`No models have been trained yet`**
Run `uv run sta train`, or use the **Severity Prediction → Train models** tab. The prediction
and evaluation pages stay disabled until a model exists.

**`The saved artefact could not be loaded`**
The `.joblib` file is truncated or was built with a different library version. Delete
`models/*.joblib` and retrain.

**The map is blank**
Records with out-of-range coordinates are excluded from maps by design. Check the coordinate
validation table on the Geographic Analysis page, and confirm no filter is excluding everything.

**`Ingestion produced an empty file`**
Usually a disk-space problem. Check with `df -h`. The full pipeline needs roughly 6 GB free.

**Out of memory during training or clustering**
Lower the sample size in the UI, or `uv run sta train --rows 100000`.

**Port 8501 already in use**
`uv run streamlit run app.py --server.port 8502`

**Watchdog warning on startup**
Cosmetic. Run `xcode-select --install` for faster file watching.

**Checking what went wrong**
`logs/app.log` and `logs/sta.log` hold the full trace. Log records have home-directory paths
replaced with `~` so logs are safe to share.

---

## 17. Performance considerations

| Operation | Approach | Why |
| --- | --- | --- |
| Ingestion | Chunked reads or DuckDB push-down | Bounded memory regardless of file size |
| Charts | SQL aggregates over Parquet | One scan, no materialisation |
| Maps | Grid aggregation in SQL | Thousands of cells, not millions of points |
| PCA | Capped reproducible sample | O(n·p) in memory |
| Hierarchical clustering | Hard 5,000-row cap | Requires an O(n²) distance matrix |
| K-Means | Capped sample, cached results | Silhouette search is expensive |
| Training | Capped sample, explicit trigger | Never runs on app launch |
| Repeated queries | `@st.cache_data` with TTL | Filter option lookups do not rescan |

**Memory.** Peak usage is bounded by the ingestion chunk size, not the file size. DuckDB is
configured with a memory limit and will spill to disk rather than fail.

**Expected runtime** on the full 7.7M-row dataset (Apple Silicon, 16 GB):

| Step | Time |
| --- | --- |
| `sta build` (chunked) | ~6–10 min |
| `sta build --method duckdb` | ~2–4 min |
| `sta validate` | ~10 s |
| `sta train` (400k rows, 3 models) | ~5–10 min |

---

## 18. Exporting results

**From the app.** Every analytical page has CSV download buttons. The Key Findings page can also
build a **zip bundle** containing every supporting table plus a `provenance.json`, and a
Markdown report.

**Provenance is always included.** Exports record the dataset, its row count, its coverage
period, and the filters active at export time. A CSV without that context is not interpretable.

**From the CLI:**

```bash
uv run sta validate > validation_report.txt
```

Exports are written to `data/analytics/`. Nothing is uploaded anywhere — all output is local.

---

## Licence and attribution

Dataset © the Kaggle publisher, used under the terms published on the
[Kaggle dataset page](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents). This
repository contains only analysis code.
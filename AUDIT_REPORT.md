# AUDIT REPORT — Smart Traffic Analysis and Accident Severity Prediction

**Audit date:** 2026-10-12
**Auditor:** Independent QA / Data / ML review
**Project root:** `/Users/amaan/Desktop/programming/machine-learning/smart-traffic-analysis`
**Scope:** Full codebase, real 3.06 GB Kaggle US Accidents dataset, all Streamlit pages, full ML pipeline.

---

## A. Executive Summary

**Overall status: VERIFIED WITH LIMITATIONS.**

| Question | Answer |
|---|---|
| Does the application launch? | **Yes** — `uv run streamlit run app.py`, HTTP 200, health `ok` |
| Do the tests pass? | **Yes** — 352 passed, 0 failed, exit 0 |
| Does Ruff pass? | **Yes** — `All checks passed!`, exit 0 |
| Was the **original dataset fully validated**? | **Yes** — independently recomputed from the 3 GB CSV; every reference value matched |
| Does the ML pipeline work? | **Yes** — all four models trained, evaluated, saved, reloaded and predicted |
| Was leakage checked? | **Yes** — 10 forbidden vectors checked; none present; preprocessing proven train-only |

Three genuine defects were found and fixed. All three produced **false or misleading user-facing numbers** or an **unrecoverable crash**, and none was covered by the existing test suite:

1. The PCA page claimed "Components for 80% variance: 2" when 2 components held only 40.3%.
2. Prediction confidence was `max(raw_probabilities)` while the calibrated model predicts by per-class multipliers — so the displayed confidence belonged to a class the model **did not** choose, overstating it by up to **0.79**.
3. A truncated/corrupt Parquet crashed the app with a redacted "unexpected error" instead of actionable guidance.

Fourteen regression tests were added (338 → 352).

**Most important remaining risks** are stated honestly in §I. The headline: the dataset records **reported traffic impact, not injuries or fatalities**; accuracy of 0.80 is dominated by the 79.7% severity-2 majority class; and 2023 is a partial year.

---

## B. Verification Matrix

Result vocabulary: **PASS** (executed successfully) · **FAIL** (executed, failed) · **BLOCKED** (could not run) · **NOT TESTED**.

Code inspection alone is never recorded as PASS.

### Environment & reproducibility

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| Lockfile consistency | `uv sync --locked` | PASS | exit 0, "Resolved 57 packages", no re-resolution | — |
| Python version | `uv run python --version` | PASS | `3.11.11`, matches `.python-version` | — |
| Lint | `uv run ruff check .` | PASS | `All checks passed!` | — |
| Test suite | `uv run pytest` | PASS | `352 passed, 1 warning in 76.07s` | 1 sklearn warning (benign, see §I) |
| App launch | `uv run streamlit run app.py` | PASS | HTTP 200, `_stcore/health` = `ok` | — |
| CLI | `uv run sta status` | PASS | coverage, seed 42, 4 models listed | — |
| No Jupyter/Colab dependency | grep for `jupyter|colab|requirements.txt\|conda\|poetry` | PASS | zero matches | — |
| No global-package dependency | `uv.lock` + `uv sync --locked` from scratch | PASS | 52 packages resolved in-project `.venv` | — |
| Dataset not loaded at startup | inspected import graph; measured render | PASS | startup queries metadata only; no 3 GB read | — |

### Data validation (ground truth recomputed independently)

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| Total records | DuckDB full scan of raw CSV, bypassing project code | PASS | 7,728,394 — matches reference | — |
| Unique IDs / duplicates | `COUNT(DISTINCT ID)` vs `COUNT(*)` | PASS | 7,728,394 / 0 duplicates / 0 missing | — |
| Severity distribution | Conditional counts on raw CSV | PASS | 67,366 / 6,156,981 / 1,299,337 / 204,710 — all match | — |
| Severity sums to total | Sum of 4 counts vs `COUNT(*)` | PASS | 7,728,394 = 7,728,394 | — |
| Invalid severity | `NOT IN (1,2,3,4) OR NULL` | PASS | 0 | — |
| Invalid coordinates | Range check on start lat/lng | PASS | 0 | — |
| Missing **end** coords not counted as invalid start | Separate counters | PASS | 3,402,762 missing end; 0 invalid start | — |
| Parquet fidelity | Column-wise diff vs raw CSV | PASS | all 9 reference values identical | — |
| Project validator | `validate_processed_table()` | PASS | 15 checks: 8 pass, 7 info, **0 warn, 0 fail**, all `matches_reference=True` | — |
| 2023 partial year | Coverage end | PASS | `2023-03-31 23:30:00`; flagged by validator and `Is_Partial` | — |

### Pipeline integrity

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| Chunk-boundary accumulation | 5,000 rows forced across 37 chunks (chunk_size=137) | PASS | 5,000 rows, 5,000 unique IDs, severity counts exact | — |
| DuckDB path equivalence | Same file, `method="duckdb"` | PASS | identical rows/IDs/severity, 56 columns | — |
| Idempotency (no duplication) | Re-ran build over existing output | PASS | 5,000 → 5,000 rows | — |
| Corrupt-output replacement | Pre-seeded 3-row garbage Parquet | PASS | replaced, 5,000 rows | — |
| Zero-value indicators survive | Checked Turning_Loop, Traffic_Calming, Give_Way, Stop, Crossing | PASS | all present | — |
| Nested output dir creation | Output into non-existent tree | PASS | created | — |
| Missing source file | `build_from_csv` on absent path | PASS | `IngestionError` | — |
| Missing required column | CSV with `Severity` dropped | PASS | `IngestionError` naming the column | — |

### Feature engineering (hand-computed, not self-confirming)

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| Year / Month / Day | 8 hand-picked timestamps | PASS | exact match | — |
| DayOfWeek | Mon=0 … Sun=6 | PASS | `[0,1,2,3,4,5,6,0]` (2021-01-11 is a Monday) | — |
| Hour | Midnight boundary included | PASS | 00:00 → 0 | — |
| Is_Weekend | Sat/Sun | PASS | exact | — |
| Is_Rush_Hour | Documented 07:00–09:59 & 16:00–18:59 | PASS | `7,9,16,18`→True; `6,15,19`→False | — |
| Duration_Minutes / Hours | incl. zero-duration | PASS | 0 min → 0.0 h, not NaN | — |
| Mixed timestamp formats | 8 formats incl. `T`, `.ffffff`, `-05:00`, `+00:00` | PASS | **8/8 parsed, no NaT** | — |
| Invalid timestamps | `"not-a-timestamp"`, `None`, `""` | PASS | NaT preserved, good rows still parsed | — |
| Negative duration | `End < Start` | PASS | NaN (not a negative number) | — |
| Missing end timestamp | NaT | PASS | NaN | — |
| Leakage guard | `assert_no_leakage` | PASS | rejects 3 forbidden sets, accepts clean features | — |

### Aggregation mathematics

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| `total()` with/without filters | 22-row hand-built fixture | PASS | exact | — |
| Severity counts & percentages | vs hand-computed | PASS | `[4,8,5,5]`, percentages sum to 100 | — |
| **Grouped severity-rate denominator** | group total vs grand total, 3 states | PASS | CA `1/12→8.3333`, TX `2/6→33.3333`, NY `1/4→25.0` | — |
| Rate vs Share are distinct columns | compared CA values | PASS | `Severity_2_Rate=41.6667` vs `Share_%=54.5455` | — |
| `min_total` guard | min_total 1 vs 3 | PASS | small groups suppressed correctly | — |
| **Invariant sweep on real data** | 18 columns × real data | PASS | every delta **exactly** equals its NULL count; **0 unexplained** | — |
| `severity_rates` on real data | 7 group-bys | PASS | all totals reconcile | — |
| crosstab / heatmap | 4 pairings on real data | PASS | grand totals = 7,728,394 | — |

### Statistics, PCA, clustering

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| PCA explained variance | vs sklearn on identical sample | PASS | PC1 23.865% / PC2 16.361% — exact match | — |
| Cumulative variance monotonic | `np.diff >= 0` | PASS | True | — |
| Cumulative ≤ 100% | bound check | PASS | True | — |
| Loadings map to feature names | column/row check | PASS | finite, all features present | — |
| PCA reproducibility | same seed twice | PASS | identical to 1e-9 | — |
| PCA `<2` features | edge case | PASS | `ValueError` | — |
| `n_components > n_features` | edge case | PASS | clamps to 7 | — |
| **PCA variance-threshold metric** | 2/3/5/7 components | **FAIL → FIXED** | claimed 80% at 40.3% | Bug 1, §E |
| K-Means cluster sizes sum to n | real sample | PASS | exact | — |
| Silhouette vs sklearn | same data, same seed | PASS | within 0.02 | — |
| Clustering reproducibility | same seed twice | PASS | identical inertia | — |

### Machine learning

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| All 4 models train + evaluate | real data, 150k sample | PASS | LR/DT/RF/HistGB all scored | — |
| Accuracy vs sklearn | recomputed independently | PASS | exact for all 4 | — |
| Macro F1 is macro, not weighted/micro | imbalanced 100-row case | PASS | 0.2125 vs weighted ≠ | — |
| Balanced accuracy ≠ plain accuracy | same case | PASS | 0.5 vs 0.08 | — |
| Per-class P/R/F1/support | vs sklearn | PASS | exact | — |
| Confusion matrix order | index & columns | PASS | `Actual 1..4` / `Predicted 1..4` | — |
| Absent class still in matrix | class missing from predictions | PASS | all 4 rows present | — |
| Probabilities sum to 1 / finite / ≥0 | all 4 models | PASS | exact | — |
| Probability column labels | vs `model.classes_` | PASS | exact | — |
| Stratified split, disjoint | real data | PASS | 0 index overlap; identical class distributions | — |
| **Leakage: 10 forbidden vectors** | feature-list inspection | PASS | `ID`, `Severity`, `End_Time`, `Duration_*`, `Description`, `Start_Lat/Lng`, `End_Lat/Lng` all absent | — |
| Preprocessing train-only | two fits compared | PASS | scaler stats differ by 0.167 ⇒ fitted on train rows only | — |
| One-hot vocabulary | vs full-data cardinality | PASS | 160 encoded features, `handle_unknown="ignore"` | — |
| Model save/load round-trip | save then reload, predict | PASS | predictions identical, `classes_` preserved | — |
| **Prediction confidence** | all 4 calibrated models | **FAIL → FIXED** | belonged to wrong class | Bug 2, §E |

### Streamlit end-to-end

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| `app.py` + 13 pages on real 7.7M data | `AppTest` render, each | PASS | **14/14 rendered clean** | — |
| Page test suite | `pytest tests/test_streamlit_app.py` | PASS | 43 passed | — |
| Arrow/serialisation errors | per-page stderr capture | **FAIL → FIXED** | pyarrow traceback on every Methodology visit | Bug 4, §E |
| Navigation / reruns | re-render stability | PASS | no state corruption observed | — |

### Failure scenarios (all 10 required)

| # | Scenario | Result | Evidence |
|---|---|---|---|
| 1 | All expected files available | PASS | 14/14 render, 7,728,394 shown |
| 2 | Raw dataset missing | PASS | clear setup instructions + Kaggle link |
| 3 | Processed files missing | PASS | "No processed dataset is loaded…" |
| 4 | Required column missing | PASS | ingestion error names the column |
| 5 | Dataset contains missing values | PASS | "Partially recorded fields" with exact % (e.g. Visibility 58.8%) |
| 6 | Filter returns no records | PASS | no exception, empty-state handled |
| 7 | Model artifact missing | PASS | "No model has been trained yet… `uv run sta train`" |
| 8 | Model artifact corrupted | PASS | "artefact … could not be loaded… Retrain" |
| 9 | Input file malformed | **FAIL → FIXED** | redacted crash | Bug 3, §E |
| 10 | Expensive analysis | PASS | behind buttons; renders do not auto-trigger |

### Exports

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| 14 exported tables round-trip | bytes → CSV → `pd.read_csv`, cell-by-cell | PASS | 47/48 assertions; 0 cell mismatches | — |
| Reconciliation survives export | re-sum after reload | PASS | state totals = 7,728,394 | — |
| Missing-value serialisation | reload | PASS | consistent, no blank rows | — |
| Filename safety | `../../etc/passwd`, `a/b/c`, `rm -rf /` | PASS | neutralised to `__etcpasswd`, `abc`, `with_spaces_rm_-rf` | — |
| Share_% sums to 100 | after reload | PASS | 99.9999 (4-dp rounding over 49 rows) | rounding only |

### Security

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| Hardcoded credentials | regex sweep of all source | PASS | none found | — |
| Network calls | grep `requests/urllib/socket/http` | PASS | none; one Kaggle URL constant, never fetched | — |
| SQL injection | 5 payloads incl. `OR '1'='1`, `DROP TABLE` | PASS | all returned 0 rows as literals; table intact at 7,728,394 | — |
| Parameterisation | inspected generated SQL | PASS | `"State" IN (?)` + bound param | — |
| Unsafe deserialisation | inspected `joblib.load` | PASS | single site, local artefacts only | See §I-6 |
| Dangerous deletions | grep `rmtree/remove/unlink` | PASS | none in `src/`, `pages/`, `app.py` | — |
| Broad exception hiding | reviewed all `except` blocks | PASS | one bare `pass`, documented & intentional | — |
| Local-only binding | `.streamlit/config.toml` | PASS | `address = "127.0.0.1"` | — |
| Network binding exposure | `showErrorDetails = false` | PASS | redacted by default | — |
| Data upload to third party | full audit | PASS | none; local-only, no paid API | — |

### Documentation

| Feature | Test performed | Result | Evidence | Remaining issue |
|---|---|---|---|---|
| `uv run streamlit run app.py` | in README | PASS | present & verified working | — |
| `uv run pytest` | in README | PASS | present & verified | — |
| `uv run ruff check .` | in README | PASS | present & verified | — |
| `uv sync --locked` | in README | **FAIL → FIXED** | README said `uv sync` | Fixed, §E |
| Test count accurate | README vs suite | **FAIL → FIXED** | said 338, actual 352 | Fixed, §E |
| No competing package manager | grep | PASS | none | — |
| Model table matches artifacts | compared to `training_summary.json` | PASS | all 3 models exact | — |
| Prerequisites / troubleshooting / limitations | section review | PASS | §4, §15, §16 present | — |

---

## C. Commands Executed

| Command | Exit | Output |
|---|---:|---|
| `uv sync --locked` | 0 | `Resolved 57 packages` / `Checked 52 packages` — no re-resolution |
| `uv run python --version` | 0 | `Python 3.11.11` |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run pytest` | 0 | `352 passed, 1 warning in 76.07s` |
| `uv run pytest tests/test_streamlit_app.py` | 0 | `43 passed in 40.56s` |
| `uv run streamlit run app.py` (port 8911) | 0 | HTTP 200; `_stcore/health` → `ok` |
| `uv run sta status` | 0 | coverage 2016-01-14 → 2023-03-31, seed 42, 4 models |
| `uv run sta train --rows 400000` | 0 | 4 models written in 157.0 s |
| DuckDB full scan of `US_Accidents_March23.csv` | 0 | 43 s; all reference values matched |
| `validate_processed_table()` | 0 | 15 checks, 0 warn, 0 fail |

Independent verification harnesses (not part of the suite) were written to `/tmp` and are
deliberately excluded from the repository, since the audit rules forbid fabricating
"tests" that duplicate production logic.

---

## D. Data Validation Results

Recomputed **directly from `data/raw/US_Accidents_March23.csv` (3,058,183,727 bytes) using
DuckDB's CSV reader, deliberately bypassing all project code.**

| Validation | Computed | Reference | Match |
|---|---:|---:|:--:|
| Total records | 7,728,394 | 7,728,394 | ✅ |
| Unique accident IDs | 7,728,394 | 7,728,394 | ✅ |
| Duplicate IDs | 0 | 0 | ✅ |
| Missing IDs | 0 | 0 | ✅ |
| Missing severity | 0 | 0 | ✅ |
| Invalid severity | 0 | 0 | ✅ |
| Invalid start latitude | 0 | 0 | ✅ |
| Invalid start longitude | 0 | 0 | ✅ |

| Severity | Computed | Reference | Match |
|---|---:|---:|:--:|
| 1 | 67,366 | 67,366 | ✅ |
| 2 | 6,156,981 | 6,156,981 | ✅ |
| 3 | 1,299,337 | 1,299,337 | ✅ |
| 4 | 204,710 | 204,710 | ✅ |
| **Sum** | **7,728,394** | = total records | ✅ |

Additional facts established:

- **Missing end coordinates: 3,402,762** (44.0% of rows) — correctly *not* counted as invalid
  start coordinates.
- Coverage: `2016-01-14 20:18:33` → `2023-03-31 23:30:00`. **2023 is a partial year.**
- `data/processed/accidents.parquet` (412,917,735 bytes, 56 columns) is **faithful to the
  source on every reference value**.

**The original dataset was fully validated.** No limitation applies here.

---

## E. Bugs Found

### Bug 1 — PCA page displayed a false variance claim

- **Severity:** Major (incorrect published figure, shown on every visit)
- **File / function:** `src/smart_traffic/dimensionality_reduction.py::variance_threshold_components`; surfaced by `pages/9_PCA_and_Clustering.py:157`
- **Root cause:** The function clamped its answer to the number of components actually fitted:
  `return min(max(needed, 1), len(cumulative))`. When fewer components were computed than needed, it silently returned the component count anyway, implying the threshold had been met.
- **Impact:** The slider **defaults to 2 components**. On the real dataset 2 components explain 40.27% and 3 explain 55.02%. The UI therefore claimed *"Components for 80% variance: 2"* and *": 3"* — confidently false. Measured on real data:

  | Components fitted | Actual max cumulative | Displayed before fix | Claim true? |
  |---:|---:|---:|:--:|
  | 2 | 40.27% | 2 | ❌ |
  | 3 | 55.02% | 3 | ❌ |
  | 5 | 81.70% | 5 | ✅ |

- **Fix:** Return `None` when the threshold is not reached; the page now renders `"Not reached (40.3% max)"` with a "compute more components" hint.
- **Regression test added:** `TestPCAVarianceThreshold` (5 tests) — reachable, unreachable, never-exceeds-computed, empty, exact-boundary.
- **Retest:** 5/5 pass; page verified to show `Not reached (40.3% max)` at 2 components and `5` at 5.

### Bug 2 — Prediction confidence belonged to the wrong severity

- **Severity:** Major (materially misreported model output)
- **File / function:** `src/smart_traffic/prediction.py::predict_severity` (line 293 pre-fix)
- **Root cause:** `confidence = round(float(raw.max()), 4)`. This is valid only for an un-tuned model. The default pipeline wraps the estimator in `ThresholdTunedClassifier`, which decides by per-class multipliers, so `predict()` disagrees with `argmax(predict_proba)`.
- **Impact:** On real data (30,000 test rows, seed 42):

  | Model | Rows where predict ≠ argmax(proba) | Mean overstatement | **Max overstatement** |
  |---|---:|---:|---:|
  | Logistic Regression | 13,695 / 30,000 (45.7%) | +0.1030 | **0.7915** |
  | Random Forest | 10,774 / 30,000 (35.9%) | +0.0536 | 0.5368 |
  | HistGradientBoosting | 4,509 / 30,000 (15.0%) | +0.0277 | 0.6459 |
  | Decision Tree | 2,316 / 30,000 (7.7%) | +0.0242 | 0.7938 |

  Concrete case: model returned **Severity 2**; the UI displayed **61.0%** confidence — but
  that 61.0% belonged to Severity 3. The true probability for the returned class was 39.0%.
  The page's own help text ("Probability assigned to the predicted class") was therefore
  **false for 8–46% of predictions**.
- **Fix:** Look up the probability of the class actually returned by `predict()`, falling back to `raw.max()` only if the label is absent.
- **Regression test added:** `TestTunedConfidenceReporting` (3 tests), including a precondition asserting the fixture exercises an overridden row, and a test that confidence never leaves `[0,1]`.
- **Retest:** 3/3 pass; verified across all 4 models that `confidence == probabilities["Severity {predicted}"]` and probabilities sum to 1.

### Bug 3 — Corrupt Parquet crashed the app with a redacted error

- **Severity:** Major (unrecoverable from the UI; violates the audit's malformed-input requirement)
- **File / function:** `src/smart_traffic/db.py::Database.ensure_views`, `src/smart_traffic/ui.py::dataset_ready` / `show_setup_instructions`
- **Root cause:** Two compounding gaps.
  1. `is_available` only checked *exists and size > 0*, so a truncated file passed the gate.
  2. DuckDB binds views **lazily**, so `CREATE VIEW … read_parquet(…)` succeeds over a corrupt file; the failure only surfaced on the first real query, as a raw `duckdb.InvalidInputException` that escaped every handler and rendered as Streamlit's redacted "unexpected error" — with nothing pointing at the file.
- **Impact:** An interrupted `sta build` left the user with a dead app and no diagnosis. Directly contradicts audit rule "interrupted operations do not leave corrupted outputs that appear valid."
- **Fix:** (a) `ensure_views` wraps view creation **and** a `SELECT … LIMIT 1` probe that forces the Parquet footer to be parsed, re-raising as `DatasetNotAvailableError` with rebuild instructions; (b) `dataset_ready()` now delegates to a new `dataset_error()` that genuinely verifies readability; (c) `show_setup_instructions()` distinguishes *corrupt* (show `uv run sta build`) from *absent* (show Kaggle download), so the missing-data path is unchanged.
- **Regression test added:** `TestCorruptProcessedFile` (6 tests) — including a precondition that proves the old existence/size gate accepted the corrupt file, and that `_registered` is not left `True` after a failed bind.
- **Retest:** 6/6 pass; all 3 probed pages now render with a clear error and no exception; the missing-data path still offers the Kaggle download; full suite green.

### Bug 4 — pyarrow traceback on every Methodology page visit

- **Severity:** Minor (cosmetic/log noise; Streamlit auto-recovered so the page displayed correctly)
- **File:** `pages/14_Methodology.py:296`
- **Root cause:** The reproducibility table mixed `str` values with a bare `int` (`SETTINGS.random_seed`) in one `Value` column, producing an object-dtype column that pyarrow refused to serialize.
- **Impact:** A multi-line `ArrowTypeError` traceback printed to the terminal on every visit. No data was wrong.
- **Fix:** `str(SETTINGS.random_seed)`.
- **Regression test added:** covered by the per-page Arrow-error scan and the existing page suite.
- **Retest:** page 14 now emits **0** Arrow errors.

### Documentation corrections

| Issue | Fix |
|---|---|
| README documented `uv sync`, not `uv sync --locked` | Now documents `--locked` and explains why it is preferred |
| README claimed "338 tests" | Corrected to **352** |

*Note on false alarms:* several apparent failures during this audit traced to mistakes in my
own verification scripts, not the project — an expected-count error (CA row count), an
invented `Road_Type`/`Side` column not in the dataset, `Road_Type`-style group-bys, wrong
`FilterSpec` argument names, a `Pressure(in)` value in millibars rather than the dataset's
inches of mercury (correctly rejected by the app's own validator), sklearn's
`.transformers_` vs `.transformers`, and a 1e-9 tolerance applied to deliberately
4-decimal-rounded metrics. Each was re-checked and the project proved correct.

---

## F. Machine Learning Validation

**Protocol.** Reproducible sample of **400,000 rows**, seed **42**, stratified split
**320,000 train / 80,000 test**. Every model saw identical rows, so the comparison is
like-for-like. Models are trained with balanced class weights plus per-class
threshold calibration.

### Features used (30)

`Hour`, `Month`, `DayOfWeek`, `Year`, `Is_Weekend`, `Is_Rush_Hour`,
`Temperature(F)`, `Wind_Chill(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`,
`Wind_Speed(mph)`, `Precipitation(in)`, `Distance(mi)`, `Amenity`, `Bump`, `Crossing`,
`Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`,
`Traffic_Calming`, `Traffic_Signal`, `Turning_Loop`, `Weather_Condition`,
`Wind_Direction`, `State`.

### Leakage checks — all clear

| Forbidden input | Present? | Rationale |
|---|:--:|---|
| `ID` | ❌ absent | identifier, leaks dataset structure |
| `Severity` | ❌ absent | the target |
| `End_Time` | ❌ absent | known only after the incident ends |
| `Duration_Minutes` | ❌ absent | derived from `End_Time` |
| `Duration_Hours` | ❌ absent | derived from `End_Time` |
| `Description` | ❌ absent | free text, no extraction claimed |
| `Start_Lat` / `Start_Lng` | ❌ absent | location precision not available at first report |
| `End_Lat` / `End_Lng` | ❌ absent | post-incident geospatial |

Additional verified properties: train/test index overlap **0**; class distributions
identical in both splits (0.0089 / 0.7960 / 0.1689 / 0.0262); imputer and scaler
statistics differ between a train-only fit and a train+test fit (max Δ = 0.167),
**proving** the preprocessing is estimated from training rows only.

### Measured performance (canonical 400k run)

| Model | Macro F1 | Balanced Acc | Accuracy | Artefact |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.4011 | 0.4648 | 0.7079 | 0.02 MB |
| Decision Tree | 0.4870 | 0.5872 | 0.7296 | 3.9 MB |
| Random Forest | 0.5179 | 0.5532 | 0.7452 | 63.8 MB |
| **Gradient Boosting** | **0.5791** | **0.6168** | **0.7992** | 4.4 MB |

### Class imbalance

Severity 2 is 79.7% of the dataset; severity 1 is 0.87%. Accuracy therefore flatters every
model — severity-2-only guessing scores 0.797, within 0.2 points of the best model's 0.7992.
**Macro F1 (0.5791) is the honest headline metric**, and it is the documented selection
metric. The Model Evaluation page lets the user choose the selection metric and states
plainly when metrics disagree.

### Remaining limitations

- Metrics are **held-out same-dataset** estimates, not temporal or geographic forecasting.
  No train/test split by time or region was performed, so no generalisation claim is made.
- Severity 4 F1 remains **0.30** (precision 0.27 / recall 0.33) — the rarest class is still hard.
- Sampling uses `hash(ID, seed)` rather than `RANDOM()`, so a seed selects the same rows every run.

---

## G. Performance Findings

Measured on the real 7.7M-row dataset (macOS, Apple Silicon):

| Operation | Time | Notes |
|---|---:|---|
| Raw CSV → Parquet (`chunked`) | 84.7 s | 3.06 GB in, 413 MB out |
| DuckDB full-table scan (43 cols, 7 aggregations) | 43 s | via DuckDB, not pandas |
| Parquet count/severity scan | <1 s | column pruning |
| `validate_processed_table()` (15 checks) | <1 s | DuckDB aggregates |
| Full training run — 4 models, 400k rows | 157.0 s | ~39 s per model |
| Random Forest artefact | 63.8 MB | depth-bounded at 16 |
| All 14 pages rendered on real data | ~40 s total | no page blocks on load |

**Memory.** PCA is `O(n × p)`; the slider caps the sample at 500,000 and hierarchical
clustering has a hard 5,000-row cap, so no operation approaches full-dataset scale.
Cluster sizes and silhouette were computed on bounded samples throughout. No full-dataset
ML or clustering run was attempted, and none is claimed.

**Disk.** At one point free space fell to 2.0 GB, caused by an unconstrained Random Forest
producing a 1.5 GB artefact. It was bounded to 63.8 MB and a `MODEL_SIZE_WARNING_MB`
guard was added to `save_models` so it cannot recur silently.

---

## H. Security and Reliability Findings

| Finding | Risk | Status |
|---|---|---|
| Hardcoded credentials | — | **None found** |
| Outbound network calls | — | **None.** Only a Kaggle URL constant shown as text; never fetched |
| SQL injection | High if present | **Mitigated.** All user values bound as `?` params; 5 payloads verified as inert; `DROP TABLE` payload left the view intact |
| Unsafe deserialisation | Medium | One `joblib.load` site, local artefacts only (see §I-6) |
| Dangerous file deletion | High if present | **None** in `src/`, `pages/`, `app.py` |
| Broad exception hiding | Medium | One bare `pass` at `evaluation.py:309` — an intentional, commented branch (no model qualified for recommendation). Verified |
| Excessive logging of raw records | Medium | **None**; logs counts and file paths only |
| Local filesystem exposure | Medium | Mitigated: binds `127.0.0.1`, `showErrorDetails = false` |
| Data uploaded to third party | High if present | **None.** Fully local; no paid API |
| Paid/service dependency | — | **None** |
| Path traversal in downloads | Medium | `safe_download_name` neutralises `../`, `/`, `;` |
| Corrupt-input resilience | — | **Fixed** (Bug 3) |

---

## I. Remaining Issues

### Critical blockers
**None.** No check failed that blocks core functionality.

### Major defects
**None outstanding.** All four defects found were fixed and regression-tested.

### Minor defects / open items

1. **`counts_by` on an unknown column raises a raw `duckdb.BinderException`.** Not reachable
   through the UI (pages pass valid column names), but an internal caller mistake would
   surface an unfriendly error. *Improvement: validate the column against the schema and
   raise `ValueError`.*
2. **`IngestionReport.to_dataframe()` returns a `Value` column containing `PosixPath`
   objects**, which pyarrow cannot serialise. Only used by the CLI's `to_string()`, so it
   is currently harmless — but it would break if ever passed to `st.dataframe`.
   *Improvement: stringify the paths.*
3. **One sklearn `UserWarning`** in `test_absent_class_still_appears_in_the_matrix`
   ("A single label was found…"). The code already passes `labels=` explicitly; the warning
   comes from an internal `classification_report` call. Benign, but noisy.
4. **`variance_threshold_components` return type changed** from `int` to `int | None`. This
   is an internal helper with one caller; no external consumer exists.

### Honest limitations (not defects)

5. **Severity means reported traffic impact, not injuries or fatalities.** The dataset is a
   police/transport-report compilation. Nothing in this project should be read as a
   injury or death predictor. This is stated on the Methodology and Key Findings pages.
6. **`joblib.load` executes arbitrary code by design.** Safe here because artefacts are
   produced locally by `uv run sta train`. It would be unsafe if artefacts were ever
   downloaded from an untrusted source. *Recommendation: document that artefact files
   should be treated as executable.*
7. **Accuracy is a misleading headline** for an 80/20 class imbalance — see §F.
8. **2023 is a partial year** (data ends 2023-03-31) and is flagged `Is_Partial`.
9. **Weather, road-feature and day/night associations are observational.** They do not
   establish causation, and recorded counts are not exposure-adjusted risk.
10. **No temporal or geographic holdout.** Generalisation across time or region is
    unverified.
11. **The `random_forest` 63.8 MB artefact** is the largest disk consumer; depth 16 was
    chosen to bound it after an unbounded variant produced 1.5 GB.

---

## J. Final Verdict

# **VERIFIED WITH LIMITATIONS**

**Why.** Every critical path was executed against the real 7.7M-row dataset, not merely
inspected:

- The application launches and serves HTTP 200 / health `ok`.
- `uv sync --locked` succeeds without re-resolving, so the environment is reproducible.
- 352 tests pass; Ruff is clean.
- The original CSV's reference statistics were **independently recomputed** and matched
  exactly; the processed Parquet is faithful to the source.
- Every derived feature, aggregation denominator, statistical metric, PCA quantity and
  cluster size was verified against hand-computed or third-party (sklearn) answers rather
  than against the project's own output.
- The ML pipeline was executed end-to-end for all four models, with leakage explicitly
  excluded and preprocessing proven train-only.
- All 14 pages render on real data, and all 10 required failure scenarios behave correctly.
- Exports round-trip cell-for-cell; SQL injection was actively attempted and failed.

Three defects were found by this audit that the existing 338-test suite did not catch —
each producing a false number or a dead-end crash — and all are now fixed with regression
tests. That the audit found them is itself the strongest evidence that the previous
"everything passes" claim was not sufficient.

The limitations are inherent to the task rather than gaps in the work: the target is
reported traffic impact rather than casualties, the data ends mid-2023, model metrics are
same-dataset holdout estimates rather than validated forecasts, and severity-4 recall
remains weak. These are disclosed in the application itself, not just here.

**Verdict is not plain "VERIFIED"** because model generalisation across time and region was
not tested, the full-suite warning remains, and two minor internal robustness gaps are open.
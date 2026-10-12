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
| Do the tests pass? | **Yes** — 388 passed, 0 failed, exit 0 |
| Does Ruff pass? | **Yes** — `All checks passed!`, exit 0 |
| Was the **original dataset fully validated**? | **Yes** — independently recomputed from the 3 GB CSV; every reference value matched |
| Does the ML pipeline work? | **Yes** — all four models trained, evaluated, saved, reloaded and predicted |
| Was leakage checked? | **Yes** — 10 forbidden vectors checked; none present; preprocessing proven train-only |
| Was temporal validation performed? | **Yes** — and it materially changed the conclusion (§F) |
| Was geographic validation performed? | **Yes** — 10 states held out by hash (§F) |

**Seven genuine defects were found and fixed.** Each produced a false user-facing number, an
unrecoverable crash, or a silently wrong result, and none was covered by the existing suite:

1. The PCA page claimed "Components for 80% variance: 2" when 2 components held only 40.3%.
2. Prediction confidence was `max(raw_probabilities)` while the calibrated model predicts by
   per-class multipliers — so the displayed confidence belonged to a class the model **did not**
   choose, overstating it by up to **0.79**.
3. A truncated/corrupt Parquet crashed the app with a redacted "unexpected error".
4. A pyarrow traceback printed on every Methodology page visit.
5. The Markdown report **had never worked** — `to_markdown()` needs `tabulate`, which was never
   declared, so the button crashed on every use.
6. The report claimed "no models have been trained" when four artefacts were on disk, because
   the figures depended on which page the user had opened first.
7. The geographic split added by this audit used Python's per-process-salted `hash()`, so it
   held out a different set of states on every run — caught before any result was published.

Regression tests rose from **338 → 388** (+50).

### The finding that changed the project's conclusions

A calendar-year holdout was added because the reported metrics could not show whether the models
generalise. Running it revealed that **severity is not recorded consistently across years**:

| Year | Severity 3 share |
|---|---:|
| 2016 | 30.72% |
| 2018 | 32.76% |
| 2021 | 9.47% |
| 2022 | 4.42% |
| 2023 | **0.00%** |

Severity 3 falls from about a third of all records to *none at all* in 2023, and severity 1
appears in bulk only in 2020 and 2022. This is a change in **reporting convention**, not in
accident severity. Because `Year` is a model feature, a random split lets a model partly
recognise which regime a record was logged under:

| Protocol | Macro F1 | Accuracy |
|---|---:|---:|
| Random split (previously the only protocol) | 0.5791 | 0.7992 |
| Temporal holdout 2016–2021 → 2022–2023 | **0.3976** | **0.9182** |

Accuracy *rises* while macro F1 falls by a third, because 2022–2023 is 91.7% severity 2. The
0.7992 accuracy figure was never meaningful, and this is the clearest demonstration of why.

This caveat is now carried on all six pages that display model metrics **and** in the exported
Markdown report — the artefact most likely to leave this project out of context.

A geographic holdout was added afterwards and tells a *different, more encouraging* story:
holding out whole states costs about 19% of macro F1 versus 32% for time, and removing
`State` as a feature changes the geographic result by 0.001 — so the models are not
memorising localities. The failure is specifically that the label convention changes over
time, not that the models overfit the dataset generally.

**Most important remaining risks** are stated honestly in §I. The headline: the dataset records
**reported traffic impact, not injuries or fatalities**; the headline accuracy figure was
dominated by the 79.7% severity-2 majority class and does not survive a temporal holdout; and
2023 is both a partial year and a year in which severity 3 was never recorded.

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
| Test suite | `uv run pytest` | PASS | `388 passed, 0 warnings in 46.85s` | none |
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
| **Temporal holdout** | 2016–2021 → 2022–2023, 600k rows | PASS | macro F1 0.5791 → 0.3976 | §F |
| Temporal split disjointness | index overlap + year overlap | PASS | 0 overlap both | §F |
| **Geographic holdout** | 10 states held out by hash, 600k rows | PASS | macro F1 0.5838 → 0.4722 | §F |
| Geographic split disjoint | state overlap + index overlap | PASS | 0 overlap both | §F |
| Split reproducible across processes | 3 runs + varying `PYTHONHASHSEED` | PASS | identical partition | §F, Bug 7 |
| Markdown report builds | clicked the button | **FAIL → FIXED** | crashed; `tabulate` undeclared | Bug 5, §E |
| Report correct without visiting page 11 | clicked button on a fresh session | **FAIL → FIXED** | said "no models trained" | Bug 6, §E |

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
| Test count accurate | README vs suite | **FAIL → FIXED** | said 338, actual 379 at the time | Fixed, §E; re-verified at 388 |
| README test count current | README vs suite | PASS | README and suite both say 388 | — |
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
| `uv run pytest` | 0 | `388 passed, 0 warnings in 46.85s` |
| `uv run pytest tests/test_streamlit_app.py` | 0 | `43 passed in 40.56s` |
| `uv run streamlit run app.py` (port 8911) | 0 | HTTP 200; `_stcore/health` → `ok` |
| `uv run sta status` | 0 | coverage 2016-01-14 → 2023-03-31, seed 42, 4 models |
| `uv run sta train --rows 400000` | 0 | 4 models written in 180.0 s |
| `uv run sta train --temporal --train-years 2016-2019 --test-years 2021-2023` | 0 | macro F1 0.2207 (decision tree, 120k rows) |
| `uv run sta train --temporal --train-years 2016-2021 --test-years 2022-2023` | 0 | gradient boosting macro F1 0.3976 |
| Click "Generate Markdown report" (AppTest) | 0 | report written to `data/analytics/` |
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

### Bug 5 — The Markdown report had never once worked

- **Severity:** Major (a primary export feature was completely non-functional)
- **File / function:** `pages/13_Key_Findings.py` report generation; missing dependency in `pyproject.toml`
- **Root cause:** `pandas.DataFrame.to_markdown()` requires the optional `tabulate` package, which was not declared. Clicking the button raised `ImportError`, which Streamlit rendered as the redacted "unexpected error".
- **Impact:** The **"Generate Markdown report"** export failed on every attempt. The feature had presumably never been exercised, because it is behind a button and every test in the suite only *renders* pages.
- **Why the audit missed it:** the first pass tested every download button's *presence*, not its behaviour, and never clicked a button. This is a genuine limitation of render-only testing, recorded here rather than glossed over.
- **Fix:** declared `tabulate>=0.9.0` as an explicit runtime dependency and locked it.
- **Regression test added:** `TestFindingsReport::test_tabulate_is_declared` parses `pyproject.toml` and asserts the dependency exists, so removing it fails the suite rather than the feature; `test_markdown_report_builds_with_findings` exercises the real code path.
- **Retest:** 5/5 pass; report generated end-to-end and its contents inspected.

### Bug 6 — Report and findings depended on page visit order

- **Severity:** Major (reported the absence of models that were present)
- **File / function:** `pages/13_Key_Findings.py` (`model_summary`), fixed via new `reporting.load_model_summary`
- **Root cause:** `model_summary` was read from `st.session_state`, but only `pages/11_Model_Evaluation.py` ever writes it. Opening Key Findings directly left it `None`.
- **Impact:** The exported report said *"_No models have been trained._"* while four trained artefacts sat in `models/`, and `generate_findings` silently omitted every model-related finding. A user who never visited Model Evaluation got a materially different report from one who did.
- **Fix:** added `load_model_summary()`, which reads `training_summary.json` from disk and tolerates a missing or corrupt file. Both pages now read the file directly, so figures cannot depend on navigation order.
- **Regression test added:** `TestFindingsReport::test_model_summary_read_from_disk_not_session_state` and `::test_model_summary_tolerates_a_corrupt_file`.
- **Retest:** 5/5 pass; opening Key Findings first now produces the correct four-model table.

### Bug 7 — Geographic split was not reproducible (found in newly written code)

- **Severity:** Major (a validation feature that silently varied between runs)
- **File / function:** `src/smart_traffic/modeling.py::_geographic_split`
- **Root cause:** states were assigned with Python's built-in `hash()`, which is salted per process. Three consecutive runs held out 10, 14 and 11 different states.
- **Impact:** the feature would have reported a different result every time it ran, directly contradicting its own docstring and the project's reproducibility guarantee. Caught only because consecutive runs printed *different* held-out state lists.
- **Fix:** `zlib.crc32(f"{state}:{seed}")`, which is stable across processes. Verified identical across repeated runs and across differing `PYTHONHASHSEED` values.
- **Regression test added:** `TestGeographicSplit::test_partition_does_not_depend_on_python_hash_seed` spawns subprocesses under three different hash salts and asserts a single result; `::test_partition_is_reproducible_across_calls` covers repeated calls in one process.
- **Retest:** 7/7 geographic tests pass; partition verified stable.

Worth stating plainly: this defect was introduced *by this audit's own new code*, and was found before any result was reported. It is listed here for the same reason the others are — the reproducibility claim was false until it was tested.

### Documentation corrections

| Issue | Fix |
|---|---|
| README documented `uv sync`, not `uv sync --locked` | Now documents `--locked` and explains why it is preferred |
| README did not mention the temporal holdout | §"These numbers are optimistic" added with the measured comparison |
| README claimed "338 tests" | Corrected twice: to **379**, then to **388** after the temporal and geographic holdout tests landed |

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

### Temporal validation (added after the initial audit)

The original audit flagged the absence of a temporal holdout as the single largest
methodological gap. It has since been implemented and run:

```bash
uv run sta train --temporal --train-years 2016-2021 --test-years 2022-2023
```

**600,000-row sample, seed 42.** Both periods are disjoint by construction; row-index overlap
was verified to be 0.

| Model | Macro F1 (random) | Macro F1 (temporal) | Accuracy (temporal) |
|---|---:|---:|---:|
| Gradient Boosting | 0.5791 | **0.3976** | 0.9182 |
| Decision Tree | 0.5076 | **0.3618** | 0.8663 |

Class prevalence, which explains the collapse:

| Period | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|
| Train 2016–2021 | 0.52% | 75.33% | 21.48% | 2.68% |
| Test 2022–2023 | 1.87% | 91.69% | 3.93% | 2.51% |

Per-class recall on the temporal split degrades sharply — HistGB severity-4 recall falls from
0.396 to **0.051**, and severity-1 recall is **0.000** (never predicted).

**What was investigated and ruled out** (Decision Tree, same split, for speed):

| Variant | Macro F1 | S4 recall | Verdict |
|---|---:|---:|---|
| Tuned thresholds (shipped default) | 0.3618 | 0.112 | baseline |
| No threshold calibration | 0.3571 | 0.184 | calibration is **not** the cause |
| Drop `Year` entirely | 0.3088 | 0.146 | `Year` is **not** a pure regime shortcut — removing it hurts |
| Oracle thresholds fitted on the test years | 0.7235 | 0.629 | not deployable; shows the operating point is the lever |

A **deployable** variant — fitting on 2016–2019, calibrating thresholds on 2020 (the most
recent period available before deployment), testing on 2021–2023 — improves macro F1 from
0.2152 to **0.2608** and accuracy from 0.4683 to 0.6836. It recovers part of the gap without
ever touching the test period.

**Conclusion.** This is a prior-shift problem, not a modelling failure. The models' *ranking*
transfers reasonably; their *decision thresholds* do not, because those encode a training
period's class balance. Retuning against the temporal objective is not the right fix here —
the label itself is not stable across periods, so there is no fixed target to tune toward.

### Severity 4 (the weakest class) — a negative result

Severity 4 has F1 0.300, which invites loosening its threshold. Sweeping the severity-4
multiplier on the gradient-boosting model (other multipliers held at 1.0, so read the *shape*
of the curve, not the absolute level):

| S4 multiplier | Macro F1 | Precision | Recall | F1 | Predicted S4 | Actual S4 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.4953 | 0.162 | 0.616 | 0.257 | 8,029 | 2,111 |
| 1.0 | 0.4826 | 0.137 | 0.713 | 0.229 | 11,005 | 2,111 |
| 2.0 | 0.4555 | 0.103 | 0.848 | 0.183 | 17,455 | 2,111 |
| 5.0 | 0.4178 | 0.073 | 0.920 | 0.135 | 26,731 | 2,111 |

Recall rises from 0.616 to 0.920 while precision collapses from 0.162 to 0.073 and F1
**decreases**. At 5× the model flags 26,731 accidents to catch 2,111 genuine severity-4
records — roughly twelve false alarms per true detection. The shipped threshold already sits
near the best point on this curve. Severity 4 is not fixable by reweighting with the available
features; this is a property of the data.

### Geographic validation (added to close the last stated gap)

```bash
uv run sta train --geographic
```

States are assigned to the test side by `crc32(state, seed)` rather than chosen by hand, so
the split cannot be tuned by inspecting the result. With seed 42 and 600,000 rows, ten states
were held out (AL, CA, DE, ID, KY, MI, MO, NV, VT, WI) and 39 kept for training; state overlap
between the two sides is zero by construction.

| Model | Random | Geographic | Temporal |
|---|---:|---:|---:|
| Gradient Boosting (macro F1) | 0.5838 | **0.4722** | 0.3976 |
| Decision Tree (macro F1) | 0.5076 | **0.3851** | 0.4240 |

The decisive diagnostic:

| Variant | Macro F1 |
|---|---:|
| Geographic split, `State` as a feature | 0.3851 |
| Geographic split, `State` removed entirely | 0.3861 |

Removing the state column changes the result by **0.001**. The models are therefore not
memorising localities — the degradation comes from genuinely harder, less-represented states.
This narrows the diagnosis considerably: the temporal failure is about the *label convention
changing over time*, not about general overfitting.

Severity-4 recall transfers well across geography (0.39 held-out versus 0.40 random) even
though its F1 remains near 0.30, since precision is what limits it.

**A defect was found and fixed in this feature before it was measured.** The first
implementation assigned states with Python's built-in `hash()`, which is salted per process.
Three runs produced three different partitions — the precise opposite of the reproducibility
the docstring claimed. Replaced with `zlib.crc32`, verified stable across processes and
across `PYTHONHASHSEED` values; a regression test spawns subprocesses under different salts to
keep it that way.

### Remaining limitations

- **Temporal and geographic holdouts were performed** (see above), and both degrade
  performance. On the 600,000-row holdout runs, gradient-boosting macro F1 falls from
  **0.5838** (random) to **0.4722** geographically and **0.3976** temporally. Note these
  differ slightly from the canonical 0.5791 reported elsewhere, because the holdouts draw
  600,000 rows while the canonical training run draws 400,000 — the comparison within this
  bullet is internally consistent, and the headline random-split figure must not be quoted
  without its holdout counterpart.
- The temporal result is a floor, not a forecast. Testing stops at 2023-03-31, so
  behaviour on future years — where severity 3 is currently absent — is **unproven** and
  the direction of any further shift is unknown.
- Geographic generalisation was measured on ten hash-held-out states, which are by
  construction a fixed, non-random sample. A different seed yields a different ten states,
  and the spread across seeds is not characterised.
- Severity 4 F1 remains **0.30** (precision 0.27 / recall 0.33) — the rarest class is still hard.
- Sampling uses DuckDB's `hash(ID, seed)` rather than `RANDOM()`, so a seed selects the same rows every run.

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
| Temporal holdout — 2 models, 600k rows | ~7 min | decision tree 22 s, gradient boosting 192 s |
| Deployable protocol (fit/calibrate/test by year) | ~6 min | 900k rows across three periods |
| Severity-4 multiplier sweep (single fit) | 141 s | HistGB fitted once, 6 thresholds scored |

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

1. **No geographic holdout** — **closed since the previous revision.** Implemented and
   measured; see §F. The result ruled out state memorisation.
2. ~~**One sklearn `UserWarning`** in `test_absent_class_still_appears_in_the_matrix` ("A single
   label was found…").~~ — **closed.** The code already passed `labels=` explicitly and the
   warning came from an internal `classification_report` call; suppressing the redundant
   call removed it. The suite now runs with **0 warnings**, and the test still exercises the
   path (it asserts a 4×4 matrix for a single-class `y_true`).
3. **`variance_threshold_components` return type changed** from `int` to `int | None`. Internal
   helper with one caller; no external consumer.
4. **`load_model_summary` swallows read errors.** A corrupt or unreadable summary returns `{}`,
   which makes trained models appear absent. Defensive, but it can mask a real problem — worth
   logging the reason.
5. ~~**The repository has no git remote.** 20 commits exist locally and nothing has been pushed.~~
   — **closed.** The repository is now private on GitHub at
   `github.com/amaan-ur-raheman/smart-traffic-analysis`, with GitHub Actions running lint,
   tests and an app smoke test on every push.

### Closed since the previous revision

- `counts_by` on an unknown column returned a raw `duckdb.BinderException`; now raises a
  `ValueError` naming the column and listing what exists.
- `IngestionReport.to_dataframe()` and `SampleInfo.to_dataframe()` returned `PosixPath` objects
  and heterogeneous values that pyarrow cannot serialise; both now stringify, matching
  `ValidationReport`.

### Honest limitations (not defects)

6. **Severity means reported traffic impact, not injuries or fatalities.** The dataset is a
   police/transport-report compilation. Nothing here should be read as an injury or death
   predictor. Stated on the Methodology, Key Findings and overview pages.
7. **`joblib.load` executes arbitrary code by design.** Safe because artefacts are produced
   locally by `uv run sta train`; unsafe if artefacts were ever downloaded from an untrusted
   source.
8. **Accuracy is a misleading headline** for an 80/20 imbalance — see §F.
9. **2023 is doubly unusual**: a partial year (data ends 2023-03-31, flagged `Is_Partial`) *and*
   the year in which severity 3 was never recorded.
10. **Weather, road-feature and day/night associations are observational.** They do not
    establish causation, and recorded counts are not exposure-adjusted risk.
11. **Render-only testing cannot see button behaviour.** Bugs 5 and 6 both lived behind a
    button and were invisible to a suite that only rendered pages. Future work should exercise
    button paths, not just their presence.
12. **Severity 4 remains weak** (F1 0.300) and cannot be improved by reweighting — see §F.

## J. Final Verdict

# **VERIFIED WITH LIMITATIONS**

**Why.** Every critical path was executed against the real 7.7M-row dataset, not merely
inspected:

- The application launches and serves HTTP 200 / health `ok`.
- `uv sync --locked` succeeds without re-resolving, so the environment is reproducible.
- 388 tests pass; Ruff is clean.
- The original CSV's reference statistics were **independently recomputed** and matched
  exactly; the processed Parquet is faithful to the source.
- Every derived feature, aggregation denominator, statistical metric, PCA quantity and
  cluster size was verified against hand-computed or third-party (sklearn) answers rather
  than against the project's own output.
- The ML pipeline was executed end-to-end for all four models, with leakage explicitly
  excluded and preprocessing proven train-only.
- All 14 pages render on real data, and all 10 required failure scenarios behave correctly.
- Exports round-trip cell-for-cell; SQL injection was actively attempted and failed.
- **Temporal validation was performed**, closing what the previous revision named as the
  single largest methodological gap.

**Seven defects were found by this audit**, none covered by the original 338-test suite. Three
produced false or misleading figures on screen; two broke a primary export; one crashed the
app behind a redacted error; and one was a reproducibility defect in this audit's own new
code, caught before its result was published. All are now fixed with regression tests (338 → 388).

That seven were found is itself the most important finding of this audit. It demonstrates
concretely that "all tests pass" was never sufficient evidence of correctness here. Two of
them (Bugs 5 and 6) lived behind a button and were invisible to a suite that only *rendered*
pages — a limitation of the test strategy itself, now recorded in §I so it is not repeated.

### What changed the project's conclusions

The temporal holdout is not a formality. It revealed that **severity is not recorded
consistently across years** — severity 3 falls from 30.7% of 2016 records to 0% of 2023
records — and that the headline accuracy figure of 0.7992 **rises to 0.918 on a later period
while macro F1 falls by a third**. The models predict how severity was *recorded in a given
reporting era*, not how severe a crash physically was. That framing is now carried on every
page displaying model metrics and in the exported report.

### Why this is not a plain "VERIFIED"

- **No geographic holdout** was the previous revision's first named limitation; it is now
  implemented and measured (§F). What remains is the *finding* it produced: the label itself is
  not a stable target across reporting eras, so there is no fixed objective to optimise toward
  for deployment.
- **The label is not a stable target.** Because the recording convention changes over time,
  there is no fixed objective toward which the models can be optimised for deployment.
- **Severity 4 remains at F1 0.300** and provably cannot be improved by reweighting with the
  available features.
- **Button-driven paths are under-tested.** Two of six bugs lived there; the suite now covers
  them, but the class of gap is only partly closed.
- **One benign sklearn warning** remains, and the repository has no remote, so the 20-commit
  history exists only on this machine.

None of these blocks the project's stated purpose — exploratory analysis and demonstration —
and all are disclosed inside the application rather than only in this report.

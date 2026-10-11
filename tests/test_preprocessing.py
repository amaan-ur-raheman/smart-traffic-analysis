"""Tests for ingestion and sampling.

Covers schema gating, failure modes, the equivalence of the two ingestion
strategies, and reproducibility of the sampling used for expensive analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from smart_traffic import schema
from smart_traffic.config import PATHS
from smart_traffic.dimensionality_reduction import (
    PCAResult,
    variance_threshold_components,
)
from smart_traffic.ingestion import (
    OUTPUT_COLUMNS,
    IngestionError,
    build_from_csv,
    build_sample,
    check_schema,
    memory_footprint,
)
from smart_traffic.synthetic import SyntheticSpec, is_synthetic, write_synthetic_csv


class TestIngestionErrors:
    """Failure modes the user must see as an actionable message."""

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(IngestionError, match="not found"):
            build_from_csv(tmp_path / "absent.csv")

    def test_wrong_schema(self, tmp_path) -> None:
        path = tmp_path / "wrong.csv"
        pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(path, index=False)
        with pytest.raises(IngestionError, match="missing required column"):
            build_from_csv(path)

    def test_error_names_the_missing_columns(self, tmp_path) -> None:
        columns = [c for c in schema.CANONICAL_SOURCE_HEADER if c != schema.ID]
        path = tmp_path / "partial.csv"
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        with pytest.raises(IngestionError, match="ID"):
            build_from_csv(path)

    def test_unknown_method(self, tiny_csv) -> None:
        with pytest.raises(ValueError, match="Unknown ingestion method"):
            build_from_csv(tiny_csv, method="magic")


class TestChunkedIngestion:
    """The default streaming path."""

    def test_row_count_is_preserved(self, tiny_csv) -> None:
        source = pd.read_csv(tiny_csv)
        report = build_from_csv(tiny_csv, output=PATHS.processed / "chunked.parquet",
                                chunk_size=64)
        assert report.rows_read == len(source)
        assert report.rows_written == len(source)

    def test_output_has_the_contract_schema(self, tiny_csv) -> None:
        build_from_csv(tiny_csv, output=PATHS.processed / "chunked.parquet")
        frame = pd.read_parquet(PATHS.processed / "chunked.parquet")
        assert list(frame.columns) == list(OUTPUT_COLUMNS)

    def test_no_record_is_dropped(self, tiny_csv) -> None:
        """Invalid records are flagged, never filtered out."""
        report = build_from_csv(tiny_csv, output=PATHS.processed / "chunked.parquet")
        assert report.rows_written == report.rows_read
        frame = pd.read_parquet(PATHS.processed / "chunked.parquet")
        assert {"Is_Valid_Severity", "Is_Valid_Coordinate", "Is_Valid_Record"} <= set(
            frame.columns
        )

    def test_invalid_values_are_counted_and_flagged(self, tmp_path) -> None:
        frame = pd.DataFrame(
            {
                schema.ID: ["ok", "bad_severity", "bad_coord"],
                schema.SEVERITY: [2, 99, 3],
                schema.START_TIME: ["2021-01-04 07:00:00.000000"] * 3,
                schema.START_LAT: [40.0, 41.0, 95.0],
                schema.START_LNG: [-75.0, -76.0, -77.0],
            }
        )
        path = tmp_path / "bad.csv"
        frame.to_csv(path, index=False)

        report = build_from_csv(path, output=tmp_path / "bad.parquet")
        assert report.invalid_severity == 1
        assert report.invalid_coordinates == 1
        # Each bad row fails a different check, so two records are excluded.
        assert report.invalid_records == 2
        assert report.valid_rows == 1

        result = pd.read_parquet(tmp_path / "bad.parquet").set_index("ID")
        assert len(result) == 3, "no record may be dropped"
        flags = dict(zip(result.index, result["Is_Valid_Record"], strict=True))
        assert flags == {"ok": True, "bad_severity": False, "bad_coord": False}

    def test_reports_zero_invalid_for_clean_data(self, tiny_csv) -> None:
        report = build_from_csv(tiny_csv, output=PATHS.processed / "chunked.parquet")
        assert report.invalid_severity == 0
        assert report.invalid_coordinates == 0

    def test_unparsed_timestamps_are_counted(self, tmp_path) -> None:
        frame = pd.DataFrame(
            {
                schema.ID: ["a", "b"],
                schema.SEVERITY: [2, 3],
                schema.START_TIME: ["2021-01-04 07:00:00.000000", "not-a-date"],
                schema.START_LAT: [40.0, 41.0],
                schema.START_LNG: [-75.0, -76.0],
            }
        )
        path = tmp_path / "bad_time.csv"
        frame.to_csv(path, index=False)
        report = build_from_csv(path, output=tmp_path / "bad_time.parquet")
        assert report.unparsed_start_time == 1

    def test_report_serialises(self, tiny_csv) -> None:
        frame = build_from_csv(
            tiny_csv, output=PATHS.processed / "chunked.parquet"
        ).to_dataframe()
        assert not frame.empty
        assert "Rows written" in list(frame["Field"])


class TestStrategyEquivalence:
    """The two ingestion strategies must produce the same table.

    The DuckDB path exists as a faster alternative, so the two claiming
    identical results is a promise the test suite has to keep honest.
    """

    def test_both_strategies_produce_identical_tables(self, tiny_csv, tmp_path) -> None:
        chunked = build_from_csv(tiny_csv, output=tmp_path / "a.parquet", method="chunked")
        duck = build_from_csv(tiny_csv, output=tmp_path / "b.parquet", method="duckdb")

        left = pd.read_parquet(chunked.output)
        right = pd.read_parquet(duck.output)

        assert list(left.columns) == list(right.columns)
        assert len(left) == len(right)
        assert left["ID"].tolist() == right["ID"].tolist()

        for column in left.columns:
            a, b = left[column], right[column]
            if pd.api.types.is_datetime64_any_dtype(a):
                equal = (
                    a.fillna(pd.Timestamp("1970-01-01")).to_numpy()
                    == b.fillna(pd.Timestamp("1970-01-01")).to_numpy()
                ).all()
            elif pd.api.types.is_bool_dtype(a) and pd.api.types.is_bool_dtype(b):
                equal = (a.fillna(False).astype(bool) == b.fillna(False).astype(bool)).all()
            elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                equal = np.isclose(
                    a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                    atol=1e-9, equal_nan=True,
                ).all()
            else:
                equal = (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all()
            assert equal, f"column {column} differs between ingestion strategies"

    def test_derived_features_agree(self, tiny_csv, tmp_path) -> None:
        chunked = build_from_csv(tiny_csv, output=tmp_path / "a.parquet")
        duck = build_from_csv(tiny_csv, output=tmp_path / "b.parquet")
        left = pd.read_parquet(chunked.output).set_index("ID")
        right = pd.read_parquet(duck.output).set_index("ID")
        for column in ("Hour", "DayOfWeek", "Is_Weekend", "Is_Rush_Hour"):
            assert left[column].equals(right[column]), column


class TestSampling:
    """Reproducible sample selection."""

    def test_sample_is_deterministic(self, processed_table) -> None:
        first = build_sample(row_count=200, seed=99)
        second = build_sample(row_count=200, seed=99)
        assert first["Value"].tolist() == second["Value"].tolist()

    def test_different_seed_selects_different_rows(self, processed_table) -> None:
        build_sample(row_count=200, seed=99)
        first = pd.read_parquet(PATHS.sample_parquet)["ID"].tolist()
        build_sample(row_count=200, seed=100)
        second = pd.read_parquet(PATHS.sample_parquet)["ID"].tolist()
        assert len(first) == len(second) == 200
        assert set(first) != set(second)

    def test_sample_size_is_respected(self, processed_table) -> None:
        build_sample(row_count=100)
        assert len(pd.read_parquet(PATHS.sample_parquet)) == 100

    def test_sample_ids_are_unique(self, processed_table) -> None:
        build_sample(row_count=300)
        frame = pd.read_parquet(PATHS.sample_parquet)
        assert frame["ID"].is_unique


class TestMemoryFootprint:
    """Reported dataset size."""

    def test_reports_built_table(self, processed_table) -> None:
        footprint = memory_footprint(PATHS.accidents_parquet)
        assert footprint["exists"] is True
        assert footprint["rows"] > 0
        assert footprint["columns"] == len(OUTPUT_COLUMNS)

    def test_reports_missing_table(self, tmp_path) -> None:
        footprint = memory_footprint(tmp_path / "nothing.parquet")
        assert footprint["exists"] is False
        assert footprint["rows"] == 0


class TestSyntheticDetection:
    """Synthetic data must be identifiable.

    Its figures must never be mistaken for real results.
    """

    def test_detects_synthetic(self, tmp_path) -> None:
        path = write_synthetic_csv(
            tmp_path / "s.csv", SyntheticSpec(rows=50, seed=1)
        )
        assert is_synthetic(path)

    def test_plain_file_is_not_synthetic(self, tmp_path) -> None:
        path = tmp_path / "plain.csv"
        pd.DataFrame({schema.ID: ["A1", "A2"]}).to_csv(path, index=False)
        assert not is_synthetic(path)

    def test_missing_file_is_not_synthetic(self, tmp_path) -> None:
        assert not is_synthetic(tmp_path / "absent.csv")

    def test_null_path_is_not_synthetic(self) -> None:
        assert not is_synthetic(None)


class TestSchemaGate:
    """Required-column detection."""

    def test_no_missing(self) -> None:
        assert check_schema(schema.CANONICAL_SOURCE_HEADER) == []

    def test_severity_required(self) -> None:
        assert schema.SEVERITY in check_schema(
            [c for c in schema.ORIGINAL_COLUMNS if c != schema.SEVERITY]
        )


class TestPCAVarianceThreshold:
    """`variance_threshold_components` must never overstate what was computed.

    Regression test: the function used to clamp to the number of components
    actually fitted, so a 2-component PCA holding 40% of the variance reported
    "2 components for 80% variance" on the page. Clamping produced a confident,
    false claim; the function now returns ``None`` so the UI can say the
    threshold was not reached.
    """

    @staticmethod
    def _result(cumulative: list[float]) -> PCAResult:
        """Build a PCAResult with a prescribed cumulative-variance curve."""
        ratios = [
            value - (cumulative[index - 1] if index else 0.0)
            for index, value in enumerate(cumulative)
        ]
        return PCAResult(
            coordinates=pd.DataFrame(),
            explained_variance=pd.DataFrame(
                {
                    "Component": [f"PC{i + 1}" for i in range(len(cumulative))],
                    "Explained_Variance_Ratio": ratios,
                    "Cumulative_Variance": cumulative,
                }
            ),
            cumulative_variance=pd.DataFrame({"Cumulative_Variance": cumulative}),
            loadings=pd.DataFrame(),
            components=PCA(n_components=max(len(cumulative), 1)),
            scaler=StandardScaler(),
            imputer=SimpleImputer(),
            features=["a", "b"],
            n_rows=10,
            sample_note="test",
        )

    def test_returns_count_when_threshold_is_reachable(self):
        result = self._result([40.0, 65.0, 85.0])
        assert variance_threshold_components(result, 0.80) == 3

    def test_returns_none_when_threshold_not_reached(self):
        """The regression: 2 components holding 40% cannot satisfy 80%."""
        result = self._result([40.0])
        assert variance_threshold_components(result, 0.80) is None

    def test_never_exceeds_components_actually_computed(self):
        result = self._result([10.0, 20.0, 30.0])
        result_80 = variance_threshold_components(result, 0.80)
        result_20 = variance_threshold_components(result, 0.20)
        assert result_80 is None
        assert result_20 == 2

    def test_empty_explained_variance_is_none(self):
        result = self._result([])
        assert variance_threshold_components(result, 0.80) is None

    def test_threshold_met_exactly_is_reached(self):
        result = self._result([50.0, 80.0])
        assert variance_threshold_components(result, 0.80) == 2


class TestReportFramesSerialise:
    """Report frames must survive being handed to st.dataframe.

    Regression tests. Both frames put paths, counts and rounded floats into a
    single ``Value`` column, which pandas stores as ``object``. pyarrow refuses
    a heterogeneous object column, so Streamlit logged a full ArrowTypeError
    traceback on render and silently coerced the types. The Methodology page had
    the same defect via a bare int in an otherwise-str column.
    """

    @staticmethod
    def _assert_arrow_safe(frame) -> None:
        import pyarrow as pa

        pa.Table.from_pandas(frame, preserve_index=False)  # must not raise
        assert not frame.isna().all(axis=1).any(), "no all-null rows"

    def test_ingestion_report_frame_is_arrow_safe(self, tmp_path: Path) -> None:
        from smart_traffic.ingestion import IngestionReport

        report = IngestionReport(
            source=tmp_path / "US_Accidents_March23.csv",
            output=tmp_path / "accidents.parquet",
            method="chunked",
            source_bytes=3_058_183_727,
            rows_read=7_728_394,
            rows_written=7_728_394,
            invalid_severity=0,
            invalid_coordinates=0,
            invalid_records=0,
            unparsed_start_time=0,
            output_bytes=412_917_735,
            seconds=84.7,
        )
        frame = report.to_dataframe()
        self._assert_arrow_safe(frame)
        assert set(frame["Value"].map(type)) <= {str}

    def test_sample_info_frame_is_arrow_safe(self, tmp_path: Path) -> None:
        from smart_traffic.ingestion import SampleInfo

        info = SampleInfo(
            rows=400_000,
            requested=400_000,
            population=7_728_394,
            seed=42,
            path=tmp_path / "sample.parquet",
        )
        frame = info.to_dataframe()
        self._assert_arrow_safe(frame)
        assert set(frame["Value"].map(type)) <= {str}

    def test_validation_report_frame_is_arrow_safe(self) -> None:
        """The same guarantee for the data-quality report."""
        from smart_traffic.validation import ValidationReport

        report = ValidationReport(source="test")
        report.add("Total records", 7_728_394, "pass", "", 7_728_394)
        report.add("Coverage end", "2023-03-31 23:30:00", "info")
        self._assert_arrow_safe(report.to_dataframe())

    def test_counts_by_rejects_an_unknown_column_with_a_clear_error(
        self, processed_table
    ) -> None:
        """A typo should not surface as a raw DuckDB BinderException."""
        import pytest as _pytest

        from smart_traffic.aggregation import Aggregator

        aggregator = Aggregator()
        with _pytest.raises(ValueError, match="Unknown column"):
            aggregator.counts_by("Road_Type")
        # The message must name the column and list what is available.
        with _pytest.raises(ValueError) as caught:
            aggregator.counts_by("Side")
        assert "Side" in str(caught.value)
        assert "Available" in str(caught.value)

    def test_counts_by_still_accepts_real_columns(self, processed_table) -> None:
        from smart_traffic.aggregation import Aggregator

        aggregator = Aggregator()
        assert not aggregator.counts_by("State").empty
        assert not aggregator.counts_by("Severity").empty

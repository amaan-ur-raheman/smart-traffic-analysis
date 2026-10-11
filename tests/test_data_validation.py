"""Tests for schema and data-quality validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smart_traffic import schema
from smart_traffic.db import Database
from smart_traffic.ingestion import check_schema, read_header
from smart_traffic.schema import (
    CANONICAL_SOURCE_HEADER,
    matches_source_header,
)
from smart_traffic.synthetic import SyntheticSpec
from smart_traffic.validation import (
    is_valid_latitude,
    is_valid_longitude,
    is_valid_severity,
    validate_processed_table,
    validate_records,
)


class TestColumnNamesMatchTheRealFile:
    """Guard against a typo in the schema constants.

    The synthetic fixture generates its header *from* these constants, so a
    misspelling is invisible to every other test -- it just silently drops a
    column during ingestion. This test pins the names against the documented
    header of the shipped Kaggle file.
    """

    def test_every_expected_column_exists_in_the_source_header(self) -> None:
        missing = matches_source_header(CANONICAL_SOURCE_HEADER)
        assert missing == [], (
            f"these column names are referenced by the code but do not exist in the "
            f"real dataset header: {missing}"
        )

    def test_canonical_header_has_no_duplicates(self) -> None:
        assert len(set(CANONICAL_SOURCE_HEADER)) == len(CANONICAL_SOURCE_HEADER)

    def test_original_columns_are_a_subset_of_the_real_header(self) -> None:
        declared = set(schema.ORIGINAL_COLUMNS)
        real = set(CANONICAL_SOURCE_HEADER)
        assert declared <= real, f"not in the real file: {sorted(declared - real)}"

    def test_generated_fixture_uses_the_real_header(self) -> None:
        from smart_traffic.synthetic import generate_accidents

        frame = generate_accidents(SyntheticSpec(rows=5, seed=1))
        assert set(frame.columns) == set(CANONICAL_SOURCE_HEADER)

    def test_numeric_columns_are_all_real_columns(self) -> None:
        assert set(schema.NUMERIC_COLUMNS) <= set(CANONICAL_SOURCE_HEADER)


class TestRequiredColumns:
    """The ingestion schema gate."""

    def test_full_original_schema_is_accepted(self) -> None:
        assert check_schema(schema.ORIGINAL_COLUMNS) == []

    def test_missing_required_column_is_reported(self) -> None:
        columns = [c for c in CANONICAL_SOURCE_HEADER if c != schema.SEVERITY]
        assert schema.SEVERITY in check_schema(columns)

    def test_all_required_columns_missing_are_reported(self) -> None:
        assert sorted(check_schema([])) == sorted(schema.REQUIRED_COLUMNS)

    def test_optional_columns_do_not_block_processing(self) -> None:
        minimal = list(schema.REQUIRED_COLUMNS)
        assert check_schema(minimal) == []


class TestReadHeader:
    """Header reading and its failure modes."""

    def test_reads_column_names(self, tiny_csv) -> None:
        columns = read_header(tiny_csv)
        assert columns == list(CANONICAL_SOURCE_HEADER)

    def test_missing_file_raises(self, tmp_path) -> None:
        from smart_traffic.ingestion import IngestionError

        with pytest.raises(IngestionError, match="not found"):
            read_header(tmp_path / "nope.csv")


class TestValueValidators:
    """Domain rules for individual values."""

    @pytest.mark.parametrize("value", [1, 2, 3, 4, 1.0, "2"])
    def test_valid_severity(self, value) -> None:
        assert is_valid_severity(value)

    @pytest.mark.parametrize("value", [0, 5, -1, 99, None, "high", float("nan")])
    def test_invalid_severity(self, value) -> None:
        assert not is_valid_severity(value)

    @pytest.mark.parametrize("value", [25.0, 40.7128, 18.0, 71.9])
    def test_valid_latitude(self, value) -> None:
        assert is_valid_latitude(value)

    @pytest.mark.parametrize("value", [0.0, 95.0, -200.0, None, float("nan"), "north"])
    def test_invalid_latitude(self, value) -> None:
        assert not is_valid_latitude(value)

    @pytest.mark.parametrize("value", [-75.0, -122.4, -180.0, -40.0])
    def test_valid_longitude(self, value) -> None:
        assert is_valid_longitude(value)

    @pytest.mark.parametrize("value", [190.0, -400.0, 0.0, None, float("nan")])
    def test_invalid_longitude(self, value) -> None:
        assert not is_valid_longitude(value)


class TestValidateRecords:
    """In-memory record validation."""

    def test_clean_frame_has_no_problems(self, tiny_frame) -> None:
        result = validate_records(tiny_frame)
        assert result["rows"] == len(tiny_frame)
        assert result["invalid_severity"] == 0
        assert result["invalid_coordinates"] == 0
        assert result["duplicate_ids"] == 0

    def test_detects_invalid_severity(self, messy_frame) -> None:
        assert validate_records(messy_frame)["invalid_severity"] == 2

    def test_detects_invalid_coordinates(self, messy_frame) -> None:
        assert validate_records(messy_frame)["invalid_coordinates"] == 2

    def test_detects_duplicate_ids(self, messy_frame) -> None:
        assert validate_records(messy_frame)["duplicate_ids"] == 2

    def test_empty_frame_is_handled(self) -> None:
        result = validate_records(pd.DataFrame())
        assert result["rows"] == 0
        assert result["invalid_severity"] == 0

    def test_distinguishes_missing_from_invalid(self) -> None:
        frame = pd.DataFrame(
            {
                schema.ID: ["a", "b", "c"],
                schema.SEVERITY: [2.0, None, 9.0],
                schema.START_LAT: [40.0, 40.0, None],
                schema.START_LNG: [-75.0, -75.0, -75.0],
            }
        )
        result = validate_records(frame)
        # Two rows are out of domain (NaN severity, severity 9); one is merely missing.
        assert result["invalid_severity"] == 2


class TestValidateProcessedTable:
    """Aggregate validation against a built table."""

    def test_report_is_populated(self, processed_table) -> None:
        from smart_traffic.db import DB

        report = validate_processed_table(DB)
        assert report.row_count > 0
        assert len(report.checks) > 5
        assert report.is_healthy

    def test_reports_row_count(self, processed_table) -> None:
        from smart_traffic.db import DB

        report = validate_processed_table(DB)
        total = next(c for c in report.checks if c.name == "Total records")
        assert int(total.value) == report.row_count

    def test_unique_ids_match_row_count(self, processed_table) -> None:
        from smart_traffic.db import DB

        report = validate_processed_table(DB)
        unique = next(c for c in report.checks if c.name == "Unique accident IDs")
        assert int(unique.value) == report.row_count

    def test_reference_comparison_is_labelled(self, processed_table) -> None:
        from smart_traffic.db import DB

        report = validate_processed_table(DB)
        with_reference = [c for c in report.checks if c.reference is not None]
        assert with_reference, "expected at least one reference comparison"
        # A synthetic dataset will not match the published totals, and that must
        # be visible rather than hidden.
        assert all(c.matches_reference is False for c in with_reference)

    def test_dataframe_export(self, processed_table) -> None:
        from smart_traffic.db import DB

        frame = validate_processed_table(DB).to_dataframe()
        assert {"Check", "Value", "Status"}.issubset(frame.columns)
        assert not frame.empty


class TestCorruptProcessedFile:
    """A truncated Parquet must be reported, not crash the app.

    Regression test. ``Database.is_available`` only checked that the file existed
    and was non-empty, so a build interrupted partway left a file that passed the
    gate and then failed inside the first query. That escaped to Streamlit as a
    redacted "unexpected error" with nothing pointing at the file.
    """

    def _handle(self, tmp_path: Path) -> Database:
        return Database(
            db_path=tmp_path / "corrupt.duckdb",
            parquet_path=tmp_path / "accidents.parquet",
        )

    def test_existing_non_empty_file_still_passes_is_available(self, tmp_path: Path):
        handle = self._handle(tmp_path)
        handle.parquet_path.write_bytes(b"not parquet, but present and non-empty")
        assert handle.is_available, "precondition: the existence/size gate accepts it"

    def test_ensure_views_raises_actionable_error(self, tmp_path: Path):
        from smart_traffic.db import DatasetNotAvailableError

        handle = self._handle(tmp_path)
        handle.parquet_path.write_bytes(b"not parquet, but present and non-empty")

        with pytest.raises(DatasetNotAvailableError) as caught:
            handle.ensure_views()
        message = str(caught.value)
        assert "could not be read" in message
        assert "uv run sta build" in message

    def test_query_raises_rather_than_leaking_a_duckdb_error(self, tmp_path: Path):
        from smart_traffic.db import DatasetNotAvailableError

        handle = self._handle(tmp_path)
        handle.parquet_path.write_bytes(b"garbage")
        with pytest.raises(DatasetNotAvailableError):
            handle.sql("SELECT COUNT(*) FROM accidents")

    def test_views_are_not_marked_registered_after_failure(self, tmp_path: Path):
        """A failed bind must not leave the handle claiming success."""
        from smart_traffic.db import DatasetNotAvailableError

        handle = self._handle(tmp_path)
        handle.parquet_path.write_bytes(b"garbage")
        with pytest.raises(DatasetNotAvailableError):
            handle.ensure_views()
        assert handle._registered is False

    def test_ui_reports_the_dataset_as_not_ready(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        import smart_traffic.ui as ui_module

        handle = self._handle(tmp_path)
        handle.parquet_path.write_bytes(b"garbage")
        # ui imports DB by name, so the seam is ui_module.DB.
        monkeypatch.setattr(ui_module, "DB", handle)

        assert ui_module.dataset_ready() is False
        assert "could not be read" in ui_module.dataset_error()

    def test_ui_reports_ready_for_a_valid_dataset(self, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe must not reject a genuinely good file."""
        import smart_traffic.ui as ui_module
        from smart_traffic.synthetic import SyntheticSpec, generate_accidents

        frame = generate_accidents(SyntheticSpec(rows=50, seed=3))
        good = tmp_path / "accidents.parquet"
        frame.to_parquet(good, index=False)

        monkeypatch.setattr(
            ui_module, "DB",
            Database(db_path=tmp_path / "ok.duckdb", parquet_path=good),
        )
        assert ui_module.dataset_ready() is True
        assert ui_module.dataset_error() == ""

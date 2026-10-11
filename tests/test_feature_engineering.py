"""Tests for timestamp parsing, feature engineering and the leakage policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smart_traffic import schema
from smart_traffic.config import SETTINGS
from smart_traffic.feature_engineering import (
    DAY_NAMES,
    MONTH_NAMES,
    add_derived_features,
    assert_no_leakage,
    compute_duration_minutes,
    day_night_category,
    is_rush_hour_mask,
    parse_mixed_timestamps,
    rush_hour_definition_text,
)


class TestTimestampParsing:
    """Mixed-format and offset-bearing timestamps."""

    def test_parses_standard_format(self) -> None:
        series = pd.Series(["2016-02-14 22:00:00.000000"])
        result = parse_mixed_timestamps(series)
        assert result.iloc[0] == pd.Timestamp("2016-02-14 22:00:00")

    def test_parses_iso_t_format(self) -> None:
        result = parse_mixed_timestamps(pd.Series(["2018-06-15T14:30:00"]))
        assert result.iloc[0] == pd.Timestamp("2018-06-15 14:30:00")

    def test_parses_without_microseconds(self) -> None:
        result = parse_mixed_timestamps(pd.Series(["2018-06-15 14:30:00"]))
        assert result.iloc[0] == pd.Timestamp("2018-06-15 14:30:00")

    def test_offset_preserves_local_wall_clock(self) -> None:
        """A -05:00 offset must not shift the hour into UTC.

        Local wall-clock time is what hour-of-day analytics depend on, so the
        offset is stripped rather than applied.
        """
        result = parse_mixed_timestamps(pd.Series(["2018-06-15 14:30:00-05:00"]))
        assert result.iloc[0] == pd.Timestamp("2018-06-15 14:30:00")
        assert result.dt.hour.iloc[0] == 14

    def test_mixed_formats_in_one_column(self) -> None:
        series = pd.Series(
            [
                "2016-02-14 22:00:00.000000",
                "2018-06-15T14:30:00",
                "2019-01-01 00:00:00",
                "2020-07-04 12:00:00-04:00",
            ]
        )
        result = parse_mixed_timestamps(series)
        assert result.notna().all()
        assert result.dt.year.tolist() == [2016, 2018, 2019, 2020]

    def test_unparseable_values_become_nat(self) -> None:
        result = parse_mixed_timestamps(pd.Series(["2016-02-14 22:00:00", "garbage", None]))
        assert result.notna().sum() == 1
        assert result.isna().sum() == 2

    def test_empty_series(self) -> None:
        result = parse_mixed_timestamps(pd.Series([], dtype="object"))
        assert len(result) == 0

    def test_date_only_values_do_not_get_mangled(self) -> None:
        """A regex stripping offsets must not eat part of a date."""
        result = parse_mixed_timestamps(pd.Series(["2016-12-14 08:00:00.000000"]))
        assert result.iloc[0] == pd.Timestamp("2016-12-14 08:00:00")


class TestRushHour:
    """The documented rush-hour definition."""

    def test_documented_windows(self) -> None:
        hours = pd.Series(range(24))
        result = is_rush_hour_mask(hours)
        # Both stated endpoints are inclusive: 7, 8, 9 and 16, 17, 18.
        assert result[7] and result[8] and result[9]
        assert result[16] and result[17] and result[18]
        assert not result[6] and not result[10]
        assert not result[15] and not result[19]

    def test_midnight_is_not_rush_hour(self) -> None:
        assert not is_rush_hour_mask(pd.Series([0]))[0]
        assert not is_rush_hour_mask(pd.Series([23]))[0]

    def test_missing_hour_is_not_rush_hour(self) -> None:
        result = is_rush_hour_mask(pd.Series([np.nan, 8]))
        assert not result[0]
        assert result[1]

    def test_definition_text_mentions_the_windows(self) -> None:
        text = rush_hour_definition_text()
        assert f"{SETTINGS.rush_hour_morning[0]:02d}:00" in text
        assert "Rush hour" in text


class TestDayNight:
    """Day/night classification from the source field."""

    def test_known_values(self) -> None:
        series = pd.Series(["Day", "Night", "day", " NIGHT ", "weird"])
        result = day_night_category(series)
        assert list(result) == ["Day", "Night", "Day", "Night", "Unknown"]

    def test_absent_column_yields_unknown(self) -> None:
        result = day_night_category(None)
        assert set(result.unique()) == {"Unknown"}

    def test_unknown_is_not_folded_into_day(self) -> None:
        """Folding Unknown into Day would invent data."""
        result = day_night_category(pd.Series(["ambiguous"]))
        assert result.iloc[0] == "Unknown"


class TestDuration:
    """Duration computation and negative-duration handling."""

    def test_computes_minutes(self) -> None:
        start = parse_mixed_timestamps(pd.Series(["2021-01-04 07:00:00.000000"]))
        end = parse_mixed_timestamps(pd.Series(["2021-01-04 07:45:00.000000"]))
        assert compute_duration_minutes(start, end).iloc[0] == pytest.approx(45.0)

    def test_negative_duration_becomes_missing(self) -> None:
        start = parse_mixed_timestamps(pd.Series(["2021-01-04 07:00:00.000000"]))
        end = parse_mixed_timestamps(pd.Series(["2021-01-04 06:00:00.000000"]))
        assert pd.isna(compute_duration_minutes(start, end).iloc[0])

    def test_missing_end_time_becomes_missing(self) -> None:
        start = parse_mixed_timestamps(pd.Series(["2021-01-04 07:00:00.000000"]))
        end = parse_mixed_timestamps(pd.Series([None]))
        assert pd.isna(compute_duration_minutes(start, end).iloc[0])

    def test_no_end_column_at_all(self) -> None:
        start = parse_mixed_timestamps(pd.Series(["2021-01-04 07:00:00.000000"]))
        assert compute_duration_minutes(start, None).isna().all()


class TestDerivedFeatures:
    """End-to-end feature derivation on hand-checked timestamps."""

    def test_calendar_fields(self, known_temporal_frame) -> None:
        result = add_derived_features(known_temporal_frame)
        assert list(result["Year"]) == [2021] * 7
        assert list(result["Month"]) == [1] * 7
        assert list(result["Hour"]) == [7, 8, 17, 12, 18, 3, 23]
        assert list(result["Day_Name"]) == [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        assert list(result["Month_Name"]) == ["Jan"] * 7

    def test_weekend_flags(self, known_temporal_frame) -> None:
        result = add_derived_features(known_temporal_frame)
        assert list(result["Is_Weekend"]) == [
            False, False, False, False, False, True, True
        ]

    def test_rush_hour_flags(self, known_temporal_frame) -> None:
        result = add_derived_features(known_temporal_frame)
        # 07, 08, 17 and 18 are all in-window; 12, 03 and 23 are not.
        assert list(result["Is_Rush_Hour"]) == [
            True, True, True, False, True, False, False
        ]

    def test_dayofweek_index(self, known_temporal_frame) -> None:
        result = add_derived_features(known_temporal_frame)
        assert list(result["DayOfWeek"]) == [0, 1, 2, 3, 4, 5, 6]
        assert DAY_NAMES[0] == "Monday"

    def test_durations(self, known_temporal_frame) -> None:
        result = add_derived_features(known_temporal_frame)
        assert list(result[schema.DURATION_MINUTES]) == [30, 15, 30, 120, 5, 60, 30]
        assert result[schema.DURATION_HOURS].iloc[3] == pytest.approx(2.0)

    def test_day_night_categories(self, known_temporal_frame) -> None:
        result = add_derived_features(known_temporal_frame)
        assert list(result["Day_Night"]) == [
            "Night", "Day", "Day", "Day", "Night", "Night", "Night"
        ]
        # Is_Night is nullable so Unknown survives to the imputer.
        assert result["Is_Night"].iloc[0] == 1
        assert result["Is_Night"].iloc[1] == 0

    def test_is_night_is_nullable(self, tiny_frame) -> None:
        result = add_derived_features(tiny_frame)
        assert str(result["Is_Night"].dtype) == "Int8"

    def test_input_frame_is_not_mutated(self, known_temporal_frame) -> None:
        before = known_temporal_frame.copy()
        add_derived_features(known_temporal_frame)
        pd.testing.assert_frame_equal(known_temporal_frame, before)

    def test_severity_is_integer_typed(self, known_temporal_frame) -> None:
        assert str(add_derived_features(known_temporal_frame)["Severity"].dtype) == "Int8"

    def test_handles_missing_end_time_column(self, tiny_frame) -> None:
        result = add_derived_features(tiny_frame.drop(columns=[schema.END_TIME]))
        assert schema.DURATION_MINUTES not in result.columns
        assert "Year" in result.columns

    def test_month_names_cover_all_twelve(self) -> None:
        assert len(MONTH_NAMES) == 12
        assert MONTH_NAMES[0] == "Jan" and MONTH_NAMES[-1] == "Dec"

    def test_unparseable_timestamps_produce_null_calendar_fields(self, messy_frame) -> None:
        result = add_derived_features(messy_frame)
        assert result["Start_Time"].isna().sum() == 1
        null_rows = result[result["Start_Time"].isna()]
        assert null_rows["Hour"].isna().all()


class TestLeakagePolicy:
    """The enforced boundary around post-onset information."""

    @pytest.mark.parametrize(
        "feature",
        [schema.ID, schema.SEVERITY, schema.DURATION_MINUTES, schema.DURATION_HOURS,
         schema.END_TIME],
    )
    def test_leaking_features_are_rejected(self, feature) -> None:
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage(["Hour", feature])

    def test_safe_features_pass(self) -> None:
        assert_no_leakage(["Hour", "Month", "State", "Temperature(F)"])

    def test_duration_hours_rejected_via_policy(self) -> None:
        from smart_traffic.modeling import LeakagePolicy

        # Start_Lat is excluded by policy but is not caught by the earlier
        # generic leakage guard, so this exercises the policy message itself.
        with pytest.raises(ValueError, match="excluded by the leakage policy"):
            LeakagePolicy().validate(["Hour", "Start_Lat"])

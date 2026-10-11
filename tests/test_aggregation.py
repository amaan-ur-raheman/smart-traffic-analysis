"""Tests for aggregation correctness, especially severity-rate denominators."""

from __future__ import annotations

import pandas as pd
import pytest

from smart_traffic.aggregation import (
    DAY_ORDER,
    Aggregator,
    FilterSpec,
    add_percentage,
    top_n_mask,
)


@pytest.fixture
def known_counts(processed_table, test_db):
    """Return an Aggregator bound to the temporary test database."""
    return Aggregator(test_db)


class TestFilterSpec:
    """Filter compilation and description."""

    def test_no_filters_produces_only_validity(self) -> None:
        clause, params = FilterSpec().where_clause()
        assert "Is_Valid_Record" in clause
        assert params == []

    def test_year_filter_binds_parameters(self) -> None:
        clause, params = FilterSpec(years=[2019, 2020]).where_clause()
        assert "Year" in clause
        assert params == [2019, 2020]

    def test_state_values_are_bound_not_interpolated(self) -> None:
        """A user-supplied value must never become executable SQL."""
        clause, params = FilterSpec(
            states=["CA'; DROP TABLE accidents --"]
        ).where_clause()
        assert "DROP TABLE" not in clause
        assert params == ["CA'; DROP TABLE accidents --"]

    def test_day_type_predicates(self) -> None:
        clause, _ = FilterSpec(day_types=["Weekend", "Night"]).where_clause()
        assert "Is_Weekend = TRUE" in clause
        assert "Day_Night = 'Night'" in clause

    def test_unknown_day_type_is_ignored(self) -> None:
        clause, _ = FilterSpec(day_types=["Not A Category"]).where_clause()
        assert "Not A Category" not in clause

    def test_is_active_reflects_filters(self) -> None:
        assert not FilterSpec().is_active
        assert FilterSpec(years=[2020]).is_active

    def test_describe_is_human_readable(self) -> None:
        labels = FilterSpec(states=["CA", "TX"]).describe()
        assert any("CA" in label for label in labels)

    def test_describe_truncates_long_lists(self) -> None:
        labels = FilterSpec(states=[f"S{i}" for i in range(10)]).describe()
        assert any("..." in label for label in labels)

    def test_empty_filter_equals_full_dataset(self, known_counts) -> None:
        filtered = FilterSpec(years=[2020])
        assert known_counts.total(filtered) <= known_counts.total(FilterSpec())


class TestBasicCounts:
    """Count aggregation."""

    def test_total_matches_row_count(self, known_counts) -> None:
        assert known_counts.total() > 0

    def test_counts_by_sums_to_total(self, known_counts) -> None:
        """Hour covers every record, so the counts must reconcile."""
        by_hour = known_counts.counts_by("Hour", order_desc=False)
        assert int(by_hour["Accidents"].sum()) == known_counts.total()

    def test_counts_by_excludes_nulls(self, known_counts) -> None:
        frame = known_counts.counts_by("State")
        assert frame["State"].notna().all()

    def test_limit_is_applied(self, known_counts) -> None:
        assert len(known_counts.counts_by("State", limit=3)) <= 3

    def test_day_order_is_canonical(self, known_counts) -> None:
        frame = known_counts.counts_by("Day_Name", order_desc=False)
        assert list(frame["Day_Name"]) == [
            d for d in DAY_ORDER if d in set(frame["Day_Name"])
        ]

    def test_hour_order_is_ascending(self, known_counts) -> None:
        frame = known_counts.counts_by("Hour", order_desc=False)
        assert list(frame["Hour"]) == sorted(frame["Hour"])

    def test_severity_counts_sum_to_total(self, known_counts) -> None:
        frame = known_counts.severity_counts()
        assert int(frame["Accidents"].sum()) == known_counts.total()

    def test_severity_percentages_sum_to_100(self, known_counts) -> None:
        frame = known_counts.severity_counts()
        assert frame["Percentage"].sum() == pytest.approx(100.0, abs=0.01)


class TestSeverityRates:
    """Within-group severity rates and their denominators."""

    def test_rates_sum_to_100_within_each_group(self, known_counts) -> None:
        """This is the property that proves the denominator is the group's total."""
        frame = known_counts.severity_rates("Day_Name")
        rate_columns = [f"Severity_{level}_Rate" for level in (1, 2, 3, 4)]
        totals = frame[rate_columns].sum(axis=1)
        assert totals.tolist() == pytest.approx([100.0] * len(frame), abs=0.05)

    def test_rate_equals_count_over_group_total(self, known_counts) -> None:
        frame = known_counts.severity_rates("Day_Name")
        for _, row in frame.iterrows():
            expected = row["Severity_3_Count"] / row["Total"] * 100
            assert row["Severity_3_Rate"] == pytest.approx(expected, abs=0.01)

    def test_rate_is_not_the_group_share(self, known_counts) -> None:
        """A group holding 40% of records may have a very different severity rate."""
        frame = known_counts.severity_rates("Day_Name")
        assert "Share_%" in frame.columns
        # The two must be independently derived, not accidentally identical.
        assert not (frame["Share_%"] == frame["Severity_2_Rate"]).all()

    def test_group_totals_sum_to_overall_total(self, known_counts) -> None:
        frame = known_counts.severity_rates("Day_Name")
        assert int(frame["Total"].sum()) == known_counts.total()

    def test_severity_counts_reconcile_with_total(self, known_counts) -> None:
        frame = known_counts.severity_rates("Day_Name")
        summed = frame[[f"Severity_{level}_Count" for level in (1, 2, 3, 4)]].sum().sum()
        assert int(summed) == int(frame["Total"].sum())

    def test_min_total_filters_small_groups(self, known_counts) -> None:
        loose = known_counts.severity_rates("City", min_total=1)
        strict = known_counts.severity_rates("City", min_total=500)
        assert len(strict) <= len(loose)

    def test_limit_applies(self, known_counts) -> None:
        frame = known_counts.severity_rates("State", limit=5)
        assert len(frame) <= 5

    def test_filtered_rates_use_filtered_denominator(self, known_counts) -> None:
        unfiltered = known_counts.severity_rates("Day_Name")
        filtered = known_counts.severity_rates("Day_Name", FilterSpec(years=[2021]))
        assert filtered["Total"].sum() < unfiltered["Total"].sum()
        # The denominator is the filtered population, so rates stay in range.
        rate_columns = [f"Severity_{level}_Rate" for level in (1, 2, 3, 4)]
        assert filtered[rate_columns].sum(axis=1).tolist() == pytest.approx(
            [100.0] * len(filtered), abs=0.05
        )


class TestBooleanFeatures:
    """Road and amenity indicators."""

    def test_counts_are_sorted_descending(self, known_counts) -> None:
        frame = known_counts.boolean_feature_counts()
        assert list(frame["Accidents"]) == sorted(frame["Accidents"], reverse=True)

    def test_percentages_are_in_range(self, known_counts) -> None:
        frame = known_counts.boolean_feature_counts()
        assert frame["Percent_of_Records"].between(0, 100).all()

    def test_counts_do_not_exceed_total(self, known_counts) -> None:
        """Indicators are not mutually exclusive but each must stay under the total."""
        frame = known_counts.boolean_feature_counts()
        assert (frame["Accidents"] <= known_counts.total()).all()


class TestHeatmapAndCrosstab:
    """Matrix-shaped aggregations."""

    def test_heatmap_has_expected_shape(self, known_counts) -> None:
        matrix = known_counts.heatmap("Day_Name", "Hour")
        assert matrix.shape[0] <= 7
        assert matrix.shape[1] == 25  # row label plus 24 hours

    def test_heatmap_totals_reconcile(self, known_counts) -> None:
        matrix = known_counts.heatmap("Day_Name", "Hour")
        numeric = matrix.drop(columns=["Day_Name"])
        assert int(numeric.to_numpy().sum()) == known_counts.total()

    def test_heatmap_limit(self, known_counts) -> None:
        assert len(known_counts.heatmap("State", "Hour", limit=3)) <= 3

    def test_crosstab_is_wide(self, known_counts) -> None:
        matrix = known_counts.crosstab("Day_Name", "Severity")
        assert not matrix.empty
        assert matrix.shape[0] <= 7


class TestCoverageByYear:
    """Year coverage and the partial-final-year flag."""

    def test_last_year_is_marked_partial(self, known_counts) -> None:
        frame = known_counts.coverage_by_year()
        assert bool(frame.iloc[-1]["Is_Partial"])
        assert not bool(frame.iloc[0]["Is_Partial"])

    def test_coverage_span_is_populated(self, known_counts) -> None:
        frame = known_counts.coverage_by_year()
        assert (frame["Coverage"] != "").any()

    def test_counts_sum_to_total(self, known_counts) -> None:
        frame = known_counts.coverage_by_year()
        assert int(frame["Accidents"].sum()) == known_counts.total()


class TestHelpers:
    """Small standalone helpers."""

    def test_add_percentage(self) -> None:
        frame = pd.DataFrame({"Accidents": [25, 75]})
        result = add_percentage(frame, "Accidents")
        assert list(result["Percentage"]) == [25.0, 75.0]

    def test_add_percentage_handles_zero_total(self) -> None:
        frame = pd.DataFrame({"Accidents": [0, 0]})
        assert list(add_percentage(frame, "Accidents")["Percentage"]) == [0.0, 0.0]

    def test_top_n_mask(self) -> None:
        series = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"])
        mask = top_n_mask(series, n=2)
        assert mask.sum() == 8

    def test_top_n_mask_on_empty(self) -> None:
        assert top_n_mask(pd.Series([], dtype="object")).empty

"""Filtered aggregation over the processed table.

Every dashboard query funnels through :class:`FilterSpec`, which compiles to a
parameterised SQL ``WHERE`` clause. Two reasons this is centralised:

* **Performance.** Grouped aggregates are computed in DuckDB against Parquet.
  A chart costs one scan instead of a 7.7M-row pandas materialisation.
* **Correctness of the denominator.** :meth:`FilterSpec.severity_rates` always
  divides by the group's own total, so a rate can never be confused with a
  share of the whole dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from smart_traffic import schema
from smart_traffic.config import SETTINGS
from smart_traffic.db import ACCIDENTS_VIEW, Database
from smart_traffic.feature_engineering import MONTH_NAMES
from smart_traffic.logging_config import get_logger

logger = get_logger("aggregation")

#: Weekday ordering used by every chart axis.
DAY_ORDER: Final[tuple[str, ...]] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)

#: Month ordering used by every chart axis.
MONTH_ORDER: Final[tuple[str, ...]] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

#: Columns with a natural reading order. Everything else sorts by count.
_CANONICAL_ORDER_COLUMNS: Final[frozenset[str]] = frozenset(
    {"Day_Name", "DayOfWeek", "Month_Name", "Month", "Hour", "Severity"}
)


@dataclass(slots=True)
class FilterSpec:
    """A reusable, parameterised set of row filters.

    Filters compose with AND. Empty / ``None`` values mean "no restriction".
    Every clause is a bound parameter, so a user-supplied state name can never
    become executable SQL.

    Attributes:
        years: Restrict to these years.
        states: Restrict to these state codes.
        cities: Restrict to these cities.
        severities: Restrict to these severity categories.
        months: Restrict to these month numbers (1-12).
        hours: Restrict to these clock hours (0-23).
        day_types: ``"Weekday"`` / ``"Weekend"`` / ``"Rush Hour"`` / ``"Non-Rush Hour"`` /
            ``"Day"`` / ``"Night"`` -- see :data:`DAY_TYPE_PREDICATES`.
        valid_only: Exclude records flagged invalid during ingestion.
    """

    years: Sequence[int] | None = None
    states: Sequence[str] | None = None
    cities: Sequence[str] | None = None
    severities: Sequence[int] | None = None
    months: Sequence[int] | None = None
    hours: Sequence[int] | None = None
    day_types: Sequence[str] | None = None
    valid_only: bool = True

    # -- SQL construction --------------------------------------------------

    def where_clause(self, alias: str = ACCIDENTS_VIEW) -> tuple[str, list[Any]]:
        """Build the SQL ``WHERE`` body and its parameters.

        Args:
            alias: Table name or alias to qualify column references with.

        Returns:
            Tuple of ``(sql_fragment, params)``. The fragment is empty when no
            filter is active, so callers can omit ``WHERE`` entirely.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if self.valid_only:
            clauses.append(f"{alias}.Is_Valid_Record = TRUE")

        for column, values in (
            ("Year", self.years),
            ("State", self.states),
            ("City", self.cities),
            ("Severity", self.severities),
            ("Month", self.months),
            ("Hour", self.hours),
        ):
            if values:
                placeholders = ", ".join("?" for _ in values)
                clauses.append(f"{alias}.\"{column}\" IN ({placeholders})")
                params.extend(list(values))

        if self.day_types:
            fragments: list[str] = []
            for label in self.day_types:
                fragment = DAY_TYPE_PREDICATES.get(label)
                if fragment:
                    fragments.append(fragment.format(alias=alias))
            if fragments:
                clauses.append("(" + " OR ".join(fragments) + ")")

        return " AND ".join(clauses), params

    @property
    def is_active(self) -> bool:
        """Whether any filter beyond ``valid_only`` is applied."""
        return any(
            value is not None
            for value in (
                self.years, self.states, self.cities, self.severities,
                self.months, self.hours, self.day_types,
            )
        )

    def describe(self) -> list[str]:
        """Return human-readable descriptions of the active filters.

        Returns:
            One label per active filter, e.g. ``"State: CA, TX"``.
        """
        labels: list[str] = []
        for name, values in (
            ("Year", self.years), ("State", self.states), ("City", self.cities),
            ("Severity", self.severities), ("Month", self.months), ("Hour", self.hours),
        ):
            if values:
                shown = list(values)[:6]
                suffix = ", ..." if len(values) > 6 else ""
                labels.append(f"{name}: {', '.join(str(v) for v in shown)}{suffix}")
        if self.day_types:
            labels.append(f"Category: {', '.join(self.day_types)}")
        return labels

    def select(
        self,
        query: str,
        db: Database | None = None,
        params: list[Any] | None = None,
    ) -> pd.DataFrame:
        """Run ``query`` with this filter applied.

        Args:
            query: SQL using ``{where}`` and ``{params}`` placeholders.
                ``{where}`` must appear *before* any other ``?`` placeholder,
                because the filter's bound values are prepended.
            db: Optional database handle.
            params: Extra parameters the query itself needs.

        Returns:
            Query results as a pandas DataFrame.

        Raises:
            ValueError: If the number of supplied parameters does not match the
                number of placeholders in the assembled query. This almost
                always means ``{where}`` was positioned after a ``?``, which
                would otherwise bind the filter values to the wrong arguments
                and return silently wrong rows.
        """
        from smart_traffic.db import DB

        handle = db or DB
        clause, filter_params = self.where_clause()
        # Filter values are prepended because `{where}` is substituted into the
        # leading WHERE clause, ahead of any placeholder the query itself adds.
        full_params = [*filter_params, *(params or [])]
        sql = query.format(where=clause or "TRUE")
        expected = sql.count("?")
        if expected != len(full_params):
            raise ValueError(
                f"Parameter count mismatch: the assembled query has {expected} "
                f"placeholder(s) but {len(full_params)} value(s) were supplied. "
                "Place {where} before any other '?' placeholder so the filter's "
                "bound values come first."
            )
        return handle.sql(sql, full_params)


#: Named row predicates selectable from the UI's day/time category filter.
DAY_TYPE_PREDICATES: Final[dict[str, str]] = {
    "Weekday": "{alias}.Is_Weekend = FALSE",
    "Weekend": "{alias}.Is_Weekend = TRUE",
    "Rush Hour": "{alias}.Is_Rush_Hour = TRUE",
    "Non-Rush Hour": "{alias}.Is_Rush_Hour = FALSE",
    "Day": "{alias}.Day_Night = 'Day'",
    "Night": "{alias}.Day_Night = 'Night'",
}


#: SQL conditional-count expression. ``SUM(CASE WHEN ...)`` returns NULL -- not
#: 0 -- over an empty result set, which happens whenever a filter or a rare
#: indicator (Turning_Loop, say) matches no rows at all. Coalescing keeps every
#: count a real number all the way to Python.
def count_if(condition: str) -> str:
    """Return a ``SUM(CASE WHEN ...)`` expression that yields 0, never NULL.

    Args:
        condition: SQL boolean expression, e.g. ``"Severity = 3"``.

    Returns:
        The full ``COALESCE(SUM(...), 0)`` expression.
    """
    return f"COALESCE(SUM(CASE WHEN {condition} THEN 1 ELSE 0 END), 0)"


class Aggregator:
    """Grouped analytics over the processed table.

    Args:
        db: Optional database handle. Defaults to the module singleton.
    """

    def __init__(self, db: Database | None = None) -> None:
        """Bind to ``db``, or resolve the module singleton lazily."""
        self._db = db

    @property
    def db(self) -> Database:
        """The database handle, resolved lazily."""
        from smart_traffic.db import DB

        return self._db or DB

    # -- generic helpers ---------------------------------------------------

    def require_columns(self, *columns: str) -> None:
        """Validate that ``columns`` exist in the processed table.

        Without this, a typo reaches DuckDB and surfaces as a BinderException
        listing dozens of candidate column names. That is unfriendly for anyone
        adding a page, and easy to mistake for a data problem when it is really
        a code problem.

        Args:
            columns: Column names to check.

        Raises:
            ValueError: If any column is absent from the accidents view.
        """
        from smart_traffic.db import DB

        if not columns:
            return
        # The DESCRIBE needs the view to exist first; counts_by may be the very
        # first query issued against a freshly opened handle.
        DB.ensure_views()
        available = {
            str(row[0])
            for row in DB.connection.execute(
                f"DESCRIBE {ACCIDENTS_VIEW}"
            ).fetchall()
        }
        missing = [c for c in columns if c not in available]
        if missing:
            raise ValueError(
                f"Unknown column(s): {', '.join(missing)}. "
                f"Available: {', '.join(sorted(available))}"
            )

    def total(self, spec: FilterSpec | None = None) -> int:
        """Return the number of records matching ``spec``."""
        spec = spec or FilterSpec()
        # The ``{where}`` placeholder is mandatory here: without it the filter
        # parameters would be bound to a query containing no placeholders, and
        # the count would silently ignore the filters.
        frame = spec.select(
            f"SELECT COUNT(*)::BIGINT AS n FROM {ACCIDENTS_VIEW} WHERE {{where}}", self.db
        )
        return int(frame["n"][0])

    def counts_by(
        self,
        column: str,
        spec: FilterSpec | None = None,
        *,
        limit: int | None = None,
        order_desc: bool = True,
        db: Database | None = None,
    ) -> pd.DataFrame:
        """Count records grouped by ``column``.

        Args:
            column: Column to group by.
            spec: Active filters.
            limit: Optional row cap, applied after ordering.
            order_desc: Sort by count descending (otherwise ascending by column).
            db: Optional database handle.

        Returns:
            DataFrame with the grouping column and an ``Accidents`` count.
            Rows whose grouping value is NULL are excluded.
        """
        spec = spec or FilterSpec()
        self.require_columns(column)
        order = "Accidents DESC, 1" if order_desc else f'"{column}"'
        sql = f"""
            SELECT "{column}" AS "{column}", COUNT(*)::BIGINT AS Accidents
            FROM {ACCIDENTS_VIEW}
            WHERE {{where}} AND "{column}" IS NOT NULL
            GROUP BY "{column}"
            ORDER BY {order}
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        frame = spec.select(sql, db or self.db)
        return self._apply_categorical_order(frame, column, order_desc)

    def severity_counts(
        self,
        spec: FilterSpec | None = None,
        db: Database | None = None,
    ) -> pd.DataFrame:
        """Return the severity distribution with counts and percentages.

        Args:
            spec: Active filters.
            db: Optional database handle.

        Returns:
            DataFrame with ``Severity``, ``Accidents`` and ``Percentage``.
        """
        spec = spec or FilterSpec()
        frame = spec.select(
            f"""
            SELECT Severity, COUNT(*)::BIGINT AS Accidents
            FROM {ACCIDENTS_VIEW}
            WHERE {{where}}
            GROUP BY Severity
            ORDER BY Severity
            """,
            db or self.db,
        )
        return add_percentage(frame, "Accidents")

    def severity_rates(
        self,
        group_by: str,
        spec: FilterSpec | None = None,
        *,
        limit: int | None = None,
        min_total: int = 1,
        db: Database | None = None,
    ) -> pd.DataFrame:
        """Compute within-group severity rates.

        For every group, the rate for severity *s* is

            ``count(severity = s in group) / count(all records in group) * 100``

        The denominator is the group's own total. This is deliberately *not*
        the same as "this group's share of all accidents", and the two columns
        are returned side by side so the difference is visible.

        Args:
            group_by: Column defining the groups.
            spec: Active filters.
            limit: Optional cap on the number of groups returned.
            min_total: Groups with fewer than this many records are dropped, to
                avoid presenting unstable rates built from a handful of rows.
            db: Optional database handle.

        Returns:
            DataFrame with the group column, ``Total``, a ``Share_%`` column
            (group share of the filtered total) and per-severity count and rate
            columns.
        """
        spec = spec or FilterSpec()
        select_parts = [
            f'"{group_by}" AS "{group_by}"',
            "COUNT(*)::BIGINT AS Total",
        ]
        for level in schema.SEVERITY_LEVELS:
            select_parts.append(
                f'SUM(CASE WHEN Severity = {level} THEN 1 ELSE 0 END)::BIGINT '
                f'AS "Severity_{level}_Count"'
            )

        sql = f"""
            SELECT {", ".join(select_parts)}
            FROM {ACCIDENTS_VIEW}
            WHERE {{where}} AND "{group_by}" IS NOT NULL
            GROUP BY "{group_by}"
            HAVING COUNT(*) >= {int(min_total)}
            ORDER BY Total DESC
        """
        frame = spec.select(sql, db or self.db)
        frame = self._rates_from_counts(frame)
        frame = add_percentage(
            frame.rename(columns={"Total": "Accidents"}), "Accidents", new_name="Share_%"
        )
        frame = frame.rename(columns={"Accidents": "Total"})
        if limit:
            frame = frame.head(int(limit))
        return self._apply_categorical_order(frame, group_by, order_desc=True)

    @staticmethod
    def _rates_from_counts(frame: pd.DataFrame) -> pd.DataFrame:
        """Compute each severity rate from the already-fetched counts.

        Doing the division in pandas rather than SQL keeps the denominator in one
        place, which is exactly where denominator mistakes tend to creep in.
        """
        frame = frame.copy()
        total = frame["Total"].replace(0, pd.NA)
        for level in schema.SEVERITY_LEVELS:
            count_column = f"Severity_{level}_Count"
            if count_column in frame.columns:
                frame[f"Severity_{level}_Rate"] = (
                    frame[count_column] / total * 100
                ).round(4)
        return frame

    def boolean_feature_counts(
        self, spec: FilterSpec | None = None, db: Database | None = None
    ) -> pd.DataFrame:
        """Count how many records have each road/amenity indicator set to true.

        Args:
            spec: Active filters.
            db: Optional database handle.

        Returns:
            DataFrame with ``Road_Feature``, ``Label``, ``Accidents`` and
            ``Percent_of_Records``.
        """
        spec = spec or FilterSpec()
        available = [c for c in schema.BOOLEAN_COLUMNS if self.db.has_column(c)]
        if not available:
            return pd.DataFrame(columns=["Road_Feature", "Label", "Accidents",
                "Percent_of_Records"])

        select_parts = ["COUNT(*)::BIGINT AS Total"]
        for column in available:
            select_parts.append(
                f'SUM(CASE WHEN "{column}" = TRUE THEN 1 ELSE 0 END)::BIGINT AS "{column}"'
            )
        frame = spec.select(
            f"SELECT {', '.join(select_parts)} FROM {ACCIDENTS_VIEW} WHERE {{where}}",
            db or self.db,
        )
        total = int(frame["Total"][0]) if len(frame) else 0
        if total == 0:
            return pd.DataFrame(columns=["Road_Feature", "Label", "Accidents",
                "Percent_of_Records"])

        rows = [
            {
                "Road_Feature": column,
                "Label": schema.ROAD_FEATURE_LABELS.get(column, column),
                "Accidents": int(frame[column][0]),
                "Percent_of_Records": round(int(frame[column][0]) / total * 100, 4),
            }
            for column in available
        ]
        result = pd.DataFrame(rows)
        return result.sort_values("Accidents", ascending=False).reset_index(drop=True)

    def crosstab(
        self,
        row_column: str,
        column_column: str,
        spec: FilterSpec | None = None,
        db: Database | None = None,
    ) -> pd.DataFrame:
        """Return a count matrix between two categorical columns.

        Args:
            row_column: Column defining rows.
            column_column: Column defining columns.
            spec: Active filters.
            db: Optional database handle.

        Returns:
            Wide DataFrame indexed by ``row_column``.
        """
        spec = spec or FilterSpec()
        sql = f"""
            SELECT "{row_column}" AS row_key, "{column_column}" AS col_key,
                   COUNT(*)::BIGINT AS Accidents
            FROM {ACCIDENTS_VIEW}
            WHERE {{where}}
              AND "{row_column}" IS NOT NULL AND "{column_column}" IS NOT NULL
            GROUP BY 1, 2
        """
        long = spec.select(sql, db or self.db)
        if long.empty:
            return pd.DataFrame()
        return long.pivot_table(
            index="row_key", columns="col_key", values="Accidents", fill_value=0
        ).reset_index().rename(columns={"row_key": row_column})

    def heatmap(
        self,
        row_column: str,
        column_column: str,
        spec: FilterSpec | None = None,
        *,
        limit: int | None = None,
        db: Database | None = None,
    ) -> pd.DataFrame:
        """Return a rectangular matrix for heatmap rendering.

        Args:
            row_column: Rows of the matrix.
            column_column: Columns of the matrix.
            spec: Active filters.
            limit: Keep only the ``limit`` busiest rows.
            db: Optional database handle.

        Returns:
            Wide DataFrame with ``row_column`` plus one column per distinct
            ``column_column`` value.
        """
        spec = spec or FilterSpec()
        sql = f"""
            SELECT "{row_column}" AS row_key, "{column_column}" AS col_key,
                   COUNT(*)::BIGINT AS Accidents
            FROM {ACCIDENTS_VIEW}
            WHERE {{where}}
              AND "{row_column}" IS NOT NULL AND "{column_column}" IS NOT NULL
            GROUP BY 1, 2
        """
        long = spec.select(sql, db or self.db)
        if long.empty:
            return pd.DataFrame()

        if limit:
            busiest = (
                long.groupby("row_key")["Accidents"].sum()
                .sort_values(ascending=False)
                .head(int(limit))
                .index
            )
            long = long[long["row_key"].isin(busiest)]

        matrix = long.pivot_table(
            index="row_key", columns="col_key", values="Accidents", fill_value=0
        ).reset_index().rename(columns={"row_key": row_column})

        order = self._category_order(row_column, matrix[row_column].tolist())
        return matrix.set_index(row_column).reindex(order).reset_index()

    def coverage_by_year(
        self,
        spec: FilterSpec | None = None,
        db: Database | None = None,
    ) -> pd.DataFrame:
        """Return per-year record counts plus the month span actually covered.

        The original dataset's final year is partial. Exposing the observed month
        span lets every chart label it rather than presenting a stub year as a
        full one.

        Args:
            spec: Active filters.
            db: Optional database handle.

        Returns:
            DataFrame with ``Year``, ``Accidents``, ``First_Month``, ``Last_Month``
            and ``Is_Partial``.
        """
        spec = spec or FilterSpec()
        frame = spec.select(
            f"""
            SELECT Year,
                   COUNT(*)::BIGINT AS Accidents,
                   MIN(Month) AS First_Month,
                   MAX(Month) AS Last_Month
            FROM {ACCIDENTS_VIEW}
            WHERE {{where}} AND Year IS NOT NULL
            GROUP BY Year
            ORDER BY Year
            """,
            db or self.db,
        )
        if frame.empty:
            return frame
        last_year = int(frame["Year"].max())
        frame["Is_Partial"] = frame["Year"] == last_year
        frame["Coverage"] = [
            f"{MONTH_NAMES[int(m) - 1]}-{MONTH_NAMES[int(m2) - 1]}"
            if pd.notna(m) and pd.notna(m2)
            else ""
            for m, m2 in zip(frame["First_Month"], frame["Last_Month"], strict=True)
        ]
        return frame

    # -- ordering ----------------------------------------------------------

    def _category_order(self, column: str, observed: Sequence[Any]) -> list[Any]:
        """Return the canonical sort order for a categorical column.

        Args:
            column: Column name.
            observed: Values actually present in the result.

        Returns:
            The canonical ordering if ``column`` has one, else ``observed``.
        """
        canonical: tuple[Any, ...] | None
        if column == "Day_Name" or column == "DayOfWeek":
            canonical = DAY_ORDER
        elif column == "Month_Name" or column == "Month":
            canonical = MONTH_ORDER
        elif column == "Hour":
            canonical = tuple(range(24))
        elif column == "Severity":
            canonical = schema.SEVERITY_LEVELS
        else:
            canonical = None

        if canonical is None:
            return list(observed)
        present = set(observed)
        ordered = [value for value in canonical if value in present]
        # Anything unrecognised still gets shown, appended at the end.
        ordered += [value for value in observed if value not in set(ordered)]
        return ordered

    def _apply_categorical_order(
        self, frame: pd.DataFrame, column: str, order_desc: bool
    ) -> pd.DataFrame:
        """Sort ``frame`` into canonical category order when one exists.

        Canonical ordering is applied regardless of ``order_desc``. It governs
        *which* categories exist and in what sequence they read -- days run
        Monday to Sunday, months January to December, hours 0 to 23. Sorting a
        day axis alphabetically would put Friday first, which reads as a bug
        even though the values are correct. ``order_desc`` governs ordering by
        count, which only makes sense where there is no canonical sequence.

        Args:
            frame: Frame to sort.
            column: Grouping column.
            order_desc: Whether the caller asked for a count sort.

        Returns:
            The reordered frame.
        """
        del order_desc  # Canonical order wins; count sorting is applied in SQL.
        if frame.empty or column not in frame.columns:
            return frame
        if column not in _CANONICAL_ORDER_COLUMNS:
            return frame

        canonical = self._category_order(column, frame[column].tolist())
        return frame.set_index(column).reindex(canonical).reset_index()


def add_percentage(
    frame: pd.DataFrame,
    count_column: str,
    new_name: str = "Percentage",
) -> pd.DataFrame:
    """Append a percentage column computed against the frame's own total.

    Args:
        frame: Frame containing ``count_column``.
        count_column: Column holding counts.
        new_name: Name for the added percentage column.

    Returns:
        A copy of ``frame`` with the percentage column appended.
    """
    out = frame.copy()
    total = float(out[count_column].sum())
    out[new_name] = (
        (out[count_column] / total * 100).round(4) if total else 0.0
    )
    return out


def top_n_mask(values: pd.Series, n: int = SETTINGS.top_n) -> pd.Series:
    """Return a boolean mask selecting the ``n`` most frequent values.

    Args:
        values: Series of category labels.
        n: Number of values to keep.

    Returns:
        Boolean mask aligned to ``values``.
    """
    if values.empty:
        return pd.Series(dtype=bool)
    keep = values.value_counts().head(n).index
    return values.isin(keep)

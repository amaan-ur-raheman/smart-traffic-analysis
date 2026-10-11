"""Shared Streamlit UI components and page scaffolding.

Everything that appears on more than one page lives here: the sidebar, the
global filters, the KPI row, the data-unavailable state, and the disclosure
widgets that keep methodological caveats attached to the figures they qualify.

Caching is deliberate and explicit. Only genuinely expensive, deterministic
work is cached, and every cached function's key includes the filter state, so a
changed filter cannot serve a stale result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import streamlit as st

from smart_traffic import schema
from smart_traffic.aggregation import FilterSpec
from smart_traffic.analytics import DatasetProfile
from smart_traffic.config import (
    ASSOCIATION_DISCLAIMER,
    DATASET_DISPLAY_NAME,
    DATASET_SOURCE_URL,
    EXPOSURE_DISCLAIMER,
    PATHS,
    SETTINGS,
    SEVERITY_MEANING,
)
from smart_traffic.db import DB, DatasetNotAvailableError
from smart_traffic.feature_engineering import rush_hour_definition_text
from smart_traffic.logging_config import setup_logging
from smart_traffic.reporting import safe_download_name

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

PAGE_PREFIX = "Smart Traffic Analysis"


def initialise() -> None:
    """Configure logging and page-level defaults. Safe to call repeatedly."""
    setup_logging(log_file=PATHS.logs / "app.log")
    st.set_page_config(
        page_title=PAGE_PREFIX,
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def dataset_error() -> str:
    """Return why the dataset is unusable, or an empty string when it is usable.

    Existence and file size are not sufficient: a build interrupted partway
    leaves a truncated Parquet that is present and non-empty but unreadable.
    Binding the DuckDB views parses the Parquet footer, so this genuinely
    verifies the file rather than trusting its size.
    """
    try:
        if not DB.is_available:
            return (
                "No processed dataset is loaded, so there is nothing to analyse yet."
            )
        DB.ensure_views()
    except DatasetNotAvailableError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - never let a bad file crash the app
        return (
            f"The processed dataset at {DB.parquet_path} could not be read: {exc}. "
            "Rebuild it with: uv run sta build"
        )
    return ""


def dataset_ready() -> bool:
    """Whether a processed dataset is available *and* readable by the app."""
    return dataset_error() == ""


def running_on_synthetic() -> bool:
    """Whether the loaded data came from the synthetic generator.

    Driven by the ``SYNTH-`` ID prefix rather than a stored flag, so it stays
    correct even if the Parquet was produced outside this app.
    """
    if not dataset_ready():
        return False
    try:
        sample = DB.sql(
            "SELECT ID FROM accidents LIMIT 50",  # noqa: S608 - static SQL
        )
    except Exception:  # noqa: BLE001
        return False
    if sample.empty:
        return False
    return sample["ID"].astype(str).str.startswith("SYNTH-").any()


def synthetic_warning() -> None:
    """Show a persistent banner when the loaded data is synthetic."""
    if running_on_synthetic():
        st.warning(
            "**Synthetic data in use.** These are generated records, not real "
            "accidents. Every figure on this page describes the generator's output "
            "and must not be reported as an analytical result. Run "
            "`uv run sta build` against the real Kaggle CSV to produce real findings.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
# Missing-data state
# ---------------------------------------------------------------------------


def show_setup_instructions() -> None:
    """Render the full 'no dataset' experience with copy-pasteable commands."""
    problem = dataset_error()
    # Distinguish "the file is there but unusable" from "there is no file yet":
    # the former needs a rebuild, the latter a download.
    corrupt = bool(DB.is_available)
    if problem:
        st.error(problem, icon="🚨")
        if corrupt:
            st.subheader("Rebuild the processed dataset")
            st.code("uv run sta build", language="bash")
            st.markdown(
                "If the rebuild fails, delete the file and build again:\n\n"
                f"```bash\nrm {PATHS.accidents_parquet}\nuv run sta build\n```"
            )
            return

    st.error("No processed dataset is loaded, so there is nothing to analyse yet.", icon="🚨")

    st.subheader("Get the dataset")
    st.markdown(
        f"""
The analysis runs on **{DATASET_DISPLAY_NAME}**, a public Kaggle dataset of roughly
7.7 million recorded accidents. It is about 3 GB, so it is not bundled with this
project and cannot be uploaded through the browser.

1. Sign in to Kaggle and open the dataset page.
2. Download `US_Accidents_March23.csv`.
3. Place it at:

```
{PATHS.raw_csv}
```

A free Kaggle account is required. There is no paid API and no external service
is contacted at any point.
"""
    )
    st.link_button("Open the Kaggle dataset", DATASET_SOURCE_URL, type="primary")

    st.subheader("Then build the processed dataset")
    st.code("uv sync\nuv run sta build", language="bash")

    st.subheader("Or explore immediately with synthetic data")
    st.markdown(
        "If you just want to see the application working, generate a "
        "schema-faithful synthetic file. It exercises every part of the pipeline "
        "and is clearly labelled so its figures are never mistaken for results."
    )
    st.code(
        "uv run sta demo\nuv run sta build --source data/synthetic/synthetic_accidents.csv\n"
        "uv run streamlit run app.py",
        language="bash",
    )


def require_dataset() -> bool:
    """Ensure a dataset is present, or render setup instructions.

    Returns:
        ``True`` when the page may proceed.
    """
    initialise()
    if not dataset_ready():
        show_setup_instructions()
        return False
    synthetic_warning()
    return True


# ---------------------------------------------------------------------------
# Sidebar and filters
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_filter_options() -> dict[str, list[Any]]:
    """Return the distinct values available for each filter widget.

    Cached because these are full-table scans that do not change unless the
    dataset is rebuilt. ``cache_data`` also handles the DataFrame->list
    conversion that ``st.session_state`` cannot.
    """
    DB.ensure_views()
    years = DB.sql("SELECT DISTINCT Year FROM accidents WHERE Year IS NOT NULL ORDER BY Year")
    states = DB.sql("SELECT DISTINCT State FROM accidents WHERE State IS NOT NULL ORDER BY State")
    return {
        "years": [int(y) for y in years["Year"].tolist()],
        "states": [str(s) for s in states["State"].tolist()],
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_top_cities(limit: int = 400) -> list[str]:
    """Return the most frequent city names, for the optional city filter.

    Only the top ``limit`` are offered: the dataset contains well over a hundred
    thousand distinct city strings, and a multi-select over all of them is not
    usable in a browser.
    """
    frame = DB.sql(
        f"SELECT City, COUNT(*) AS n FROM accidents WHERE City IS NOT NULL "
        f"GROUP BY City ORDER BY n DESC LIMIT {int(limit)}"
    )
    return [str(c) for c in frame["City"].tolist()]


def sidebar_header() -> None:
    """Render the app identity block at the top of the sidebar."""
    st.sidebar.markdown(
        """
<div style="padding:2px 0 6px 0">
  <div style="font-size:1.35rem;font-weight:700;letter-spacing:-0.02em;color:#102a43">
    🚦 Smart Traffic Analysis
  </div>
  <div style="font-size:0.82rem;color:#627d98;margin-top:2px">
    Accident Severity Prediction
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()


def global_filters(*, show_cities: bool = True) -> FilterSpec:
    """Render the global filter sidebar and return the active filter set.

    Filters persist across pages via ``st.session_state`` so that navigating
    from the overview to the temporal page does not silently reset the user's
    selection.

    Args:
        show_cities: Whether to render the city multi-select.

    Returns:
        The :class:`FilterSpec` the page should query with.
    """
    if not dataset_ready():
        return FilterSpec()

    try:
        options = _cached_filter_options()
    except DatasetNotAvailableError:
        return FilterSpec()

    state = st.session_state
    state.setdefault("filters", {})

    with st.sidebar.expander("Filters", expanded=True):
        st.caption("Filters apply to every analytical page.")

        selected_years = st.multiselect(
            "Year", options["years"], key="flt_years",
            help="Leave empty for all years.",
        )
        selected_states = st.multiselect(
            "State", options["states"], key="flt_states",
            help="Two-letter state codes.",
        )
        selected_severities = st.multiselect(
            "Severity", list(schema.SEVERITY_LEVELS), key="flt_severity",
            help="Reported traffic impact categories.",
        )
        selected_months = st.multiselect(
            "Month", list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"),
            key="flt_months",
        )
        selected_categories = st.multiselect(
            "Day / time category",
            ["Weekday", "Weekend", "Rush Hour", "Non-Rush Hour", "Day", "Night"],
            key="flt_categories",
        )

        selected_cities: list[str] = []
        if show_cities:
            with st.expander("City (top 400 only)"):
                cities = _cached_top_cities()
                selected_cities = st.multiselect(
                    "City", cities, key="flt_cities",
                    help="The full dataset has over 100,000 distinct city values, so only "
                         "the 400 most frequent are listed.",
                )

    spec = FilterSpec(
        years=selected_years or None,
        states=selected_states or None,
        cities=selected_cities or None,
        severities=selected_severities or None,
        months=selected_months or None,
        day_types=selected_categories or None,
        valid_only=True,
    )
    state["filters"] = spec
    render_filter_summary(spec)
    return spec


def render_filter_summary(spec: FilterSpec) -> None:
    """Show which filters are active, with a one-click reset."""
    labels = spec.describe()
    st.sidebar.markdown("---")
    if not labels:
        st.sidebar.caption("No filters applied — showing the full dataset.")
        return
    st.sidebar.markdown("**Active filters**")
    for label in labels:
        st.sidebar.caption(f"• {label}")
    if st.sidebar.button("Reset filters", width="stretch"):
        for key in ("flt_years", "flt_states", "flt_cities", "flt_severity",
                    "flt_months", "flt_categories"):
            st.session_state.pop(key, None)
        st.rerun()


def sidebar_footer() -> None:
    """Render dataset provenance and controls at the bottom of the sidebar."""
    st.sidebar.markdown("---")
    if not dataset_ready():
        st.sidebar.caption("No dataset loaded.")
        return

    try:
        total = int(DB.scalar("SELECT COUNT(*) FROM accidents") or 0)
        unique = int(DB.scalar("SELECT COUNT(DISTINCT ID) FROM accidents") or 0)
    except Exception:  # noqa: BLE001
        st.sidebar.caption("Dataset present but not readable.")
        return

    st.sidebar.caption(
        f"**{total:,}** records · **{unique:,}** unique IDs"
        + (" · synthetic" if running_on_synthetic() else "")
    )
    st.sidebar.caption(
        f"Rush hour: {SETTINGS.rush_hour_morning[0]:02d}:00–"
        f"{SETTINGS.rush_hour_morning[1]:02d}:59 and "
        f"{SETTINGS.rush_hour_evening[0]:02d}:00–"
        f"{SETTINGS.rush_hour_evening[1]:02d}:59 · seed {SETTINGS.random_seed}"
    )
    st.sidebar.link_button("Dataset source (Kaggle)", DATASET_SOURCE_URL)


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a consistent page title block.

    Args:
        title: Page title.
        subtitle: One-line description of what the page covers.
        icon: Optional emoji prefix.
    """
    st.markdown(
        f"""
<div style="padding:4px 0 10px 0;border-bottom:1px solid #e4e7eb;margin-bottom:14px">
  <div style="font-size:1.7rem;font-weight:700;letter-spacing:-0.02em;color:#102a43">
    {icon} {title}
  </div>
  {f'<div style="font-size:0.95rem;color:#486581;margin-top:4px">{subtitle}</div>' if subtitle else
      ''}
</div>
""",
        unsafe_allow_html=True,
    )


def section(title: str, description: str = "") -> None:
    """Render a section heading.

    Args:
        title: Section title.
        description: Optional supporting sentence.
    """
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def kpi_row(items: dict[str, Any], *, columns: int | None = None) -> None:
    """Render a row of KPI cards.

    Args:
        items: Ordered mapping of label to display value.
        columns: Number of columns. Defaults to a responsive 4.
    """
    count = columns or min(len(items), 4)
    labels = list(items.keys())
    for start in range(0, len(labels), count):
        chunk = labels[start : start + count]
        row = st.columns(len(chunk))
        for column, label in zip(row, chunk, strict=False):
            column.metric(label, items[label])


def severity_kpi_row(spec: FilterSpec, *, include_total: bool = True) -> None:
    """Render the severity KPI cards for a filter state.

    Args:
        spec: Active filters.
        include_total: Whether to show the total-records card.
    """
    from smart_traffic.analytics import kpi_cards

    cards = kpi_cards(spec)
    items = {k: v for k, v in cards.items() if k != "_raw"}
    if not include_total:
        items.pop("Total accidents", None)
    kpi_row(items, columns=4)


def info_box(kind: str, title: str, body: str) -> None:
    """Render a styled callout.

    Args:
        kind: One of ``info``, ``warning``, ``success``, ``error``.
        title: Bold heading.
        body: Supporting text.
    """
    palette = {
        "info": ("#e6f0fb", "#1c5fa8", "ℹ️"),
        "warning": ("#fdf3e3", "#8a5a06", "⚠️"),
        "success": ("#e8f5ee", "#1e6b40", "✅"),
        "error": ("#fdecec", "#96201f", "⛔"),
    }
    background, colour, icon = palette.get(kind, palette["info"])
    st.markdown(
        f"""
<div style="background:{background};border-left:4px solid {colour};
     border-radius:6px;padding:12px 16px;margin:10px 0">
  <div style="font-weight:700;color:{colour};margin-bottom:4px">{icon} {title}</div>
  <div style="font-size:0.9rem;color:#243b53">{body}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def methodology_notes(*, include_severity: bool = True, include_exposure: bool = True) -> None:
    """Render the standing caveats inside a collapsible panel.

    Args:
        include_severity: Include the severity-meaning caveat.
        include_exposure: Include the exposure/rate caveat.
    """
    with st.expander("How to read these numbers", expanded=False):
        st.markdown(f"- {SEVERITY_MEANING}" if include_severity else "")
        st.markdown(f"- {EXPOSURE_DISCLAIMER}" if include_exposure else "")
        st.markdown(f"- {ASSOCIATION_DISCLAIMER}")
        st.markdown(f"- {rush_hour_definition_text()}")


def severity_regime_warning() -> None:
    """Warn that severity prevalence shifts across years, with the evidence.

    The mix moves far more between years than any weather or road factor does:
    severity 3 falls from about 33% of 2016-2018 to none at all in 2023, while
    severity 1 appears in bulk only in 2020 and 2022. A model carrying ``Year``
    can therefore score well by recognising the reporting convention rather
    than the incident, and a random split hides that because it mixes regimes.

    Any page reporting model performance needs this caveat, so it lives here
    rather than being retyped per page.
    """
    from smart_traffic.analytics import severity_prevalence_by_year

    try:
        frame = severity_prevalence_by_year()
    except Exception:  # noqa: BLE001 - the caveat must never break a page
        return
    if frame.empty or len(frame) < 2:
        return

    early = frame["Severity_3_%"].iloc[0]
    late = frame["Severity_3_%"].iloc[-1]
    vanished = early > 5.0 and late == 0.0

    with st.expander(
        "Severity is not recorded consistently across years — this limits the models",
        expanded=False,
    ):
        st.warning(
            "The severity field reflects a reporting convention that changed over "
            "time, not a stable physical property of the accidents. Severity 3 "
            f"falls from {early:.1f}% of records in {int(frame['Year'].iloc[0])} to "
            f"{late:.1f}% in {int(frame['Year'].iloc[-1])}"
            + (
                " — it does not appear at all in the final year."
                if vanished
                else "."
            ),
            icon="⚠️",
        )
        st.markdown(
            "Two consequences worth stating plainly:\n\n"
            "- **A random split flatters the models.** Because it mixes reporting "
            "regimes, a model carrying `Year` can partly recognise which convention "
            "a record was logged under. Reported scores should be read that way.\n"
            "- **A temporal holdout is much harsher.** Training on 2016-2021 and "
            "testing on 2022-2023 drops macro F1 from roughly 0.58 to roughly "
            "0.40 for the gradient-boosting model. Reproduce it with:\n\n"
            "```bash\nuv run sta train --temporal \\\n"
            "  --train-years 2016-2021 --test-years 2022-2023\n```\n\n"
            "Neither figure is a forecast of future accidents. They describe how "
            "severity was recorded, which is a property of the data rather than "
            "of the crash."
        )
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption("Share of each year's records by severity, computed from the loaded dataset.")


def download_table(
    frame: pd.DataFrame,
    label: str,
    *,
    key: str,
    filename_hint: str = "export",
) -> None:
    """Render a download button for a DataFrame.

    Args:
        frame: Data to export.
        label: Button label.
        key: Unique widget key.
        filename_hint: Prefix for the generated filename.
    """
    if frame is None or frame.empty:
        return
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_download_name(filename_hint)}.csv",
        mime="text/csv",
        key=key,
        width="content",
    )


def sample_disclosure(note: str) -> None:
    """Render a disclosure that a figure came from a sample, not the full data."""
    if note:
        st.caption(f"📐 {note}")


def coverage_warning(profile: DatasetProfile) -> None:
    """Warn when the visible data includes a partial year."""
    if profile.partial_year and len(profile.years_present) > 1:
        st.caption(
            f"ℹ️ **{profile.partial_year} is incomplete** in this dataset and is not "
            f"comparable to the full years. Per-year charts mark it explicitly."
        )


def empty_state(message: str, *, detail: str = "") -> None:
    """Render an informative empty state.

    Args:
        message: What is missing.
        detail: Optional suggested action.
    """
    st.info(f"**{message}**" + (f" {detail}" if detail else ""), icon="🗒️")


def guarded_section(title: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run a computation and convert failure into a friendly message.

    Technical detail goes to the log; the user sees an actionable sentence.

    Args:
        title: Section heading.
        func: Zero-argument callable performing the work.
        *args: Positional arguments for ``func``.
        **kwargs: Keyword arguments for ``func``.
    """
    from smart_traffic.logging_config import get_logger

    logger = get_logger("ui")
    section(title)
    try:
        func(*args, **kwargs)
    except DatasetNotAvailableError:
        st.error("No dataset is loaded. Return to the overview page for setup instructions.")
        logger.warning("%s blocked: no dataset", title)
    except MemoryError:
        st.error(
            "This analysis ran out of memory. Reduce the sample size in the sidebar "
            "or the row limit for this operation."
        )
        logger.exception("%s blocked: out of memory", title)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"This section could not be completed: {exc}")
        logger.exception("%s failed", title)


def charts_row(figures: Sequence[Any], columns: int = 2) -> None:
    """Render a row of Plotly figures.

    Args:
        figures: Figures to render. ``None`` entries are skipped.
        columns: Number of columns.
    """
    present = [f for f in figures if f is not None]
    if not present:
        return
    for start in range(0, len(present), columns):
        chunk = present[start : start + columns]
        for column, figure in zip(st.columns(len(chunk)), chunk, strict=False):
            column.plotly_chart(figure, width="stretch",
                                config={"displaylogo": False})

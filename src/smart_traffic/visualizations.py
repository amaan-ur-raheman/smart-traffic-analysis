"""Plotly figure builders.

Every function returns a configured ``Figure`` or ``None``. Centralising the
theme means the dashboards cannot drift apart visually, and it means the colour
choices are made once against a contrast requirement rather than per chart.

Figures are deliberately built to be read: units and denominators are in the
title or axis label, category axes are ordered canonically, and no chart
implies a causal relationship.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from smart_traffic.config import ASSOCIATION_DISCLAIMER
from smart_traffic.logging_config import get_logger

logger = get_logger("visualizations")

#: Colour scale for sequential data (record counts, density).
SEQUENTIAL_SCALE: Final = [
    [0.0, "#f7fbff"], [0.15, "#c6dbef"], [0.35, "#6baed6"],
    [0.55, "#2171b5"], [0.75, "#08306b"], [1.0, "#081d58"],
]

#: Fixed colours per severity category, used everywhere severity is shown.
SEVERITY_COLORS: Final[dict[int, str]] = {
    1: "#2c7bb6",
    2: "#7fcdbb",
    3: "#fdae61",
    4: "#d7191c",
}

#: Categorical palette for series that do not encode severity.
CATEGORICAL_COLORS: Final[tuple[str, ...]] = (
    "#2c7bb6", "#d7191c", "#1a9641", "#fdae61",
    "#7b3294", "#c994c7", "#008837", "#e08214",
)

FONT_FAMILY: Final = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

_LAYOUT_DEFAULTS: Final[dict[str, Any]] = {
    "font": {"family": FONT_FAMILY, "size": 13, "color": "#1f2933"},
    "title_font": {"size": 16, "color": "#102a43"},
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "margin": {"l": 56, "r": 24, "t": 64, "b": 56},
    "hoverlabel": {"bgcolor": "white", "font_size": 13, "font_family": FONT_FAMILY},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
}


def apply_theme(
    figure: go.Figure,
    *,
    title: str | None = None,
    height: int = 420,
    showlegend: bool | None = None,
) -> go.Figure:
    """Apply the shared theme to a figure.

    Args:
        figure: The figure to style.
        title: Optional title.
        height: Figure height in pixels.
        showlegend: Override legend visibility.

    Returns:
        The styled figure.
    """
    figure.update_layout(**_LAYOUT_DEFAULTS)
    figure.update_layout(height=height)
    if title:
        figure.update_layout(title={"text": title, "x": 0.01, "xanchor": "left"})
    if showlegend is not None:
        figure.update_layout(showlegend=showlegend)
    # `linecolor="none"` is rejected by Plotly 7; suppressing the line is done
    # with showline=False, which is also the clearer expression of intent.
    figure.update_xaxes(showgrid=False, linecolor="#cbd2d9", zeroline=False)
    figure.update_yaxes(gridcolor="#eef2f5", showline=False)
    return figure


def empty_state(title: str, message: str) -> go.Figure:
    """Return a placeholder figure for an empty result set.

    Args:
        title: Short title.
        message: Explanation shown in the body.

    Returns:
        A figure displaying the message instead of empty axes.
    """
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, align="center",
        font={"size": 14, "color": "#627d98"},
    )
    return apply_theme(figure, title=title, height=280, showlegend=False)


def _format_counts(values: pd.Series) -> list[str]:
    """Format a count series with thousands separators for hover text."""
    return [f"{int(v):,}" for v in values]


def bar_chart(
    frame: pd.DataFrame,
    x: str,
    y: str = "Accidents",
    *,
    title: str | None = None,
    color: str = "#2c7bb6",
    height: int = 420,
    x_title: str | None = None,
    y_title: str = "Recorded accidents",
    horizontal: bool = False,
    text_auto: bool = True,
) -> go.Figure:
    """Build a bar chart.

    Args:
        frame: Source data.
        x: Category column.
        y: Value column.
        title: Chart title.
        color: Bar colour, or a column name to colour by.
        height: Figure height.
        x_title: Axis title override.
        y_title: Axis title.
        horizontal: Render bars horizontally.
        text_auto: Show value labels on bars.

    Returns:
        The figure, or an empty-state figure when ``frame`` is empty.
    """
    if frame.empty:
        return empty_state(title or "No data", "No records match the current filters.")

    colours = color if isinstance(color, str) and color in frame.columns else color
    labels = _format_counts(frame[y]) if text_auto else None

    figure = go.Figure(
        go.Bar(
            x=frame[x], y=frame[y], orientation="h" if horizontal else "v",
            marker_color=colours,
            text=labels,
            textposition="outside" if text_auto else "none",
            cliponaxis=False,
            hovertemplate=f"<b>%{{{'x' if not horizontal else 'y'}}}</b><br>"
                          f"{y_title}: %{{{'y' if not horizontal else 'x'}:,.0f}}<extra></extra>",
        )
    )
    figure.update_layout(showlegend=False)
    if horizontal:
        figure.update_yaxes(title_text=x_title or x, categoryorder="array")
        figure.update_xaxes(title_text=y_title)
    else:
        figure.update_xaxes(title_text=x_title or x, type="category")
        figure.update_yaxes(title_text=y_title, rangemode="tozero")
    return apply_theme(figure, title=title, height=height)


def grouped_bar(
    frame: pd.DataFrame,
    x: str,
    series: dict[str, str],
    *,
    title: str | None = None,
    height: int = 440,
    x_title: str | None = None,
    y_title: str = "Recorded accidents",
    stacked: bool = False,
) -> go.Figure:
    """Build a multi-series bar chart.

    Args:
        frame: Source data.
        x: Category column.
        series: Mapping of series name to its column.
        title: Chart title.
        height: Figure height.
        x_title: X axis title.
        y_title: Y axis title.
        stacked: Stack the series.

    Returns:
        The figure.
    """
    if frame.empty:
        return empty_state(title or "No data", "No records match the current filters.")

    figure = go.Figure()
    for index, (name, column) in enumerate(series.items()):
        if column not in frame.columns:
            continue
        figure.add_trace(
            go.Bar(
                name=name, x=frame[x], y=frame[column],
                marker_color=CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)],
                hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:,.0f}}<extra></extra>",
            )
        )
    figure.update_layout(barmode="stack" if stacked else "group")
    figure.update_xaxes(title_text=x_title or x, type="category")
    figure.update_yaxes(title_text=y_title, rangemode="tozero")
    return apply_theme(figure, title=title, height=height)


def severity_distribution_chart(
    frame: pd.DataFrame,
    *,
    title: str = "Severity distribution",
    value_column: str = "Accidents",
    height: int = 400,
) -> go.Figure:
    """Build the severity distribution bar chart with fixed colours.

    Args:
        frame: Frame with ``Severity`` and a value column.
        title: Chart title.
        value_column: Column holding the values.
        height: Figure height.

    Returns:
        The figure.
    """
    if frame.empty:
        return empty_state(title, "No severity data available.")

    frame = frame.copy()
    figure = go.Figure(
        go.Bar(
            x=frame["Severity"],
            y=frame[value_column],
            marker_color=[SEVERITY_COLORS.get(int(s), "#8da0cb") for s in frame["Severity"]],
            text=_format_counts(frame[value_column]),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>Severity %{x}</b><br>Accidents: %{y:,.0f}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Reported severity category", type="category")
    figure.update_yaxes(title_text="Recorded accidents", rangemode="tozero")
    return apply_theme(figure, title=title, height=height, showlegend=False)


def severity_rate_chart(
    frame: pd.DataFrame,
    group_column: str,
    *,
    title: str = "Severity rate within each group",
    levels: Sequence[int] = (1, 2, 3, 4),
    height: int = 460,
    limit: int | None = 20,
) -> go.Figure:
    """Build a 100% stacked chart of within-group severity rates.

    Args:
        frame: Frame from :meth:`Aggregator.severity_rates`.
        group_column: Column defining the groups.
        title: Chart title.
        levels: Severity levels to include.
        height: Figure height.
        limit: Keep only the busiest ``limit`` groups.

    Returns:
        The figure.
    """
    if frame.empty:
        return empty_state(title, "No records match the current filters.")

    data = frame.head(limit) if limit else frame
    figure = go.Figure()
    for level in levels:
        column = f"Severity_{level}_Rate"
        if column not in data.columns:
            continue
        figure.add_trace(
            go.Bar(
                name=f"Severity {level}",
                x=data[group_column],
                y=data[column],
                marker_color=SEVERITY_COLORS.get(level, "#8da0cb"),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>Severity {level} rate: %{{y:.2f}}% of this group"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(barmode="stack")
    figure.update_yaxes(
        title_text="Share of accidents within the group (%)",
        ticksuffix="%",
        range=[0, 100],
    )
    figure.update_xaxes(title_text=group_column.replace("_", " ").title(), type="category")
    return apply_theme(
        figure,
        title=(
            f"{title}<br><sup>Denominator: all accidents within each group. "
            f"{ASSOCIATION_DISCLAIMER}</sup>"
        ),
        height=height,
    )


def heatmap(
    matrix: pd.DataFrame,
    row_column: str,
    *,
    title: str = "Heatmap",
    color_label: str = "Recorded accidents",
    height: int = 480,
) -> go.Figure:
    """Build a heatmap from a wide matrix.

    Args:
        matrix: Wide DataFrame whose first column holds the row labels.
        row_column: Name of that label column.
        title: Chart title.
        color_label: Colour bar label.
        height: Figure height.

    Returns:
        The figure.
    """
    if matrix.empty or matrix.shape[1] < 2:
        return empty_state(title, "Not enough data to build a heatmap for the current filters.")

    labels = matrix[row_column].astype(str).tolist()
    columns = [c for c in matrix.columns if c != row_column]
    values = matrix[columns].to_numpy(dtype=float)

    figure = go.Figure(
        go.Heatmap(
            z=values, x=[str(c) for c in columns], y=labels,
            colorscale=SEQUENTIAL_SCALE,
            colorbar={"title": {"text": color_label, "side": "right"}},
            hovertemplate="%{y}<br>%{x}: %{z:,.0f}<extra></extra>",
        )
    )
    figure.update_xaxes(showgrid=False, type="category")
    figure.update_yaxes(showgrid=False, autorange="reversed", type="category")
    return apply_theme(figure, title=title, height=height, showlegend=False)


def scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    title: str | None = None,
    height: int = 480,
    x_title: str | None = None,
    y_title: str | None = None,
    opacity: float = 0.55,
    max_points: int = 30_000,
) -> go.Figure:
    """Build a scatter plot, downsampled if needed.

    Args:
        frame: Source data.
        x: X column.
        y: Y column.
        color: Optional column for the marker colour.
        title: Chart title.
        height: Figure height.
        x_title: X axis title.
        y_title: Y axis title.
        opacity: Marker opacity.
        max_points: Maximum markers rendered.

    Returns:
        The figure.
    """
    if frame.empty:
        return empty_state(title or "No data", "No records match the current filters.")

    data = frame
    if len(frame) > max_points:
        data = frame.sample(n=max_points, random_state=42)

    marker: dict[str, Any] = {"size": 6, "opacity": opacity, "color": "#2c7bb6"}
    if color and color in data.columns:
        marker["color"] = data[color]
        marker["colorscale"] = SEQUENTIAL_SCALE
        marker["colorbar"] = {"title": {"text": color, "side": "right"}}
        marker["showscale"] = True

    figure = go.Figure(
        go.Scattergl(
            x=data[x], y=data[y], mode="markers", marker=marker,
            hovertemplate=f"<b>%{{x}}</b><br>{y}: %{{y:,.2f}}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text=x_title or x)
    figure.update_yaxes(title_text=y_title or y)
    return apply_theme(figure, title=title, height=height, showlegend=False)


def line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    height: int = 400,
    x_title: str | None = None,
    y_title: str | None = None,
    markers: bool = True,
) -> go.Figure:
    """Build a line chart.

    Args:
        frame: Source data.
        x: X column.
        y: Y column.
        title: Chart title.
        height: Figure height.
        x_title: X axis title.
        y_title: Y axis title.
        markers: Show point markers.

    Returns:
        The figure.
    """
    if frame.empty:
        return empty_state(title or "No data", "No records match the current filters.")

    figure = go.Figure(
        go.Scatter(
            x=frame[x], y=frame[y],
            mode="lines+markers" if markers else "lines",
            line={"color": "#2c7bb6", "width": 2.5},
            marker={"size": 7},
            text=_format_counts(frame[y]),
            hovertemplate=f"<b>%{{x}}</b><br>{y_title or y}: %{{y:,.0f}}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text=x_title or x, type="category")
    figure.update_yaxes(title_text=y_title or y, rangemode="tozero")
    return apply_theme(figure, title=title, height=height, showlegend=False)


def pie_chart(
    frame: pd.DataFrame,
    labels: str,
    values: str,
    *,
    title: str | None = None,
    height: int = 400,
    donut: bool = True,
) -> go.Figure:
    """Build a donut or pie chart.

    Args:
        frame: Source data.
        labels: Label column.
        values: Value column.
        title: Chart title.
        height: Figure height.
        donut: Render as a donut.

    Returns:
        The figure.
    """
    if frame.empty:
        return empty_state(title or "No data", "No records match the current filters.")

    figure = go.Figure(
        go.Pie(
            labels=frame[labels].astype(str),
            values=frame[values],
            hole=0.45 if donut else 0.0,
            marker={"colors": list(SEVERITY_COLORS.values())[: len(frame)]},
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} (%{percent})<extra></extra>",
        )
    )
    return apply_theme(figure, title=title, height=height)


def density_map(
    cells: pd.DataFrame,
    *,
    title: str = "Recorded accident density",
    height: int = 560,
) -> go.Figure:
    """Build a scatter-density map from aggregated grid cells.

    Args:
        cells: Output of :func:`smart_traffic.geospatial.density_grid`.
        title: Chart title.
        height: Figure height.

    Returns:
        The figure.
    """
    if cells.empty:
        return empty_state(title, "No valid coordinates available for the current filters.")

    maximum = float(cells["Accidents"].max())
    figure = go.Figure(
        go.Scattergeo(
            lat=cells["Latitude"],
            lon=cells["Longitude"],
            mode="markers",
            marker={
                "size": cells["Accidents"],
                "sizemode": "area",
                "sizeref": 2.0 * maximum / (34.0**2),
                "sizemin": 2,
                "color": cells["Accidents"],
                "colorscale": SEQUENTIAL_SCALE,
                "showscale": True,
                "colorbar": {"title": {"text": "Accidents per cell", "side": "right"}},
                "opacity": 0.75,
                "line": {"width": 0},
            },
            hovertemplate=(
                "<b>Grid cell</b><br>"
                "Lat %{lat:.2f}, Lon %{lon:.2f}<br>"
                "Recorded accidents: %{marker.color:,.0f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        geo={
            "scope": "usa",
            "projection_type": "albers usa",
            "showlakes": True,
            "lakecolor": "#e8eef2",
            "landcolor": "#f5f7f8",
        },
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
    )
    return apply_theme(figure, title=title, height=height, showlegend=False)


def correlation_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str = "Correlation between numeric variables",
    height: int = 560,
    note: str = "",
) -> go.Figure:
    """Build a diverging correlation heatmap.

    Args:
        matrix: Square correlation DataFrame.
        title: Chart title.
        height: Figure height.
        note: Optional subtitle carrying provenance.

    Returns:
        The figure.
    """
    if matrix.empty:
        return empty_state(title, "Not enough numeric variables are available.")

    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=list(matrix.columns),
            y=list(matrix.index),
            zmin=-1, zmax=1, zmid=0,
            colorscale=[
                [0.0, "#2166ac"], [0.25, "#67a9cf"],
                [0.5, "#f7f7f7"], [0.75, "#ef8a62"], [1.0, "#b2182b"],
            ],
            colorbar={"title": {"text": "Pearson r", "side": "right"}},
            hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    figure.update_xaxes(showgrid=False, tickangle=-45)
    figure.update_yaxes(showgrid=False, autorange="reversed")
    subtitle = f"<br><sup>{note}</sup>" if note else ""
    return apply_theme(
        figure, title=f"{title}{subtitle}", height=height, showlegend=False
    )


def explained_variance_chart(
    variance: pd.DataFrame,
    *,
    title: str = "Variance explained by each principal component",
    height: int = 380,
) -> go.Figure:
    """Build a bar chart of explained variance with a cumulative line.

    Args:
        variance: Frame with ``Component``, ``Explained_Variance_Ratio`` and
            ``Cumulative_Variance``.
        title: Chart title.
        height: Figure height.

    Returns:
        The figure.
    """
    if variance.empty:
        return empty_state(title, "PCA has not been run yet.")

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=variance["Component"],
            y=variance["Explained_Variance_Ratio"],
            name="Per component",
            marker_color="#2c7bb6",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=variance["Component"],
            y=variance["Cumulative_Variance"],
            name="Cumulative",
            mode="lines+markers",
            line={"color": "#d7191c", "width": 2.5},
            hovertemplate="%{x}: %{y:.2f}% cumulative<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title_text="Variance explained (%)", secondary_y=False)
    figure.update_yaxes(
        title_text="Cumulative variance (%)",
        range=[0, 105], secondary_y=True, showgrid=False,
    )
    figure.update_xaxes(title_text="Principal component", type="category")
    return apply_theme(figure, title=title, height=height)


def silhouette_chart(
    table: pd.DataFrame,
    *,
    title: str = "Silhouette score by K",
    height: int = 380,
) -> go.Figure:
    """Build the K-selection curve.

    Args:
        table: Output of :func:`smart_traffic.clustering.silhouette_search`.
        title: Chart title.
        height: Figure height.

    Returns:
        The figure.
    """
    if table.empty:
        return empty_state(title, "The K search has not been run yet.")

    colours = [
        "#d7191c" if bool(row.get("Is_Recommended")) else "#2c7bb6"
        for _, row in table.iterrows()
    ]
    figure = go.Figure(
        go.Bar(
            x=table["K"], y=table["Silhouette_Score"],
            marker_color=colours,
            hovertemplate=(
                "K=%{x}<br>Silhouette: %{y:.4f}<br>"
                "Inertia: %{customdata[0]:,.0f}<extra></extra>"
            ),
            customdata=table[["Inertia"]].to_numpy(),
        )
    )
    figure.update_xaxes(title_text="Number of clusters (K)", type="category")
    figure.update_yaxes(title_text="Silhouette score (higher is better)")
    return apply_theme(
        figure, title=f"{title}<br><sup>Red marks the best-scoring K.</sup>",
        height=height, showlegend=False,
    )


def confusion_matrix_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str = "Confusion matrix",
    height: int = 460,
    note: str = "",
) -> go.Figure:
    """Build a confusion matrix heatmap with counts and row-normalised values.

    Args:
        matrix: Frame whose index is true labels and columns are predictions.
        title: Chart title.
        height: Figure height.
        note: Optional subtitle.

    Returns:
        The figure.
    """
    if matrix.empty:
        return empty_state(title, "No evaluation results available yet.")

    values = matrix.to_numpy(dtype=float)
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=[c.replace("Predicted ", "") for c in matrix.columns],
            y=[i.replace("Actual ", "") for i in matrix.index],
            colorscale=SEQUENTIAL_SCALE,
            text=[[f"{int(v):,}" for v in row] for row in values],
            texttemplate="%{text}",
            hovertemplate="Actual %{y} / Predicted %{x}<br>%{z:,.0f} records<extra></extra>",
            colorbar={"title": {"text": "Records", "side": "right"}},
        )
    )
    figure.update_xaxes(title_text="Predicted severity", type="category")
    figure.update_yaxes(title_text="Actual severity", type="category", autorange="reversed")
    subtitle = f"<br><sup>{note}</sup>" if note else ""
    return apply_theme(figure, title=f"{title}{subtitle}", height=height, showlegend=False)


def probability_chart(
    probabilities: dict[str, float],
    predicted: int,
    *,
    title: str = "Predicted class probabilities",
    height: int = 300,
) -> go.Figure:
    """Build a bar chart of predicted class probabilities.

    Args:
        probabilities: Mapping of class label to probability.
        predicted: The selected class.
        title: Chart title.
        height: Figure height.

    Returns:
        The figure.
    """
    if not probabilities:
        return empty_state(title, "This model does not expose class probabilities.")

    frame = pd.DataFrame(
        {"Class": list(probabilities.keys()), "Probability": list(probabilities.values())}
    )
    figure = go.Figure(
        go.Bar(
            x=frame["Class"], y=frame["Probability"],
            marker_color=[
                SEVERITY_COLORS.get(predicted, "#d7191c")
                if "Severity" in c and str(predicted) in c
                else "#8da0cb"
                for c in frame["Class"]
            ],
            text=[f"{v:.1%}" for v in frame["Probability"]],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Severity class", type="category")
    figure.update_yaxes(title_text="Predicted probability", ticksuffix="%", range=[0, 1.05])
    return apply_theme(figure, title=title, height=height, showlegend=False)


def workflow_diagram(stages: Sequence[str], *, height: int = 150) -> go.Figure:
    """Build a horizontal process-flow diagram.

    Args:
        stages: Stage labels, in order.
        height: Figure height.

    Returns:
        The figure.
    """
    figure = go.Figure()
    positions = list(range(len(stages)))

    figure.add_trace(
        go.Scatter(
            x=positions,
            y=[0] * len(positions),
            mode="markers",
            marker={"size": 44, "color": "#2c7bb6", "symbol": "circle"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=positions,
            y=[0] * len(positions),
            mode="text",
            text=[f"<b>{i + 1}</b>" for i in positions],
            textfont={"size": 15, "color": "white"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=positions,
            y=[-0.42] * len(positions),
            mode="text",
            text=list(stages),
            textfont={"size": 12, "color": "#1f2933"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if len(positions) > 1:
        figure.add_trace(
            go.Scatter(
                x=positions,
                y=[0] * len(positions),
                mode="lines",
                line={"color": "#cbd2d9", "width": 3, "dash": "dot"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.update_xaxes(
        range=[-0.7, len(stages) - 0.3], showgrid=False, showticklabels=False,
        zeroline=False, showline=False,
    )
    figure.update_yaxes(
        range=[-0.75, 0.5], showgrid=False, showticklabels=False,
        zeroline=False, showline=False,
    )
    figure.update_layout(height=height, margin={"l": 20, "r": 20, "t": 10, "b": 10})
    return figure


def bar_from_series(
    series: pd.Series,
    *,
    title: str | None = None,
    color: str = "#2c7bb6",
    height: int = 380,
    x_title: str | None = None,
    y_title: str = "Recorded accidents",
    order: Sequence[str] | None = None,
) -> go.Figure:
    """Build a bar chart straight from a labelled Series.

    Args:
        series: Indexed counts.
        title: Chart title.
        color: Bar colour.
        height: Figure height.
        x_title: X axis title.
        y_title: Y axis title.
        order: Optional category order, e.g. :data:`DAY_ORDER`.

    Returns:
        The figure.
    """
    if series.empty:
        return empty_state(title or "No data", "No records match the current filters.")

    data = series if order is None else series.reindex([o for o in order if o in series.index])
    data = data.dropna()
    return bar_chart(
        data.reset_index().rename(columns={data.index.name or "index": "Category", "count":
            "Accidents"}),
        "Category", "Accidents",
        title=title, color=color, height=height, x_title=x_title, y_title=y_title,
    )


def kpi_frame(items: dict[str, float | int | str], *, height: int = 120) -> go.Figure:
    """Render a compact KPI strip as a figure.

    Used only where native ``st.metric`` cards are not appropriate, such as
    inside a downloaded report.

    Args:
        items: Ordered mapping of label to value.
        height: Figure height.

    Returns:
        The figure.
    """
    labels = list(items.keys())
    figure = go.Figure()
    for index, label in enumerate(labels):
        figure.add_annotation(
            x=index, y=1.0, text=f"<b>{items[label]}</b>",
            showarrow=False, font={"size": 26, "color": "#102a43"},
            xanchor="center", yanchor="top",
        )
        figure.add_annotation(
            x=index, y=0.0, text=label,
            showarrow=False, font={"size": 12, "color": "#627d98"},
            xanchor="center", yanchor="bottom",
        )
    figure.update_xaxes(range=[-0.6, len(labels) - 0.4], showgrid=False, showticklabels=False)
    figure.update_yaxes(range=[-0.4, 1.4], showgrid=False, showticklabels=False)
    figure.update_layout(height=height, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    return figure

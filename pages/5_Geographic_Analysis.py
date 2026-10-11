"""Page 5 — Geographic analysis.

Coordinates are aggregated into grid cells in SQL before anything reaches the
browser. Rendering 7.7 million individual markers is not possible, and a
grid aggregate communicates the same density pattern far more legibly.
"""

from __future__ import annotations

import streamlit as st

from smart_traffic import geospatial
from smart_traffic.config import EXPOSURE_DISCLAIMER, SETTINGS
from smart_traffic.ui import (
    charts_row,
    download_table,
    empty_state,
    global_filters,
    info_box,
    page_header,
    require_dataset,
    section,
    sidebar_header,
)
from smart_traffic.visualizations import bar_chart, density_map

st.set_page_config(page_title="Geographic Analysis", page_icon="🗺️", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Geographic Analysis",
    "Where accidents are recorded, using validated coordinates aggregated into "
    "grid cells.",
    icon="🗺️",
)

spec = global_filters()

info_box(
    "warning",
    "A hotspot of recorded accidents is not the most dangerous place to drive",
    EXPOSURE_DISCLAIMER
    + " Ranking places by accident risk requires traffic volume, road geometry "
    "and driver behaviour data, none of which this dataset contains.",
)

# -- controls --------------------------------------------------------------
section("Density map settings")
controls = st.columns([1, 1, 2])
cell_size = controls[0].select_slider(
    "Grid cell size",
    options=[0.05, 0.1, 0.25, 0.5, 1.0],
    value=geospatial.DEFAULT_CELL_DEGREES,
    format_func=lambda v: f"{v:g}° (~{v * 111:.0f} km)",
    help="Smaller cells resolve finer detail but return more points to the browser.",
)
show_labels = controls[1].toggle(
    "Show Folium heatmap",
    value=False,
    help="An alternative Leaflet rendering. Uses the same aggregated cells.",
)

with st.spinner("Aggregating coordinates into grid cells..."):
    grid = geospatial.density_grid(spec, cell_degrees=cell_size)
    states = geospatial.state_hotspots(spec, limit=SETTINGS.top_n)
    cities = geospatial.top_cities(spec, limit=SETTINGS.top_n)
    coordinates = geospatial.coordinate_distribution(spec)
    by_state_severity = geospatial.state_severity_by_map(spec)
    coverage = geospatial.geographic_coverage(spec)

if grid.cells.empty:
    empty_state(
        "No valid coordinates are available for the current filters.",
        detail="Records with missing or out-of-range coordinates are excluded from maps.",
    )
    st.stop()

st.caption(
    f"{len(grid.cells):,} non-empty grid cells covering {grid.total:,} records "
    f"at {cell_size:g}° resolution."
)

# -- density map -----------------------------------------------------------
section(
    "Accident density",
    "Marker area and colour both encode the number of recorded accidents in a cell.",
)
st.plotly_chart(
    density_map(grid.cells, title="Recorded accidents by grid cell"),
    width="stretch",
    config={"displaylogo": False, "scrollZoom": True},
)

peak = grid.peak
if peak:
    st.caption(
        f"**Densest cell:** latitude {peak['Latitude']}, longitude {peak['Longitude']} "
        f"with {peak['Accidents']:,} recorded accidents. This is the most-recorded "
        f"cell in the filtered data, not the highest-risk location."
    )

if show_labels:
    section(
        "Alternative Leaflet rendering",
        description="The same aggregated cells, drawn with Folium.",
    )
    try:
        import folium
        from streamlit_folium import st_folium

        centre = [float(grid.cells["Latitude"].mean()), float(grid.cells["Longitude"].mean())]
        folium_map = folium.Map(location=centre, zoom_start=4, tiles="CartoDB positron")
        heat_data = [
            [float(row.Latitude), float(row.Longitude), float(row.Accidents)]
            for row in grid.cells.itertuples()
        ]
        folium_map.add_child(
            folium.plugins.HeatMap(
                heat_data,
                radius=8,
                blur=12,
                min_opacity=0.25,
                max_zoom=10,
                name="Recorded accidents",
            )
        )
        folium_map.add_child(folium.LayerControl())
        st_folium(folium_map, height=520, width="stretch")
    except (ImportError, KeyError, ValueError) as exc:
        st.warning(f"The Folium layer could not be rendered: {exc}")

with st.expander("Download the aggregated grid", expanded=False):
    st.caption(
        "This is the exact data behind the map: one row per grid cell, not one row "
        "per accident."
    )
    download_table(grid.cells, "Download grid cells (CSV)", key="dl_grid",
                   filename_hint="density_grid")
    download_table(grid.top_cells(SETTINGS.top_n), "Download busiest cells",
                   key="dl_grid_top", filename_hint="density_grid_top_cells")

# -- state / city rankings -------------------------------------------------
section("Busiest states and cities")
charts_row(
    [
        bar_chart(states, "State", "Accidents", title=f"Top {SETTINGS.top_n} states"),
        bar_chart(cities, "City", "Accidents", title=f"Top {SETTINGS.top_n} cities"),
    ]
)

st.caption(
    f"The loaded data spans {int(coverage['Distinct_Values'][0])} states, "
    f"{int(coverage['Distinct_Values'][1])} counties and "
    f"{int(coverage['Distinct_Values'][2])} cities."
)

# -- coordinate validation -------------------------------------------------
section(
    "Coordinate validation",
    "Coordinates are checked against the USA bounding box before any map is drawn.",
)
if coordinates.empty:
    st.info("No coordinate statistics available.")
else:
    left, right = st.columns([1.3, 1])
    with left:
        st.dataframe(coordinates, width="stretch", hide_index=True)
        st.caption(
            "The 1st and 99th percentiles are more informative than the minimum and "
            "maximum, which are single outliers rather than a meaningful range."
        )
    with right:
        st.markdown("**Why validation matters here**")
        st.markdown(
            """
A small number of records carry coordinates far outside the United States.
Plotting them unfiltered would stretch the map projection and distort the
apparent spread of the data.

Records failing this check carry `Is_Valid_Coordinate = FALSE`. They are excluded
from the maps above and reported on the Data Quality page.
"""
        )

# -- severity by state -----------------------------------------------------
section(
    "High-severity share by state",
    "Percentage of each state's records that are severity 3 or 4. The denominator "
    "is that state's own record count.",
)
if not by_state_severity.empty:
    frame = by_state_severity.head(SETTINGS.top_n)
    st.plotly_chart(
        bar_chart(
            frame, "State", "High_Severity_Rate",
            title=f"Severity 3+4 rate within each of the top {SETTINGS.top_n} states",
            color="#d7191c",
            y_title="Share of the state's records (%)",
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "A high rate in a state with few records can rest on a small denominator. "
        "Check the record count before interpreting it."
    )
    download_table(
        by_state_severity, "Download state severity rates", key="dl_geo_sev",
        filename_hint="state_severity_rates",
    )

# -- export ----------------------------------------------------------------
section("Export")
left, mid, right = st.columns(3)
with left:
    download_table(states, "States", key="dl_states", filename_hint="geo_top_states")
with mid:
    download_table(cities, "Cities", key="dl_cities", filename_hint="geo_top_cities")
with right:
    download_table(
        by_state_severity, "State severity rates", key="dl_geo_rates",
        filename_hint="geo_state_severity",
    )

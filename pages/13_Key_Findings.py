"""Page 13 — Key findings and report export.

Findings are computed live from the loaded data. Nothing on this page is
hardcoded, so a different dataset produces different findings rather than stale
ones.
"""

from __future__ import annotations

import json

import streamlit as st

from smart_traffic.aggregation import Aggregator
from smart_traffic.analytics import (
    build_profile,
    executive_summary,
    generate_findings,
    kpi_cards,
)
from smart_traffic.config import (
    PATHS,
    SETTINGS,
    SEVERITY_REGIME_CAVEAT,
    SEVERITY_REGIME_CONSEQUENCE,
)
from smart_traffic.db import DB
from smart_traffic.reporting import (
    build_analytics_bundle,
    findings_frame,
    load_model_summary,
    provenance,
    write_markdown_report,
)
from smart_traffic.ui import (
    download_table,
    methodology_notes,
    page_header,
    require_dataset,
    section,
    severity_regime_warning,
    sidebar_header,
)

st.set_page_config(page_title="Key Findings", page_icon="💡", layout="wide")
sidebar_header()

if not require_dataset():
    st.stop()

page_header(
    "Key Findings",
    "Computed from the loaded dataset. Every figure here recalculates when the "
    "filters change.",
    icon="💡",
)

from smart_traffic.aggregation import FilterSpec  # noqa: E402

spec = st.session_state.get("filters")
if not isinstance(spec, FilterSpec):
    spec = FilterSpec()

with st.spinner("Computing findings..."):
    profile = build_profile(spec)
    # Read the persisted summary directly. It used to come from session state,
    # which only the Model Evaluation page populates -- so visiting Key Findings
    # directly reported "no models have been trained" while four sat on disk.
    model_summary = load_model_summary()
    findings = generate_findings(spec, model_summary=model_summary)
    cards = kpi_cards(spec)

if not findings:
    st.warning("No findings could be computed for the current filters.", icon="⚠️")
    st.stop()

# The generated findings can name a best model and quote its macro F1, so the
# regime caveat belongs here too rather than only inside the exported report.
if model_summary:
    severity_regime_warning()

# -- headline numbers ------------------------------------------------------
section("Headline numbers")
items = {k: v for k, v in cards.items() if k != "_raw"}
columns = st.columns(len(items))
for column, (label, value) in zip(columns, items.items(), strict=False):
    column.metric(label, value)

st.caption(
    f"Computed over {profile.total_records:,} records spanning "
    f"{profile.coverage_start[:10]} to {profile.coverage_end[:10]}, across "
    f"{profile.distinct_states} states and {profile.distinct_cities:,} cities."
)

# -- findings --------------------------------------------------------------
section(
    "Findings",
    "Each finding is derived from a live query and states the numbers it rests on.",
)

CATEGORY_ICONS = {
    "Severity": "🚦", "Time": "⏰", "Geography": "🗺️",
    "Conditions": "🌧️", "Model": "🤖", "General": "📌",
}

current_category = None
for finding in findings:
    if finding.category != current_category:
        current_category = finding.category
        icon = CATEGORY_ICONS.get(finding.category, "📌")
        st.markdown(f"**{icon} {finding.category}**")

    st.markdown(f"- {finding.headline}")
    if finding.evidence:
        st.caption(f"   *Evidence:* {finding.evidence}")
    if finding.caveat:
        st.caption(f"   *Caveat:* {finding.caveat}")

# -- executive summary -----------------------------------------------------
st.markdown("---")
section(
    "Executive summary",
    "A short narrative suitable for a presentation or viva.",
)
summary_text = executive_summary(profile, findings)
# st.text() takes no height argument; a code block gives a scrollable,
# monospaced rendering that suits a plain-text summary.
st.code(summary_text, language=None)

# -- export ----------------------------------------------------------------
st.markdown("---")
section("Export")
st.caption(
    "Exports carry provenance: the dataset, its coverage, and the filters active "
    "at export time. A CSV without that context is not interpretable."
)

meta = provenance(profile, spec)

left, right = st.columns(2)

with left:
    st.markdown("**Findings and summary**")
    frame = findings_frame(findings)
    download_table(frame, "Download findings (CSV)", key="dl_findings",
                   filename_hint="key_findings")
    st.download_button(
        "Download executive summary (text)",
        data=summary_text.encode("utf-8"),
        file_name="executive_summary.txt",
        mime="text/plain",
        key="dl_summary",
    )
    st.download_button(
        "Download provenance (JSON)",
        data=json.dumps(meta, indent=2, default=str).encode("utf-8"),
        file_name="analysis_provenance.json",
        mime="application/json",
        key="dl_provenance",
    )

with right:
    st.markdown("**Supporting tables**")
    aggregator = Aggregator(DB)
    tables = {
        "Severity distribution": aggregator.severity_counts(spec, DB),
        "By year": aggregator.coverage_by_year(spec, DB),
        "Top states": aggregator.counts_by("State", spec, limit=SETTINGS.top_n, db=DB),
        "Top cities": aggregator.counts_by("City", spec, limit=SETTINGS.top_n, db=DB),
        "By hour": aggregator.counts_by("Hour", spec, order_desc=False, db=DB),
        "By day": aggregator.counts_by("Day_Name", spec, order_desc=False, db=DB),
        "By month": aggregator.counts_by("Month_Name", spec, order_desc=False, db=DB),
        "Road features": aggregator.boolean_feature_counts(spec, DB),
        "State severity rates": aggregator.severity_rates("State", spec, db=DB),
    }
    bundle = build_analytics_bundle(tables, profile, spec, name="key_findings")

    with st.expander("Preview tables", expanded=False):
        selected = st.selectbox("Table", list(bundle.tables))
        st.dataframe(bundle.tables[selected], width="stretch", hide_index=True)
        download_table(
            bundle.tables[selected], f"Download {selected.lower().replace(' ', '_')}",
            key="dl_table", filename_hint=selected.lower().replace(" ", "_"),
        )

    if st.button("Build the full export bundle (.zip)", key="build_bundle"):
        with st.spinner("Packaging tables…"):
            try:
                archive = bundle.as_zip()
                st.session_state["bundle_path"] = str(archive)
            except OSError as exc:
                st.error(f"The bundle could not be written: {exc}")

    bundle_path = st.session_state.get("bundle_path")
    if bundle_path:
        from pathlib import Path

        path = Path(bundle_path)
        st.download_button(
            "Download export bundle (.zip)",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/zip",
            key="dl_bundle",
        )

# -- markdown report -------------------------------------------------------
st.markdown("---")
section("Full report")
st.caption("A Markdown report combining the findings with the model evaluation summary.")

if st.button("Generate Markdown report", key="gen_report"):
    model_section = "_No models have been trained._"
    if model_summary:
        lines = ["| Model | Macro F1 | Balanced accuracy | Accuracy |", "| --- | --- | --- | --- |"]
        for entry in model_summary.values():
            metrics = entry["metrics"]
            lines.append(
                f"| {entry['label']} | {metrics['Macro_F1']:.4f} | "
                f"{metrics['Balanced_Accuracy']:.4f} | {metrics['Accuracy']:.4f} |"
            )
        lines += [
            "",
            "**These are same-dataset holdout figures, not a forecast.** "
            + SEVERITY_REGIME_CAVEAT,
            "",
            SEVERITY_REGIME_CONSEQUENCE,
        ]
        model_section = "\n".join(lines)

    path = write_markdown_report(
        PATHS.analytics / "smart_traffic_report.md",
        title="Smart Traffic Analysis — Findings Report",
        sections=[
            ("Dataset", (
                f"- Records analysed: {profile.total_records:,}\n"
                f"- Unique accident IDs: {profile.unique_ids:,}\n"
                f"- Coverage: {profile.coverage_start[:10]} to {profile.coverage_end[:10]}\n"
                f"- States: {profile.distinct_states}\n"
                f"- Cities: {profile.distinct_cities:,}\n"
                f"- Active filters: {'; '.join(meta.get('active_filters', []))}"
            )),
            ("Executive summary", executive_summary(profile, findings)),
            ("Findings", frame.to_markdown(index=False)),
            ("Model evaluation", model_section),
            ("Methodology", (
                "1. Records are ingested from the source CSV in fixed-size chunks "
                "into Parquet, bounding memory regardless of file size. No record is "
                "dropped: invalid values are flagged and counted.\n"
                "2. Aggregates are computed in DuckDB directly against the Parquet file.\n"
                f"3. Expensive analysis uses reproducible samples selected by "
                f"hash(ID, seed={SETTINGS.random_seed}).\n"
                "4. Models use only features available at accident onset; duration and "
                "End_Time are excluded by an enforced policy.\n"
                "5. Models are compared on macro F1 and balanced accuracy because the "
                "target is heavily imbalanced."
            )),
            ("Limitations", (
                f"- **{SEVERITY_REGIME_CAVEAT}**\n"
                "- Severity records traffic impact, not human harm.\n"
                "- Counts are recorded frequency, not exposure-adjusted risk.\n"
                "- All relationships are observational associations, not causal.\n"
                "- The final year in the dataset is partial.\n"
                "- Models are exploratory and not validated for operational use."
            )),
        ],
    )
    st.session_state["report_path"] = str(path)

report_path = st.session_state.get("report_path")
if report_path:
    from pathlib import Path

    path = Path(report_path)
    st.download_button(
        "Download Markdown report",
        data=path.read_text(encoding="utf-8").encode("utf-8"),
        file_name=path.name,
        mime="text/markdown",
        key="dl_report",
    )

methodology_notes()

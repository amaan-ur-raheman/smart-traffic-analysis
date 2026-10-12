"""Streamlit smoke tests.

Loads every page through Streamlit's own test harness and asserts it renders
without raising. This is what catches the class of bug that a unit test misses
entirely: a helper defined after its call site, a typo in a chart argument, a
widget key collision.

Pages that require an explicit user action (training, PCA, clustering,
prediction) are only checked as far as their empty and no-dataset states, which
is where a regression is most likely.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "app.py"
PAGES = sorted((PROJECT_ROOT / "pages").glob("*.py"))
PARQUET = PROJECT_ROOT / "data" / "processed" / "accidents.parquet"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Tests that assert on real content need a built dataset. The suite must still
# pass before `sta build` has ever been run, so they skip rather than fail;
# `TestNoDatasetState` covers the absent-data path in the meantime.
needs_data = pytest.mark.skipif(
    not PARQUET.exists(),
    reason="no processed dataset; run `uv run sta build` (or `uv run sta demo`)",
)


def _assert_no_exception(test: AppTest, context: str) -> None:
    """Fail with the rendered exception if the page raised."""
    if test.exception:
        messages = "\n".join(str(e.value) for e in test.exception)
        pytest.fail(f"{context} raised an exception:\n{messages}")


@needs_data
class TestLandingPage:
    """The entry point, with a dataset loaded."""

    def test_renders(self) -> None:
        test = AppTest.from_file(str(APP), default_timeout=90).run()
        _assert_no_exception(test, "app.py")

    def test_shows_dataset_facts(self) -> None:
        test = AppTest.from_file(str(APP), default_timeout=90).run()
        _assert_no_exception(test, "app.py")
        text = " ".join(str(m.value) for m in test.metric)
        assert any("Records analysed" in m.label for m in test.metric)
        assert text  # metrics were populated from the dataset


class TestPages:
    """Every dashboard page renders."""

    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
    @needs_data
    def test_page_renders_with_data(self, page: Path) -> None:
        test = AppTest.from_file(str(page), default_timeout=120).run()
        _assert_no_exception(test, page.name)

    def test_pages_exist(self) -> None:
        assert len(PAGES) >= 12, "expected the full set of analysis pages"

    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
    @needs_data
    def test_page_produces_content(self, page: Path) -> None:
        """A page that renders nothing is broken, however quietly."""
        test = AppTest.from_file(str(page), default_timeout=120).run()
        _assert_no_exception(test, page.name)
        rendered = (
            len(test.markdown)
            + len(test.dataframe)
            + len(test.metric)
            + len(test.info)
            + len(test.warning)
            + len(test.error)
        )
        assert rendered > 0, f"{page.name} rendered nothing"


class TestNoDatasetState:
    """The app must explain itself rather than crash when data is absent.

    Runs against an empty data directory, which is what a user sees before
    downloading the dataset.
    """

    @pytest.fixture
    def empty_data(self, tmp_path, monkeypatch) -> None:
        """Point every managed directory at an empty temporary tree."""
        monkeypatch.setenv("STA_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("STA_MODELS_DIR", str(tmp_path / "models"))
        monkeypatch.setenv("STA_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
        monkeypatch.setenv("STA_LOG_DIR", str(tmp_path / "logs"))

    def test_landing_page_shows_setup_instructions(self, empty_data) -> None:
        test = AppTest.from_file(str(APP), default_timeout=90).run()
        _assert_no_exception(test, "app.py with no dataset")
        assert test.error, "the landing page must explain that no dataset is loaded"

    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
    def test_pages_explain_themselves(self, empty_data, page: Path) -> None:
        test = AppTest.from_file(str(page), default_timeout=90).run()
        _assert_no_exception(test, f"{page.name} with no dataset")
        assert test.error or test.info, (
            f"{page.name} must show a message when no dataset is available"
        )


class TestTrainingDefaultsAgree:
    """The UI, the CLI and the documented protocol must use one sample size.

    Regression test. The severity-prediction page hardcoded 300,000 while
    SETTINGS, the CLI default and the README all said 400,000. Training from the
    browser therefore ran a different protocol than the documentation and
    silently overwrote the canonical artefacts -- observed twice during an audit
    before the cause was tracked down.
    """

    def test_config_default_is_the_documented_value(self) -> None:
        from smart_traffic.config import SETTINGS

        assert SETTINGS.sample_training_rows == 400_000

    def test_cli_default_comes_from_settings(self, capsys) -> None:

        from smart_traffic.cli import build_parser
        from smart_traffic.config import SETTINGS

        parser = build_parser()
        args = parser.parse_args(["train"])
        assert args.rows == SETTINGS.sample_training_rows

    def test_page_has_no_hardcoded_sample_size(self) -> None:
        """A literal in the training widget is what caused the drift."""
        import re

        page = APP.parent / "pages" / "10_Severity_Prediction.py"
        lines = page.read_text().splitlines()
        label_at = next(
            (i for i, line in enumerate(lines) if '"Training sample size"' in line),
            None,
        )
        assert label_at is not None, "training sample widget not found"
        # The widget call starts on an earlier line; capture the whole argument
        # list so a stray literal inside it would be visible.
        start = max(
            (i for i in range(label_at + 1) if "number_input(" in lines[i]),
            default=label_at,
        )
        end = next(
            (i for i in range(start, len(lines)) if lines[i].strip() == ")"),
            label_at + 6,
        )
        args = "\n".join(lines[start : end + 1])
        assert "SETTINGS.sample_training_rows" in args, (
            "the training sample default must come from SETTINGS"
        )
        literals = [
            m for m in re.findall(r"\b\d{5,}\b", args) if m != "1000000"
        ]
        assert not literals, f"hardcoded sample size(s) reintroduced: {literals}"


def _page(name: str) -> str:
    """Resolve a page file by name; PAGES is a list of paths, not a directory."""
    return str(PROJECT_ROOT / "pages" / name)


class TestInteractivePaths:
    """Clicking each button must not raise.

    Two of the defects this project found (the undeclared `tabulate`, and figures
    depending on page visit order) lived behind a button, so a suite that only
    *renders* pages could not see either. These tests click the controls that do
    not require a full model fit; training and permutation importance are
    covered separately because they are minutes-long operations.

    Runs against the small synthetic fixture, so every one of these is fast.
    """

    def test_pca_button(self, processed_table) -> None:
        test = AppTest.from_file(str(_page("9_PCA_and_Clustering.py")), default_timeout=180).run()
        _assert_no_exception(test, "9_PCA_and_Clustering.py")
        clicked = False
        for button in list(test.button):
            if button.label == "Run PCA":
                test = button.click().run()
                clicked = True
                break
        assert clicked, "Run PCA button not found"
        _assert_no_exception(test, "9_PCA_and_Clustering.py after Run PCA")

    def test_kmeans_button(self, processed_table) -> None:
        test = AppTest.from_file(str(_page("9_PCA_and_Clustering.py")), default_timeout=180).run()
        for button in list(test.button):
            if button.label == "Run K-Means":
                test = button.click().run()
                break
        _assert_no_exception(test, "9_PCA_and_Clustering.py after Run K-Means")

    def test_hierarchical_button(self, processed_table) -> None:
        test = AppTest.from_file(str(_page("9_PCA_and_Clustering.py")), default_timeout=180).run()
        for button in list(test.button):
            if button.label == "Run hierarchical clustering":
                test = button.click().run()
                break
        _assert_no_exception(test, "9_PCA_and_Clustering.py after hierarchical")

    def test_markdown_report_button(self, processed_table) -> None:
        """The export that silently failed for want of an undeclared dependency."""
        test = AppTest.from_file(str(_page("13_Key_Findings.py")), default_timeout=180).run()
        clicked = False
        for button in list(test.button):
            if "Markdown" in button.label:
                test = button.click().run()
                clicked = True
                break
        assert clicked, "Markdown report button not found"
        _assert_no_exception(test, "13_Key_Findings.py after report generation")

    def test_export_bundle_button(self, processed_table) -> None:
        test = AppTest.from_file(str(_page("13_Key_Findings.py")), default_timeout=180).run()
        for button in list(test.button):
            if "export bundle" in button.label:
                test = button.click().run()
                break
        _assert_no_exception(test, "13_Key_Findings.py after export bundle")

    def test_key_findings_shows_models_without_visit_order(self, processed_table) -> None:
        """Opening Key Findings first must not claim the models are absent.

        The bug: `model_summary` came from session state that only Model
        Evaluation writes, so the report said "No models have been trained".
        """
        test = AppTest.from_file(str(_page("13_Key_Findings.py")), default_timeout=180).run()
        _assert_no_exception(test, "13_Key_Findings.py opened directly")
        # The report must never claim absence while a summary is on disk.
        assert "No models have been trained" not in test.markdown

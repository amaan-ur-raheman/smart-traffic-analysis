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

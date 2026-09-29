"""The console's HTML and JS have to agree about element ids.

This is the cheap half of a browser test, and it catches the one mistake that
editing both files makes easy: renaming an element in the markup and leaving
the lookup behind. The symptom is a page that loads, looks fine, and silently
does nothing when you click -- no error, because `document.getElementById`
returns null and the following property access throws inside a handler nobody
is watching.

Found the hard way: the console's stylesheet and script were served from a
different path than the page that referenced them, so the page arrived with no
styling and no behaviour while `GET /` returned 200 and every test passed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CONSOLE = Path(__file__).resolve().parents[1] / "src/unified_agent/api/console"


@pytest.fixture(scope="module")
def html() -> str:
    return (CONSOLE / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return (CONSOLE / "app.js").read_text(encoding="utf-8")


def ids_in(markup: str) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', markup))


def ids_looked_up(script: str) -> set[str]:
    """`$("x")` and `getElementById("x")`, which is how this file reads the DOM."""
    found = set(re.findall(r'\$\("([^"]+)"\)', script))
    found |= set(re.findall(r'getElementById\("([^"]+)"\)', script))
    return found


class TestConsoleWiring:
    def test_every_looked_up_id_exists_in_the_markup(self, html: str, js: str) -> None:
        missing = sorted(ids_looked_up(js) - ids_in(html))
        assert not missing, (
            f"app.js reads {missing} but index.html does not define them. "
            "The lookup returns null, and the failure is silent until someone "
            "clicks the thing."
        )

    #: Ids that are not read by the script, and why that is correct.
    #: Listed rather than skipped so a *new* orphan shows up as a failure.
    NOT_READ_BY_SCRIPT = {
        "app": "layout root, styled only",
        "main": "layout region, styled only",
        "sidebar": "layout region, styled only",
        "topbar": "layout region; the script reads its children, not itself",
        "approval-title": "referenced by aria-labelledby",
        "btn-send": "a submit button; the form's submit event handles it",
    }

    def test_the_markup_has_no_unexplained_ids(self, html: str, js: str) -> None:
        """An id nothing reads is either dead markup or a lookup that was
        deleted -- both worth knowing about, so the exemptions are explicit."""
        orphans = sorted(ids_in(html) - ids_looked_up(js) - set(self.NOT_READ_BY_SCRIPT))
        assert not orphans, (
            f"index.html defines {orphans}, which nothing reads. If that is "
            "deliberate, add it to NOT_READ_BY_SCRIPT with a reason."
        )

    def test_the_page_references_a_stylesheet_and_a_script(self, html: str) -> None:
        assert re.search(r'<link[^>]+href="style\.css"', html), "no stylesheet link"
        assert re.search(r'<script[^>]+src="app\.js"', html), "no script tag"

    def test_every_class_the_script_applies_is_styled(self, html: str, js: str) -> None:
        """Not exhaustive -- only the classes that carry state.

        A class the script toggles and the stylesheet never styles is a state
        the user cannot see, which is the same as not having it.
        """
        css = (CONSOLE / "style.css").read_text(encoding="utf-8")
        for name in [
            "tool",
            "running",
            "ok",
            "failed",
            "plan-step",
            "completed",
            "skipped",
            "task-item",
            "active",
            "chip-ok",
            "chip-warn",
            "chip-danger",
            "chip-accent",
            "hidden",
            "busy",
        ]:
            assert f".{name}" in css, f"the script uses .{name} but the stylesheet has no rule"

    def test_the_status_states_have_colours(self, html: str, js: str) -> None:
        """`setStatus` writes the state into a data attribute so the colour
        follows the state rather than the wording."""
        css = (CONSOLE / "style.css").read_text(encoding="utf-8")
        assert "dataset.state" in js, "setStatus should expose the state as a data attribute"
        for state in ["completed", "failed", "cancelled", "waiting_confirmation"]:
            assert f'[data-state="{state}"]' in css, f"no colour for status {state}"

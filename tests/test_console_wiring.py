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

CONSOLE = Path(__file__).resolve().parents[1] / "src/wukong/api/console"


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


class TestInterfaceIsChinese:
    """The interface is Chinese, and there are exactly two ways that breaks.

    One is a string left in English. The other is subtler and worse: a status
    is translated while the *key* it is looked up by is not, so
    `[data-state="completed"]` stops matching the moment the label becomes
    "已完成" and every status silently loses its colour.
    """

    FORBIDDEN_IN_MARKUP = [
        "Approval required",
        "Approve",
        "Refuse",
        "Describe a task",
        "Enter to send",
        "Nothing yet",
        "Pre-approve",
        "What should",
    ]

    FORBIDDEN_IN_SCRIPT = [
        "Nothing yet",
        "outcome unknown",
        "unknown error",
        "waiting for approval",
        "was interrupted mid-flight",
        "This server requires a session token",
        "steps · ",
    ]

    def test_no_english_left_in_the_markup(self, html: str) -> None:
        for phrase in self.FORBIDDEN_IN_MARKUP:
            assert phrase not in html, f"{phrase!r} is still English in index.html"

    def test_no_english_ui_prose_in_the_script(self, js: str) -> None:
        """Developer logs (`console.warn`) are exempt; anything a user reads is
        not."""
        for phrase in self.FORBIDDEN_IN_SCRIPT:
            assert phrase not in js, f"{phrase!r} is still English in app.js"

    def test_the_page_declares_chinese(self, html: str) -> None:
        assert 'lang="zh-CN"' in html

    def test_the_chinese_font_is_in_the_stack(self, html: str, js: str) -> None:
        """Latin-first with CJK as an afterthought is what makes a Chinese
        interface look like a translated English one."""
        css = (CONSOLE / "style.css").read_text(encoding="utf-8")
        for family in ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC"]:
            assert family in css, f"{family} is missing from the font stack"

    def test_letter_spacing_is_not_applied_to_labels(self, html: str, js: str) -> None:
        """Tracking is a Latin-uppercase device. On hanzi it opens gaps that
        read as broken spacing.

        Comments are stripped first: a rule explaining *why* it does not set
        letter-spacing contains the words "letter-spacing".
        """
        css = re.sub(r"/\*.*?\*/", "", (CONSOLE / "style.css").read_text(encoding="utf-8"), flags=re.S)
        for selector in [".panel-title", ".turn-head", ".plan-label"]:
            block = re.search(re.escape(selector) + r"\s*\{(.*?)\}", css, re.S)
            assert block, f"{selector} not found"
            assert "letter-spacing" not in block.group(1), (
                f"{selector} carries letter-spacing; its text is Chinese"
            )

    def test_status_labels_are_chinese(self, js: str) -> None:
        block = self._status_label_block(js)
        pairs = re.findall(r"(\w+):\s*\"([^\"]+)\"", block)
        assert pairs, "STATUS_LABEL did not parse -- did its shape change?"
        for key, label in pairs:
            assert re.search(r"[\u4e00-\u9fff]", label), f"{key} -> {label!r} is not Chinese"

    def test_every_status_the_script_sets_has_a_label(self, js: str) -> None:
        keys = set(re.findall(r'setStatus\("([a-z_]+)"', js))
        block = self._status_label_block(js)
        labels = set(re.findall(r"(\w+):\s*\"[^\"]+\"", block))
        missing = sorted(key for key in keys if key not in labels)
        assert not missing, f"setStatus({missing}) would display the raw key"

    def test_the_state_key_is_not_the_label(self, js: str) -> None:
        """The trap this whole class exists for."""
        assert "node.dataset.state = state;" in js, (
            "setStatus must write the state key, not the display label"
        )

    def test_the_composer_does_not_submit_mid_composition(self, js: str) -> None:
        """With a Chinese IME, Enter picks a pinyin candidate. Without this
        check that Enter also sends, and the agent starts on a half-typed
        sentence."""
        assert "compositionstart" in js and "compositionend" in js
        assert "event.isComposing" in js

    @staticmethod
    def _status_label_block(js: str) -> str:
        match = re.search(r"const STATUS_LABEL = \{(.*?)\};", js, re.S)
        assert match, "STATUS_LABEL not found"
        return match.group(1)

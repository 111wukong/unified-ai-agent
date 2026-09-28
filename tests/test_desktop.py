"""Desktop shell: server lifecycle, port selection, and the .app bundle.

What is verifiable here and what is not, stated plainly: the *window* cannot
be opened in a headless environment, so no test asserts that a window
appeared. Everything around it can be, and is -- the server lifecycle the
window wraps, port selection, the health gate, the bundle layout and its
Info.plist, and the generated icon bytes. `uaa desktop --check` exercises the
same lifecycle outside the test suite.

None of these tests need pywebview installed: the module imports it lazily so
the lifecycle stays testable without a GUI dependency. That is deliberate. A
test that `importorskip`s a GUI library silently vanishes in CI, which is the
failure mode this project treats as a bug -- so there are no skips here
either; the macOS-only paths assert the refusal on other platforms instead.
"""

from __future__ import annotations

import asyncio
import json
import plistlib
import socket
import sys
import zlib
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from unified_agent.agent.factory import build_agent
from unified_agent.api.app import Service
from unified_agent.desktop import bundle as bundle_mod
from unified_agent.desktop.launcher import (
    DesktopUnavailable,
    ServerHandle,
    desktop_available,
    pick_port,
    wait_for_health,
)

IS_MACOS = sys.platform == "darwin"


def build_or_assert_macos_only(tmp_path: Path, **kwargs):  # noqa: ANN201
    """Build a bundle on macOS; on other platforms assert the refusal.

    Returns None when the platform check fired, so the caller can return
    early without recording a skip.
    """
    if not IS_MACOS:
        with pytest.raises(bundle_mod.BundleError, match="macOS-only"):
            bundle_mod.build_app_bundle(target_dir=tmp_path, version="0.1.0", **kwargs)
        return None
    return bundle_mod.build_app_bundle(target_dir=tmp_path, version="0.1.0", **kwargs)


class TestPortSelection:
    def test_returns_a_port_that_can_be_bound(self) -> None:
        port = pick_port()
        assert 1024 < port < 65536
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))

    def test_does_not_collide_with_a_port_already_held(self) -> None:
        first = pick_port()
        with socket.socket() as held:
            held.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            held.bind(("127.0.0.1", first))
            assert pick_port() != first


class TestHealthGate:
    def test_times_out_quickly_on_a_dead_port(self) -> None:
        port = pick_port()
        assert wait_for_health(f"http://127.0.0.1:{port}/", timeout_s=0.5) is False

    def test_start_refuses_a_server_that_never_answers(self) -> None:
        handle = ServerHandle(host="127.0.0.1", port=pick_port(), token="t")

        class DeadApp:
            async def __call__(self, scope, receive, send):  # noqa: ANN001, ANN202
                return None

        with pytest.raises(DesktopUnavailable, match="did not become healthy"):
            handle.start(DeadApp(), timeout_s=1.0)


class TestServerLifecycle:
    """The part `uaa desktop` actually depends on."""

    def test_start_stop_round_trip_and_the_token_guard(self, settings) -> None:  # noqa: ANN001
        from tests.conftest import ScriptedModel, ScriptedModels

        model = ScriptedModel(settings.models["scripted"], [{"content": "hi"}])
        service = Service(settings)
        service.agent = asyncio.run(
            build_agent(settings=settings, models=ScriptedModels(model))
        )
        handle = ServerHandle(host="127.0.0.1", port=pick_port(), token="secret-token")
        # `build_app` binds the app to the handle's token, so the token has a
        # single origin. Building the app separately is how the console ends
        # up holding a token the server rejects.
        app = handle.build_app(settings, service=service)
        try:
            handle.start(app, timeout_s=20)
            assert handle.base_url == f"http://127.0.0.1:{handle.port}"

            with pytest.raises(Exception, match="403"):
                urlopen(f"{handle.base_url}/api/v1/health", timeout=5)

            authorised = Request(
                f"{handle.base_url}/api/v1/health",
                headers={"X-UAA-Token": "secret-token"},
            )
            with urlopen(authorised, timeout=5) as response:
                body = json.loads(response.read())
            assert body["status"] == "ok"
            assert body["token_required"] is True
        finally:
            handle.stop()
            service.agent.close()

    def test_stop_is_idempotent(self, settings) -> None:  # noqa: ANN001
        handle = ServerHandle(host="127.0.0.1", port=pick_port(), token="t")
        handle.stop()
        handle.stop()

    def test_console_url_puts_the_token_in_the_fragment(self) -> None:
        """Fragments are never sent to the server and never hit a Referer."""
        handle = ServerHandle(host="127.0.0.1", port=1234, token="abc")
        assert handle.console_url == "http://127.0.0.1:1234/#abc"
        assert "?token=" not in handle.console_url


class TestAvailabilityReporting:
    def test_reports_either_ready_or_a_reason(self) -> None:
        """Asserted on the contract, not on this host's imports.

        The reason is the underlying ImportError, which differs between "the
        package is absent" and "a transitive dependency is absent". Pinning
        the text to either one makes the test fail on a host that is merely
        configured differently -- which is what happened in CI.
        """
        ok, detail = desktop_available()
        assert isinstance(ok, bool)
        assert detail, "availability must always explain itself"
        if not ok:
            assert "desktop window" in detail
            assert "[desktop]" in detail, "the reason must say how to fix it"


class TestPngWriter:
    def test_writes_a_structurally_valid_png(self, tmp_path: Path) -> None:
        width = height = 8
        rgba = bytes([200, 100, 50, 255] * (width * height))
        path = tmp_path / "t.png"
        bundle_mod.write_png(path, width, height, rgba)

        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert b"IHDR" in data and b"IDAT" in data and b"IEND" in data
        assert int.from_bytes(data[16:20], "big") == width
        assert int.from_bytes(data[20:24], "big") == height

    def test_pixels_survive_the_round_trip(self, tmp_path: Path) -> None:
        width = height = 4
        rgba = bytes([1, 2, 3, 4] * (width * height))
        path = tmp_path / "t.png"
        bundle_mod.write_png(path, width, height, rgba)

        data = path.read_bytes()
        start = data.index(b"IDAT") + 4
        length = int.from_bytes(data[start - 8 : start - 4], "big")
        raw = zlib.decompress(data[start : start + length])
        assert len(raw) == height * (1 + width * 4)
        assert raw[1:5] == bytes([1, 2, 3, 4])

    def test_rejects_a_mismatched_buffer(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="expected"):
            bundle_mod.write_png(tmp_path / "bad.png", 4, 4, b"\x00" * 10)


class TestIconRendering:
    def test_produces_the_requested_size(self) -> None:
        assert len(bundle_mod.render_icon(32, supersample=1)) == 32 * 32 * 4

    def test_corners_are_transparent_and_the_centre_is_not(self) -> None:
        """A rounded tile: a filled corner would make the icon look square."""
        size = 64
        rgba = bundle_mod.render_icon(size, supersample=2)

        def alpha(x: int, y: int) -> int:
            return rgba[(y * size + x) * 4 + 3]

        assert alpha(0, 0) == 0, "top-left corner should be transparent"
        assert alpha(size - 1, 0) == 0
        assert alpha(size // 2, size // 2) > 0, "the mark itself must be opaque"

    def test_draws_both_the_tile_and_the_accent(self) -> None:
        rgba = bundle_mod.render_icon(128, supersample=1)
        pixels = {tuple(rgba[i : i + 4]) for i in range(0, len(rgba), 4)}
        assert bundle_mod.ACCENT in pixels, "the chevron should be drawn"
        assert bundle_mod.BG in pixels, "the tile background should be drawn"

    def test_is_deterministic(self) -> None:
        """Compare across a cache clear, or the cache makes this vacuous."""
        first = bundle_mod.render_icon(32, supersample=1)
        bundle_mod.render_icon.cache_clear()
        second = bundle_mod.render_icon(32, supersample=1)
        assert first == second

    def test_supersampling_is_bounded_by_size(self) -> None:
        """A flat 4x at 1024 px is ~17M Python iterations; that is minutes."""
        assert bundle_mod._supersample_for(16) == 4
        assert bundle_mod._supersample_for(1024) == 1
        worst = max(
            size * bundle_mod._supersample_for(size) ** 2 for size in bundle_mod.ICONSET_SIZES
        )
        assert worst <= 1024, f"largest sampled grid is {worst}px"

    def test_rendering_the_whole_iconset_is_fast_enough(self) -> None:
        import time

        bundle_mod.render_icon.cache_clear()
        started = time.perf_counter()
        for size in bundle_mod.ICONSET_SIZES:
            bundle_mod.render_icon(size)
        elapsed = time.perf_counter() - started
        assert elapsed < 10.0, f"icon set took {elapsed:.1f}s; --bundle would feel broken"


class TestAppBundle:
    def test_builds_a_well_formed_bundle(self, tmp_path: Path) -> None:
        result = build_or_assert_macos_only(tmp_path)
        if result is None:
            return
        assert result.path.name == "UnifiedAgent.app"
        assert result.mode == "launcher"
        assert result.notes, "the limitations must be stated, not implied"
        assert any("launcher bundle" in note for note in result.notes), (
            "the relocatability limitation must be stated, not implied"
        )

        contents = result.path / "Contents"
        executable = contents / "MacOS" / "UnifiedAgent"
        assert executable.is_file()
        assert executable.stat().st_mode & 0o111, "the launcher must be executable"
        assert (contents / "Resources").is_dir()

        script = executable.read_text(encoding="utf-8")
        assert script.startswith("#!/bin/sh")
        assert "-m unified_agent.desktop" in script
        # A double-clicked app that does nothing is the worst outcome, so the
        # script has to explain a missing interpreter rather than exit quietly.
        assert "is missing" in script

    def test_info_plist_is_valid_and_points_at_the_launcher(self, tmp_path: Path) -> None:
        result = build_or_assert_macos_only(tmp_path)
        if result is None:
            return
        plist = plistlib.loads((result.path / "Contents" / "Info.plist").read_bytes())
        assert plist["CFBundleExecutable"] == "UnifiedAgent"
        assert plist["CFBundleIdentifier"] == bundle_mod.BUNDLE_ID
        assert plist["CFBundlePackageType"] == "APPL"
        assert plist["CFBundleShortVersionString"] == "0.1.0"
        assert plist["NSHighResolutionCapable"] is True

    def test_rebuild_replaces_rather_than_merges(self, tmp_path: Path) -> None:
        result = build_or_assert_macos_only(tmp_path)
        if result is None:
            return
        stale = result.path / "Contents" / "stale.txt"
        stale.write_text("x", encoding="utf-8")
        rebuilt = bundle_mod.build_app_bundle(target_dir=tmp_path, version="2.0.0")
        assert not (rebuilt.path / "Contents" / "stale.txt").exists()

    def test_frozen_mode_explains_what_is_missing(self, tmp_path: Path) -> None:
        """The frozen path is not wired up, and says so instead of half-working."""
        with pytest.raises(bundle_mod.BundleError, match="PyInstaller"):
            bundle_mod.build_app_bundle(target_dir=tmp_path, version="1.0.0", frozen=True)

    def test_icns_is_optional_not_fatal(self, tmp_path: Path) -> None:
        """No `iconutil` must degrade to no icon, not to a failed build."""
        result = bundle_mod.build_icns(tmp_path)
        if result:
            assert result.path is not None and result.path.exists()
        else:
            assert result.reason, "a missing icon must say why"

    def test_the_iconset_directory_is_named_with_the_extension(self, tmp_path: Path) -> None:
        """`iconutil` rejects any staging directory not named `*.iconset`.

        The naming rule is asserted as a constant so it holds on hosts
        without iconutil, where nothing gets staged at all; the staging
        itself is asserted only where it can happen.
        """
        assert bundle_mod.ICONSET_NAME.endswith(".iconset"), (
            "iconutil rejects any other name with 'Invalid Iconset'"
        )

        result = bundle_mod.build_icns(tmp_path)
        if bundle_mod.iconutil_path() is None:
            # No iconutil here, so nothing was staged. The naming rule above
            # still held; there is nothing else to check.
            assert not result and result.reason
            return

        import tempfile

        staging = Path(tempfile.gettempdir()) / bundle_mod.ICONSET_NAME
        assert staging.is_dir(), "the staging directory must exist where it was used"
        assert any(staging.glob("icon_*.png"))

    def test_bundle_identifier_is_reverse_dns(self) -> None:
        parts = bundle_mod.BUNDLE_ID.split(".")
        assert len(parts) >= 3 and all(parts), bundle_mod.BUNDLE_ID

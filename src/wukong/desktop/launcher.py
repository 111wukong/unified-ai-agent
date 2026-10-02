"""Desktop shell.

The window is the operating system's own webview (WKWebView on macOS) wrapped
around the *same* FastAPI server and the *same* console the browser uses. That
is the whole design: no second UI, no second build system, no bundled
Chromium.

Why not Electron or Tauri:

* **Electron** adds ~150 MB per app and brings npm into a repository that
  deliberately has none. The console is already plain JS and CSS.
* **Tauri** ships smaller binaries, but it means maintaining a Rust + Node
  toolchain inside a Python project, for a window that pywebview opens in
  three lines.

The honest cost of this choice: the app bundle is a launcher around the
installed package, not a frozen binary, unless PyInstaller is available. That
is stated in `wukong desktop --bundle` output rather than hidden.
"""

from __future__ import annotations

import contextlib
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_WINDOW = (1180, 820)
MIN_WINDOW = (760, 560)


class DesktopUnavailable(RuntimeError):
    """Raised when the platform cannot show a window."""


def pick_port(preferred: int = 0, *, host: str = "127.0.0.1") -> int:
    """Find a free TCP port.

    Binding to port 0 and reading the assigned port is the standard trick;
    there is an unavoidable race between closing the probe socket and the
    server binding it, which is why the caller retries on failure rather than
    treating this as a guarantee.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, preferred))
        return int(probe.getsockname()[1])


def wait_for_health(
    url: str,
    *,
    token: str | None = None,
    timeout_s: float = 30.0,
    interval_s: float = 0.1,
) -> bool:
    """Poll until the server answers. Returns False on timeout.

    `token` is not optional in spirit: the health endpoint sits behind the
    same guard as every other API route, so a probe that does not present the
    token gets a 403 forever and the launcher reports "did not become
    healthy" for a server that is running perfectly. That is exactly the kind
    of failure that looks like a bug in the server.
    """
    request = Request(url, headers={"X-WUKONG-Token": token} if token else {})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urlopen(request, timeout=1.0) as response:  # noqa: S310 - loopback only
                if response.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - not up yet is the expected case
            time.sleep(interval_s)
    return False


@dataclass
class ServerHandle:
    """A uvicorn server running on a background thread."""

    host: str
    port: int
    token: str
    _server: Any = None
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def console_url(self) -> str:
        # The token goes in the fragment, not the query string: fragments are
        # never sent to the server and never appear in a Referer header, so it
        # cannot leak through the page itself.
        return f"{self.base_url}/#{self.token}"

    def build_app(self, settings: Any, **kwargs: Any) -> Any:
        """Create the ASGI app bound to *this* handle's token.

        The token has exactly one origin. Building the app separately and
        handing the token to both places is how the console ends up with a
        token the server does not accept -- a mismatch that only shows up as
        a 403 in the window, with nothing in the logs.
        """
        from wukong.api.app import create_app

        kwargs.setdefault("allowed_hosts", [self.host])
        return create_app(settings, token=self.token, **kwargs)

    def start(self, app: Any, *, timeout_s: float = 30.0) -> None:
        import uvicorn

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="wukong-uvicorn", daemon=True
        )
        self._thread.start()
        if not wait_for_health(
            f"{self.base_url}/api/v1/health", token=self.token, timeout_s=timeout_s
        ):
            self.stop()
            raise DesktopUnavailable(
                f"the server did not become healthy at {self.base_url} within {timeout_s:.0f}s"
            )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None


def _webview() -> Any:
    try:
        import webview  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise DesktopUnavailable(
            "the desktop window needs pywebview: pip install -e \".[desktop]\""
        ) from exc
    return webview


def create_window(
    url: str,
    *,
    title: str = "wukong",
    size: tuple[int, int] = DEFAULT_WINDOW,
    min_size: tuple[int, int] = MIN_WINDOW,
    on_closed: Any = None,
) -> Any:
    """Open the native window. Blocks until the user closes it."""
    webview = _webview()
    window = webview.create_window(
        title,
        url,
        width=size[0],
        height=size[1],
        min_size=min_size,
        # The console is local and self-contained; nothing in it should open a
        # second webview inside the app.
        confirm_close=False,
        text_select=True,
    )
    if on_closed is not None:
        window.events.closed += on_closed
    return window


def run_desktop(
    *,
    settings: Any,
    port: int = 0,
    token: str | None = None,
    size: tuple[int, int] = DEFAULT_WINDOW,
    debug: bool = False,
    headless_check: bool = False,
) -> int:
    """Start the server, open the window, and block until it closes.

    `headless_check=True` starts and stops the server without a window. That
    exists so the lifecycle can be exercised in CI, where there is no display
    -- the window itself is the one part that cannot be verified there.
    """
    resolved_token = token if token is not None else secrets.token_urlsafe(24)
    handle = ServerHandle(host="127.0.0.1", port=pick_port(port), token=resolved_token)
    handle.start(handle.build_app(settings))

    if headless_check:
        handle.stop()
        return 0

    try:
        create_window(
            handle.console_url,
            size=size,
            on_closed=lambda: None,
        )
        webview = _webview()
        # `start()` hands control to the GUI loop and returns when it exits.
        webview.start(debug=debug)
    except DesktopUnavailable:
        raise
    finally:
        handle.stop()
    return 0


def desktop_available() -> tuple[bool, str]:
    """Can this machine show a window? Answers without opening one.

    The ImportError message is passed through rather than replaced with a
    generic "not installed". A missing transitive dependency and a missing
    pywebview are different problems, and telling the user to install a
    package they already have sends them in a circle.
    """
    try:
        import webview  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        return False, (
            f'the desktop window is unavailable: {exc}. '
            'Install with: pip install -e ".[desktop]"'
        )

    import sys

    if sys.platform == "darwin":
        return True, "WKWebView (macOS system webview)"
    if sys.platform.startswith("win"):
        return True, "WebView2 (Windows system webview)"
    return True, "GTK/Qt webview (requires the corresponding system library)"


def app_bundle_path(target: Path, name: str = "UnifiedAgent") -> Path:
    return target / f"{name}.app"


__all__ = [
    "DesktopUnavailable",
    "ServerHandle",
    "app_bundle_path",
    "create_window",
    "desktop_available",
    "pick_port",
    "run_desktop",
    "wait_for_health",
]


@contextlib.contextmanager
def _noop():
    yield

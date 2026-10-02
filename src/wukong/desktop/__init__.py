"""Desktop shell: the OS webview wrapped around the same server and console.

`pywebview` is an optional dependency (`pip install -e ".[desktop]"`). The
core runtime does not import this package.
"""

from wukong.desktop.bundle import (  # noqa: F401
    APP_NAME,
    BundleError,
    BundleResult,
    build_app_bundle,
)
from wukong.desktop.launcher import (  # noqa: F401
    DesktopUnavailable,
    ServerHandle,
    desktop_available,
    pick_port,
    run_desktop,
    wait_for_health,
)

__all__ = [
    "APP_NAME",
    "BundleError",
    "BundleResult",
    "DesktopUnavailable",
    "ServerHandle",
    "build_app_bundle",
    "desktop_available",
    "pick_port",
    "run_desktop",
    "wait_for_health",
]

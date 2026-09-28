"""`python -m unified_agent.desktop` -- the entry point the .app bundle runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uaa-desktop",
        description="Open the unified-ai-agent desktop window.",
    )
    parser.add_argument("--home", type=Path, default=None, help="Config directory.")
    parser.add_argument("--workspace", type=Path, default=None, help="Workspace root.")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port.")
    parser.add_argument("--width", type=int, default=1180)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument(
        "--token",
        default=None,
        help="Session token. Generated per launch when omitted.",
    )
    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Disable the token requirement (loopback only; weaker).",
    )
    parser.add_argument("--debug", action="store_true", help="Open webview devtools.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Start and stop the server without a window (used by tests/CI).",
    )
    args = parser.parse_args(argv)

    from unified_agent.config import ConfigError, load_settings
    from unified_agent.desktop.launcher import DesktopUnavailable, run_desktop

    try:
        settings = load_settings(home=args.home, workspace=args.workspace, create_if_missing=True)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    settings.ensure_dirs()
    token = None if args.no_token else (args.token or None)

    try:
        return run_desktop(
            settings=settings,
            port=args.port,
            token=token,
            size=(args.width, args.height),
            debug=args.debug,
            headless_check=args.check,
        )
    except DesktopUnavailable as exc:
        # Printed, not raised: a double-clicked app has no console to show a
        # traceback in, and the log file is the only record the user has.
        print(f"uaa: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

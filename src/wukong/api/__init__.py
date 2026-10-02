"""HTTP service layer.

`fastapi` and `uvicorn` are optional dependencies (`pip install -e ".[api]"`).
The core runtime does not import this package, so a CLI-only install stays
dependency-light.
"""

from wukong.api.agui import AgUiEncoder  # noqa: F401

__all__ = ["AgUiEncoder", "create_app"]


def __getattr__(name: str):  # pragma: no cover - lazy to keep fastapi optional
    if name in {"create_app", "Service"}:
        from wukong.api import app as _app

        return getattr(_app, name)
    raise AttributeError(name)

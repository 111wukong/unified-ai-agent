"""Filesystem tools.

Every one of these resolves its target through the PermissionEngine before
touching disk. They are also careful to *report* rather than guess: a
missing file is a failed ToolResult, not an exception, because the model
needs the failure text to plan around it.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from unified_agent.tools.base import Tool, ToolContext, ToolSpec
from unified_agent.types import EffectClass, ToolResult

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    ".idea",
    ".DS_Store",
}

_MAX_READ_BYTES = 512_000


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _matches_glob(rel: str, name: str, pattern: str) -> bool:
    """Match a path against a glob, making `**/` optional at the root.

    `fnmatch` treats `*` as "any characters including /", so `**/*.py`
    requires a slash and silently misses every top-level file. Users write
    `**/*.py` meaning "all Python files, at any depth" -- so test the
    prefix-stripped form too.
    """
    candidates = {pattern}
    if pattern.startswith("**/"):
        candidates.add(pattern[3:])
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) for p in candidates)


class ReadFileTool(Tool):
    spec = ToolSpec(
        name="read_file",
        description=(
            "Read a UTF-8 text file. Returns content with 1-based line numbers. "
            "Use offset/limit for large files. Binary files are rejected."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, absolute or workspace-relative."},
                "offset": {"type": "integer", "minimum": 1, "description": "First line (1-based)."},
                "limit": {"type": "integer", "minimum": 1, "description": "Max lines to return."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = Path(args["path"])
        if not path.is_absolute():
            path = ctx.workspace / path
        path = Path(os.path.realpath(path))
        if not path.exists():
            return ToolResult(success=False, error=f"no such file: {path}")
        if path.is_dir():
            return ToolResult(
                success=False,
                error=f"{path} is a directory; use list_directory",
            )
        size = path.stat().st_size
        raw = path.read_bytes()[: _MAX_READ_BYTES + 1]
        if b"\x00" in raw[:8192]:
            return ToolResult(success=False, error=f"{path} looks binary; refusing to read")
        text = raw[: _MAX_READ_BYTES].decode("utf-8", errors="replace")
        truncated_bytes = size > _MAX_READ_BYTES

        lines = text.splitlines()
        offset = int(args.get("offset") or 1)
        limit = args.get("limit")
        start = max(offset - 1, 0)
        end = start + int(limit) if limit else len(lines)
        window = lines[start:end]
        numbered = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(window))
        if truncated_bytes:
            numbered += f"\n[... file is {size} bytes; only the first {_MAX_READ_BYTES} were read]"
        if end < len(lines):
            numbered += f"\n[... {len(lines) - end} more lines; pass offset={end + 1} to continue]"
        return ToolResult(
            success=True,
            output=numbered or "(empty file)",
            metadata={
                "path": str(path),
                "lines_total": len(lines),
                "bytes": size,
                "relative": _rel(path, ctx.workspace),
            },
        )


class WriteFileTool(Tool):
    spec = ToolSpec(
        name="write_file",
        description=(
            "Create or overwrite a text file. Parent directories are created. "
            "Overwriting an existing file replaces it entirely — use apply_patch "
            "for targeted edits."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "description": "Default overwrite.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.WRITE_LOCAL,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = Path(args["path"])
        if not path.is_absolute():
            path = ctx.workspace / path
        path = Path(os.path.realpath(path))
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        mode = args.get("mode") or "overwrite"
        existed = path.exists()
        before = path.stat().st_size if existed else 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with path.open("a", encoding="utf-8") as fh:
                fh.write(args["content"])
        else:
            path.write_text(args["content"], encoding="utf-8")
        after = path.stat().st_size
        verb = "appended to" if mode == "append" else ("overwrote" if existed else "created")
        return ToolResult(
            success=True,
            output=f"{verb} {_rel(path, ctx.workspace)} ({before} -> {after} bytes)",
            metadata={"path": str(path), "bytes": after, "existed": existed},
        )


class ApplyPatchTool(Tool):
    spec = ToolSpec(
        name="apply_patch",
        description=(
            "Apply exact-string replacements to a file. Each edit replaces the first "
            "occurrence of `old` with `new` (or every occurrence when replace_all). "
            "`old` must match byte-for-byte including indentation, and must be unique "
            "unless replace_all is set. All edits are validated before anything is written."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old": {"type": "string", "minLength": 1},
                            "new": {"type": "string"},
                            "replace_all": {"type": "boolean"},
                        },
                        "required": ["old", "new"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "edits"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.WRITE_LOCAL,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = Path(args["path"])
        if not path.is_absolute():
            path = ctx.workspace / path
        path = Path(os.path.realpath(path))
        if not path.exists():
            return ToolResult(success=False, error=f"no such file: {path}")
        original = path.read_text(encoding="utf-8")
        text = original
        applied = 0
        for i, edit in enumerate(args["edits"]):
            old, new = edit["old"], edit["new"]
            count = text.count(old)
            if count == 0:
                return ToolResult(
                    success=False,
                    error=(
                        f"edit #{i + 1}: the `old` string was not found in "
                        f"{_rel(path, ctx.workspace)}. Read the file first and copy the "
                        "exact text, including indentation."
                    ),
                )
            if count > 1 and not edit.get("replace_all"):
                return ToolResult(
                    success=False,
                    error=(
                        f"edit #{i + 1}: `old` occurs {count} times; add more surrounding "
                        "context to make it unique, or set replace_all=true"
                    ),
                )
            text = text.replace(old, new) if edit.get("replace_all") else text.replace(old, new, 1)
            applied += 1
        if ctx.dry_run:
            return self.dry_run(args, ctx)
        path.write_text(text, encoding="utf-8")
        return ToolResult(
            success=True,
            output=f"applied {applied} edit(s) to {_rel(path, ctx.workspace)}",
            metadata={"path": str(path), "edits": applied},
        )


class ListDirectoryTool(Tool):
    spec = ToolSpec(
        name="list_directory",
        description=(
            "List directory entries. Noise directories (.git, node_modules, __pycache__, "
            ".venv) are skipped unless include_hidden is true."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Default '.'"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 6},
                "include_hidden": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        base = Path(args.get("path") or ".")
        if not base.is_absolute():
            base = ctx.workspace / base
        base = Path(os.path.realpath(base))
        if not base.exists():
            return ToolResult(success=False, error=f"no such directory: {base}")
        if not base.is_dir():
            return ToolResult(success=False, error=f"{base} is not a directory")
        depth = int(args.get("depth") or 2)
        include_hidden = bool(args.get("include_hidden"))

        lines: list[str] = []
        count = 0
        for root, dirs, files in os.walk(base):
            rel_root = Path(root).relative_to(base)
            level = len(rel_root.parts)
            if level >= depth:
                dirs[:] = []
            if not include_hidden:
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
                files = [f for f in files if not f.startswith(".")]
            indent = "  " * level
            lines.append(f"{indent}{rel_root.name or base.name}/")
            for name in sorted(files):
                size = (Path(root) / name).stat().st_size
                lines.append(f"{indent}  {name}  ({size} B)")
                count += 1
            if count > 2000:
                lines.append("... (truncated at 2000 entries)")
                break
        return ToolResult(
            success=True,
            output="\n".join(lines) or "(empty directory)",
            metadata={"path": str(base), "relative": _rel(base, ctx.workspace)},
        )


class SearchFilesTool(Tool):
    spec = ToolSpec(
        name="search_files",
        description=(
            "Find files by glob pattern and/or search their contents with a regex. "
            "Returns matching paths, and matching lines when `content_regex` is given."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to search. Default '.'"},
                "glob": {"type": "string", "description": "e.g. '**/*.py'"},
                "content_regex": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
                "case_sensitive": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        base = Path(args.get("path") or ".")
        if not base.is_absolute():
            base = ctx.workspace / base
        base = Path(os.path.realpath(base))
        if not base.is_dir():
            return ToolResult(success=False, error=f"not a directory: {base}")
        pattern = args.get("glob") or "**/*"
        content_regex = args.get("content_regex")
        flags = 0 if args.get("case_sensitive") else re.IGNORECASE
        compiled = re.compile(content_regex, flags) if content_regex else None
        max_results = int(args.get("max_results") or 100)

        hits: list[str] = []
        files_scanned = 0
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for name in files:
                if name.startswith(".") and name not in {".env.example"}:
                    continue
                full = Path(root) / name
                rel = full.relative_to(base)
                if not _matches_glob(str(rel), name, pattern):
                    continue
                files_scanned += 1
                if compiled is None:
                    hits.append(str(rel))
                else:
                    try:
                        text = full.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    for lineno, line in enumerate(text.splitlines(), 1):
                        if compiled.search(line):
                            hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                            if len(hits) >= max_results:
                                break
                if len(hits) >= max_results:
                    break
            if len(hits) >= max_results:
                break

        body = "\n".join(hits) if hits else "(no matches)"
        if len(hits) >= max_results:
            body += f"\n[... stopped at max_results={max_results}]"
        return ToolResult(
            success=True,
            output=body,
            metadata={"matches": len(hits), "files_scanned": files_scanned},
        )


class FileInfoTool(Tool):
    spec = ToolSpec(
        name="file_info",
        description="Size, line count, mtime and sha256 of a file, or an entry count for a directory.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        effect_class=EffectClass.READ_ONLY,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = Path(args["path"])
        if not path.is_absolute():
            path = ctx.workspace / path
        path = Path(os.path.realpath(path))
        if not path.exists():
            return ToolResult(success=False, error=f"no such path: {path}")
        stat = path.stat()
        if path.is_dir():
            entries = list(path.iterdir())
            return ToolResult(
                success=True,
                output=f"directory {path}\nentries: {len(entries)}\nmtime: {stat.st_mtime}",
                metadata={"kind": "dir", "entries": len(entries)},
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        text = path.read_text(encoding="utf-8", errors="replace")
        return ToolResult(
            success=True,
            output=(
                f"file {path}\nbytes: {stat.st_size}\nlines: {len(text.splitlines())}\n"
                f"mtime: {stat.st_mtime}\nsha256[:16]: {digest}"
            ),
            metadata={"kind": "file", "bytes": stat.st_size, "sha256_16": digest},
        )


FS_TOOLS: list[Tool] = [
    ReadFileTool(),
    WriteFileTool(),
    ApplyPatchTool(),
    ListDirectoryTool(),
    SearchFilesTool(),
    FileInfoTool(),
]

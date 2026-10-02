---
name: project-summary
description: Summarise an unfamiliar codebase. Use when the user asks what a project does, asks for an overview of a repository, or asks where something is implemented and the answer requires reading the tree.
license: MIT
allowed-tools: read_file list_directory search_files file_info
metadata:
  author: wukong
  version: "1.0"
---

# Project Summary

## Goal

Explain a codebase to someone who has not seen it, in enough depth that
they can find things. Grounded in files you actually read.

## Process

1. List the top two levels. Do not go deeper until you know what the
   entry points are.
2. Read the manifest (`pyproject.toml`, `package.json`, `go.mod`, …). It
   states dependencies, entry points and scripts, and it is never stale
   in the way a README is.
3. Read the README if one exists, but treat it as a claim to verify, not
   as fact.
4. Identify the entry points: where execution starts, where the HTTP
   routes or CLI commands are registered, where the data model lives.
5. Trace one representative path end to end — one request, one command —
   from entry point to side effect. This is what turns a file listing into
   an explanation.
6. Note what is *absent*: no tests, no migrations, no error handling.
   Absence is usually the most useful thing you can report.

## Output

- One paragraph: what the project is for.
- Layout: the directories that matter and what each owns.
- Entry points: exact file paths.
- The traced path, as a short ordered list.
- Gaps and surprises.

## Rules

- Cite real paths. Never invent a directory or file name.
- If a directory is large, say how many files rather than listing them all.
- Do not describe what you did not read. Mark inferences as inferences.

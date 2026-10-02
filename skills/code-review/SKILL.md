---
name: code-review
description: Review a code change and report findings by severity. Use when the user asks to review a diff, a pull request, or a specific file's changes, or asks whether a change is safe to merge.
license: MIT
allowed-tools: read_file list_directory search_files git_status git_diff run_tests
metadata:
  author: wukong
  version: "1.0"
---

# Code Review

## Goal

Produce findings a developer can act on. A review that only says "looks
good" is a failed review; so is a review that lists style preferences as
if they were defects.

## Process

1. Establish what actually changed. Use `git_status` and `git_diff` — do
   not review from the description alone, which is frequently stale.
2. Read the full context of each changed hunk, not just the diff. A
   changed line is only correct or incorrect relative to the function
   around it.
3. Check, in this order:
   - **Correctness** — off-by-one, inverted conditions, unhandled `None`,
     error paths that swallow exceptions, resource leaks.
   - **Security** — unvalidated input reaching a sink, secrets in logs or
     source, path traversal, injection, missing authorization checks.
   - **Concurrency** — shared mutable state, missing locks, `await` inside
     a critical section, timeouts that do not exist.
   - **Tests** — is the new behaviour covered? Does an existing test now
     pass for the wrong reason?
   - **Maintainability** — only when it obscures a real defect. Do not
     report formatting.
4. Report each finding as: severity, file and line, what is wrong, why it
   matters, and the smallest fix.

## Severity

- `blocker` — wrong result, data loss, or a security hole.
- `major` — breaks in a realistic case, or is unmaintainable enough to
  cause future defects.
- `minor` — real but low impact.
- `nit` — preference. Say so explicitly.

## Rules

- Do not modify files unless explicitly asked to fix.
- Do not report a finding you have not read the surrounding code for.
- If you cannot determine whether something is a bug, say that instead of
  guessing.
- Report what you did *not* review. Silence implies coverage.

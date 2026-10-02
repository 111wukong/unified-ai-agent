"""Variable resolution and condition evaluation.

Two functions, and one rule that decides the whole design:

**An unresolvable reference is an error, not a falsey value.**

If `{{#analyse.count#}}` cannot be resolved, `evaluate` raises rather than
returning `False`. A branch that quietly never fires is indistinguishable
from a branch whose condition is genuinely false, and the workflow then
appears to work while skipping the path the author cared about. Failing
loudly is the only option that surfaces it.

The condition language is deliberately small -- comparisons, `contains`, and
`and`/`or` -- and is parsed rather than `eval`'d. A workflow file is data
that may come from anywhere; handing it `eval` would make "load this YAML" a
code-execution primitive.
"""

from __future__ import annotations

import re
from typing import Any

from wukong.orchestration.workflow import REFERENCE

COMPARISONS = ("==", "!=", ">=", "<=", ">", "<")
WORD_OPS = ("contains", "startswith", "endswith", "matches")


class ExpressionError(ValueError):
    """A reference did not resolve, or a condition could not be parsed."""


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def is_single_reference(text: str) -> bool:
    match = REFERENCE.fullmatch(text.strip())
    return match is not None


def resolve(template: str, context: dict[str, dict[str, Any]]) -> Any:
    """Substitute `{{#node.field#}}` in `template`.

    A template that is exactly one reference returns the *raw* value, so
    `goal: "{{#start.count#}}"` passes an int through as an int. Anything else
    is interpolated into a string.
    """
    stripped = template.strip()
    if is_single_reference(stripped):
        return _lookup(stripped, context)

    def replace(match: re.Match[str]) -> str:
        value = _lookup(match.group(0), context)
        return _stringify(value)

    return REFERENCE.sub(replace, template)


def resolve_deep(value: Any, context: dict[str, dict[str, Any]]) -> Any:
    """Resolve every reference in a nested structure, preserving shape."""
    if isinstance(value, str):
        return resolve(value, context)
    if isinstance(value, dict):
        return {key: resolve_deep(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_deep(item, context) for item in value]
    return value


def _lookup(reference: str, context: dict[str, dict[str, Any]]) -> Any:
    match = REFERENCE.fullmatch(reference.strip())
    if match is None:  # pragma: no cover - callers only pass matched text
        raise ExpressionError(f"not a reference: {reference!r}")
    node_id, field_name = match.group("node"), match.group("field")

    if node_id not in context:
        raise ExpressionError(
            f"reference to node {node_id!r} has no value yet; "
            f"available: {', '.join(sorted(context)) or '(none)'}"
        )
    outputs = context[node_id]
    if field_name in outputs:
        return outputs[field_name]
    # A single dotted segment is a field; a longer path indexes into it.
    head, _, rest = field_name.partition(".")
    if head in outputs and rest:
        return _index(outputs[head], rest, reference)
    raise ExpressionError(
        f"node {node_id!r} has no field {field_name!r}; "
        f"it provides: {', '.join(sorted(outputs)) or '(none)'}"
    )


def _index(value: Any, path: str, reference: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ExpressionError(f"{reference} does not resolve: no {part!r} in {type(current).__name__}")
    return current


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------

_SPLIT_OR = re.compile(r"\s+or\s+")
_SPLIT_AND = re.compile(r"\s+and\s+")
# Longest operators first, so `>=` is not read as `>`.
_OPERATOR = re.compile(
    r"\s+(?P<op>" + "|".join(re.escape(op) for op in (*WORD_OPS, *COMPARISONS)) + r")\s+"
)
_LITERAL = re.compile(r"^(?P<quote>['\"])(?P<text>.*)(?P=quote)$")


def evaluate(condition: str, context: dict[str, dict[str, Any]]) -> bool:
    """Evaluate a condition. Raises `ExpressionError` if it cannot be resolved."""
    text = condition.strip()
    if not text:
        raise ExpressionError("empty condition")

    # `or` binds loosest, so it splits first.
    if parts := _SPLIT_OR.split(text):
        if len(parts) > 1:
            return any(evaluate(part, context) for part in parts)
    if parts := _SPLIT_AND.split(text):
        if len(parts) > 1:
            return all(evaluate(part, context) for part in parts)

    return _evaluate_clause(text, context)


def _evaluate_clause(text: str, context: dict[str, dict[str, Any]]) -> bool:
    text = text.strip()
    # Bare `true` / `false` / a reference used as a boolean.
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"

    match = _OPERATOR.search(text)
    if match is None:
        # No operator: truthiness of a single operand.
        return _truthy(_operand(text, context))

    left = _operand(text[: match.start()], context)
    right = _operand(text[match.end() :], context)
    return _compare(left, match.group("op"), right)


def _operand(token: str, context: dict[str, dict[str, Any]]) -> Any:
    text = token.strip()
    if not text:
        raise ExpressionError("missing operand in condition")
    if is_single_reference(text):
        return _lookup(text, context)
    if REFERENCE.search(text):
        return resolve(text, context)
    if literal := _LITERAL.match(text):
        return literal.group("text")
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if number := _as_number(text):
        return number
    # A bare word is a string, so `status == ok` reads naturally.
    return text


def _as_number(text: str) -> int | float | None:
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "false", "0", "no", "null", "none"}:
            return False
        return True
    return bool(value)


def _compare(left: Any, op: str, right: Any) -> bool:
    if op in WORD_OPS:
        return _string_compare(_stringify(left), op, _stringify(right))

    if op in ("==", "!="):
        # Numeric equality when both sides are numeric, so `"3" == 3` holds
        # without the author having to know which side arrived as a string.
        left_number, right_number = _as_number(_stringify(left)), _as_number(_stringify(right))
        if left_number is not None and right_number is not None:
            equal = left_number == right_number
        elif isinstance(left, bool) or isinstance(right, bool):
            equal = _truthy(left) is _truthy(right) if isinstance(right, bool) else left == right
        else:
            equal = left == right
        return equal if op == "==" else not equal

    left_number = _as_number(_stringify(left))
    right_number = _as_number(_stringify(right))
    if left_number is None or right_number is None:
        # Ordered comparison of non-numbers falls back to string ordering,
        # which is what a user comparing two version strings expects.
        left_value, right_value = _stringify(left), _stringify(right)
    else:
        left_value, right_value = left_number, right_number

    if op == ">":
        return left_value > right_value
    if op == ">=":
        return left_value >= right_value
    if op == "<":
        return left_value < right_value
    if op == "<=":
        return left_value <= right_value
    raise ExpressionError(f"unsupported operator {op!r}")


def _string_compare(left: str, op: str, right: str) -> bool:
    if op == "contains":
        return right in left
    if op == "startswith":
        return left.startswith(right)
    if op == "endswith":
        return left.endswith(right)
    if op == "matches":
        try:
            return re.search(right, left) is not None
        except re.error as exc:
            raise ExpressionError(f"invalid pattern {right!r}: {exc}") from exc
    raise ExpressionError(f"unsupported operator {op!r}")


def references_in(condition: str) -> list[str]:
    return [match.group(0) for match in REFERENCE.finditer(condition)]


__all__ = [
    "COMPARISONS",
    "WORD_OPS",
    "ExpressionError",
    "evaluate",
    "is_single_reference",
    "references_in",
    "resolve",
    "resolve_deep",
]

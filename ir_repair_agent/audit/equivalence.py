"""Strict LLM-assisted equivalence review for reported constraint differences."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


EQUIVALENT_CATEGORIES = {
    "VARIABLE_RENAME",
    "INDEX_SHIFT",
    "FOLDED_INITIAL",
    "AGGREGATED_VS_PER_PERIOD",
    "DOMAIN_VS_BOUND",
    "AUXILIARY_VARIABLE",
    "REDUNDANT_CONSTRAINT",
    "VALID_BIG_M",
    "ALGEBRAIC_REARRANGEMENT",
}

CATEGORIES_REQUIRING_MAPPING = {
    "VARIABLE_RENAME",
    "INDEX_SHIFT",
    "FOLDED_INITIAL",
    "VALID_BIG_M",
    "ALGEBRAIC_REARRANGEMENT",
}


def _excerpt(value: Any, limit: int = 6000) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)[:limit]
    except Exception:
        return str(value)[:limit]


def build_equivalence_prompt(
    problem_text: str,
    code: str,
    expected_constraints: list[dict[str, Any]],
    implemented_constraints: list[dict[str, Any]],
    differences: list[dict[str, Any]],
) -> str:
    return f"""You are an equivalence auditor for Gurobi optimization models.
The main constraint auditor found differences between the problem requirements
and the implemented code. Your job is NOT to re-audit from scratch, but to
decide whether each reported difference is a TRUE_ERROR or an EQUIVALENT
formulation.

Allowed EQUIVALENT categories:
- VARIABLE_RENAME: same role, different variable names.
- INDEX_SHIFT: 0-based vs 1-based indexing, same periods.
- FOLDED_INITIAL: an initial inventory/state is folded into the first balance RHS.
- AGGREGATED_VS_PER_PERIOD: a "for all t" family is written as per-period rows, or vice versa.
- DOMAIN_VS_BOUND: a variable domain is split into addVar bounds plus explicit bound constraints.
- AUXILIARY_VARIABLE: an extra variable/constraint only links or renames an existing quantity.
- REDUNDANT_CONSTRAINT: an extra constraint is implied by the others and does not change the feasible set.
- VALID_BIG_M: a symbolic M is replaced by a concrete valid upper bound.
- ALGEBRAIC_REARRANGEMENT: algebraically equivalent rearrangement of the same expression.

Rules:
- Do NOT assume equivalence. Give a concrete mapping/substitution or reason.
- A MISSING constraint is EQUIVALENT only if it is actually encoded elsewhere.
- An EXTRA constraint is EQUIVALENT only if it does not change the feasible set.
- A MISMATCH is EQUIVALENT only with a concrete mapping or algebraic rearrangement.
- A big-M is EQUIVALENT only if the concrete constant is a valid upper bound.
- If the objective direction or a core constraint is genuinely wrong, return TRUE_ERROR.
- For VARIABLE_RENAME, INDEX_SHIFT, FOLDED_INITIAL, VALID_BIG_M and
  ALGEBRAIC_REARRANGEMENT, the "mapping" field is REQUIRED.

Return JSON only:
{{"all_equivalent": true|false,
  "confidence": 0.0-1.0,
  "items": [
    {{"difference_index": 0,
      "decision": "EQUIVALENT|TRUE_ERROR",
      "category": "ONE_OF_THE_ALLOWED_CATEGORIES",
      "reason": "specific justification",
      "mapping": "required for mapping-based categories, otherwise optional"}}
  ]}}

<PROBLEM>{problem_text[:3000]}</PROBLEM>
<CODE>{code[:6000]}</CODE>
<EXPECTED>{_excerpt(expected_constraints)}</EXPECTED>
<IMPLEMENTED>{_excerpt(implemented_constraints)}</IMPLEMENTED>
<DIFFERENCES>{_excerpt(differences, 8000)}</DIFFERENCES>"""


def verify_equivalence(
    problem_text: str,
    code: str,
    expected_constraints: list[dict[str, Any]],
    implemented_constraints: list[dict[str, Any]],
    differences: list[dict[str, Any]],
    ask_json_retry: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    if not differences:
        return {
            "all_equivalent": False,
            "confidence": 0.0,
            "items": [],
            "error": "no differences to verify",
        }
    prompt = build_equivalence_prompt(
        problem_text,
        code,
        expected_constraints,
        implemented_constraints,
        differences,
    )
    return ask_json_retry(prompt)


def equivalence_ok(
    result: dict[str, Any],
    differences: list[dict[str, Any]],
) -> bool:
    if not isinstance(result, dict) or result.get("all_equivalent") is not True:
        return False
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        return False
    if confidence < 0.8:
        return False
    items = result.get("items")
    if not isinstance(items, list) or len(items) != len(differences):
        return False
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            return False
        index = item.get("difference_index")
        if (
            not isinstance(index, int)
            or index < 0
            or index >= len(differences)
            or index in seen
        ):
            return False
        seen.add(index)
        if str(item.get("decision", "")).upper() != "EQUIVALENT":
            return False
        category = str(item.get("category", "")).upper()
        if category not in EQUIVALENT_CATEGORIES:
            return False
        if not str(item.get("reason", "")).strip():
            return False
        if (
            category in CATEGORIES_REQUIRING_MAPPING
            and not str(item.get("mapping", "")).strip()
        ):
            return False
    return len(seen) == len(differences)


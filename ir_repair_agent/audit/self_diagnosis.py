"""Consensus-based self diagnosis for unseen problem classes."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "for", "to", "in", "on",
    "and", "or", "not", "no", "all", "each", "every", "must",
    "should", "constraint", "constraints", "variable", "variables",
}


def _keywords(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    return words - STOPWORDS


def consensus_missing_constraints(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep TYPE_A missing constraints supported across every valid diagnosis."""
    if len(results) < 2:
        return []
    diagnoses: list[list[tuple[set[str], dict[str, Any]]]] = []
    for result in results:
        items: list[tuple[set[str], dict[str, Any]]] = []
        for constraint in result.get("constraints", []):
            if (
                constraint.get("type") == "TYPE_A"
                and not constraint.get("in_code", True)
            ):
                text = f"{constraint.get('name', '')} {constraint.get('formula', '')}"
                items.append((_keywords(text), constraint))
        diagnoses.append(items)
    if not diagnoses or not diagnoses[0]:
        return []
    consensus: list[dict[str, Any]] = []
    for first_keywords, first_constraint in diagnoses[0]:
        if all(
            any(first_keywords & keywords for keywords, _ in diagnosis)
            for diagnosis in diagnoses[1:]
        ):
            consensus.append(first_constraint)
    return consensus


def build_self_diagnosis_hint(
    results: list[dict[str, Any]],
    missing_constraints: list[dict[str, Any]],
) -> str:
    if not missing_constraints:
        return ""
    problem_type = results[0].get("problem_type", "?") if results else "?"
    lines = [
        f"⚠️ SELF-DIAGNOSIS (type={problem_type}):",
        "The following REQUIRED constraints appear to be MISSING:",
        f"(Verified by {len(results)}/3 independent diagnoses)",
    ]
    for constraint in missing_constraints:
        lines.append(
            f"  - {constraint.get('name', '?')}: {constraint.get('formula', '')}"
        )
        if constraint.get("reason"):
            lines.append(f"    reason: {constraint['reason']}")
    return "\n".join(lines)


def self_diagnose(
    problem_text: str,
    code: str,
    ask_json: Callable[[str], dict[str, Any]],
    runs: int = 3,
) -> str:
    prompt = f"""You are an OR expert. Analyze the optimization problem and its generated code.

<PROBLEM>
{problem_text[:1500]}
</PROBLEM>

<CODE>
{code[:2000]}
</CODE>

Task:
1. Identify the problem type.
2. List 3-5 KEY constraints that ANY correct model MUST have.
3. Classify each as TYPE_A (mathematically required) or TYPE_B (conditional).
4. Check whether the code implements each constraint.

Return JSON only:
{{"problem_type": "...",
  "constraints": [
    {{"name": "...", "formula": "...", "type": "TYPE_A|TYPE_B",
      "in_code": true|false, "reason": "..."}}
  ]}}"""
    results = []
    for _ in range(max(1, int(runs))):
        result = ask_json(prompt)
        if "error" not in result:
            results.append(result)
    if len(results) < 2:
        return ""
    missing = consensus_missing_constraints(results)
    return build_self_diagnosis_hint(results, missing)


def verify_self_diagnosis(
    problem_text: str,
    code: str,
    hint: str,
    ask_json: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    prompt = f"""A hypothesis has been raised about missing constraints.
Verify each constraint independently. Do NOT assume the hypothesis is correct.

HYPOTHESIS:
{hint}

<PROBLEM>
{problem_text[:1500]}
</PROBLEM>

<CODE>
{code[:2000]}
</CODE>

For each constraint in the hypothesis, decide whether it is really required
and really missing. If any item fails, set all_valid=false.

Return JSON only:
{{"all_valid": true|false,
  "reason": "brief explanation",
  "invalid_items": ["..."]}}"""
    result = ask_json(prompt)
    if "error" in result:
        return {"all_valid": False, "reason": "verification failed"}
    return result


"""Program-assisted objective-variable witness checks."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


_SKIP_NAMES = {
    "abs", "all", "any", "float", "int", "len", "max", "min", "range",
    "sum", "quicksum", "gp", "GRB",
}
_SKIP_CALLS = {
    "abs", "float", "gp.quicksum", "int", "len", "max", "min", "range",
    "sum",
}


@dataclass(frozen=True)
class ObjectiveWitness:
    verdict: str
    expected_objective: str
    required_symbols: tuple[str, ...]
    forbidden_symbols: tuple[str, ...]
    used_symbols: tuple[str, ...]
    objective_text: str
    reason: str
    confidence: float

    @property
    def hard_fail(self) -> bool:
        return self.verdict == "FAIL"

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "expected_objective": self.expected_objective,
            "required_symbols": list(self.required_symbols),
            "forbidden_symbols": list(self.forbidden_symbols),
            "used_symbols": list(self.used_symbols),
            "objective_text": self.objective_text[:1000],
            "reason": self.reason,
            "confidence": self.confidence,
        }


def _collect_symbols(node: ast.AST, symbols: set[str]) -> None:
    if isinstance(node, ast.Name):
        if node.id not in _SKIP_NAMES:
            symbols.add(node.id)
        return
    if isinstance(node, ast.Subscript):
        _collect_symbols(node.value, symbols)
        _collect_symbols(node.slice, symbols)
        return
    if isinstance(node, ast.Attribute):
        _collect_symbols(node.value, symbols)
        return
    if isinstance(node, ast.Call):
        function_name = ""
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            try:
                function_name = ast.unparse(node.func)
            except Exception:
                function_name = ""
        if function_name not in _SKIP_CALLS:
            _collect_symbols(node.func, symbols)
        for argument in node.args:
            _collect_symbols(argument, symbols)
        for keyword in node.keywords:
            _collect_symbols(keyword.value, symbols)
        return
    for child in ast.iter_child_nodes(node):
        _collect_symbols(child, symbols)


def extract_objective_witness(code: str) -> tuple[str, tuple[str, ...]]:
    """Return objective text and symbol families referenced by setObjective."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "", ()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 1:
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "setObjective":
            continue
        expression = node.args[0]
        symbols: set[str] = set()
        _collect_symbols(expression, symbols)
        try:
            objective_text = ast.unparse(expression)
        except Exception:
            objective_text = ""
        return objective_text, tuple(sorted(symbols))
    return "", ()


def verify_objective_witness(
    problem_text: str,
    code: str,
    ask_json: Callable[[str], dict[str, Any]],
) -> ObjectiveWitness:
    objective_text, used_symbols = extract_objective_witness(code)
    result = ask_json(f"""Verify whether the reported Gurobi objective uses the
correct decision-variable family for the problem.

Return JSON only:
{{"verdict":"PASS|FAIL|REVIEW",
  "expected_objective":"brief mathematical description",
  "required_symbols":["symbol names that must appear in the objective"],
  "forbidden_symbols":["symbol families that clearly must not drive this objective"],
  "reason":"brief",
  "confidence":0.0-1.0}}

Use required_symbols only for code-level variable families you can identify
from the provided code. If uncertain, use REVIEW.

<PROBLEM>
{problem_text[:8000]}
</PROBLEM>

<CODE>
{code[:12000]}
</CODE>

<EXTRACTED_OBJECTIVE>
{objective_text[:3000]}
</OBJECTIVE>

<USED_SYMBOLS>
{", ".join(used_symbols)}
</USED_SYMBOLS>""")
    if "error" in result:
        return ObjectiveWitness(
            verdict="REVIEW",
            expected_objective="",
            required_symbols=(),
            forbidden_symbols=(),
            used_symbols=used_symbols,
            objective_text=objective_text,
            reason=str(result.get("error", "witness failed"))[:300],
            confidence=0.0,
        )

    verdict = str(result.get("verdict", "REVIEW")).upper()
    if verdict not in {"PASS", "FAIL", "REVIEW"}:
        verdict = "REVIEW"
    required = tuple(
        str(item).strip()
        for item in result.get("required_symbols", [])
        if str(item).strip()
    )
    forbidden = tuple(
        str(item).strip()
        for item in result.get("forbidden_symbols", [])
        if str(item).strip()
    )
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    used = set(used_symbols)
    if any(symbol in used for symbol in forbidden):
        verdict = "FAIL"
    elif required and not set(required).issubset(used):
        verdict = "FAIL"
    elif verdict == "PASS" and not required:
        verdict = "REVIEW"

    return ObjectiveWitness(
        verdict=verdict,
        expected_objective=str(result.get("expected_objective", ""))[:800],
        required_symbols=required,
        forbidden_symbols=forbidden,
        used_symbols=used_symbols,
        objective_text=objective_text,
        reason=str(result.get("reason", ""))[:500],
        confidence=confidence,
    )


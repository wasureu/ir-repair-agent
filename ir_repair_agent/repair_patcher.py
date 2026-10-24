"""Prompt construction and candidate patching for LLM repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm.parsing import extract_python_code


@dataclass(frozen=True)
class RepairPromptContext:
    original_code: str
    audit_result: str
    problem_text: str
    execution_error: str = ""
    problem_class: str = "unknown"
    hint: str = ""
    historical_cases: tuple[dict[str, Any], ...] = ()


def build_hint_block(problem_class: str, hint: str) -> str:
    if not hint:
        return ""
    return f"""
PROBLEM-TYPE REFERENCE ({problem_class}):
NOTE: The following is a REFERENCE LIST, not a mandate. Only apply items
that are ACTUALLY REQUIRED by the current problem.

{hint}
"""


def build_memory_block(
    historical_cases: tuple[dict[str, Any], ...],
    *,
    enabled: bool = True,
) -> str:
    if not enabled or not historical_cases:
        return ""
    case_lines = []
    for index, case in enumerate(historical_cases[:3], 1):
        case_lines.append(
            f"--- Case {index} ---\n"
            f"error pattern: {(case.get('error_pattern') or '')[:300]}\n"
            f"before (fragment):\n{(case.get('original_snippet') or '')[:500]}\n"
            f"after (fragment):\n{(case.get('repaired_snippet') or '')[:500]}"
        )
    return (
        "\nHISTORICAL REPAIR CASES (same problem class, most similar first).\n"
        "Treat as reference only; do NOT copy verbatim. Apply the same fix "
        "strategy only where the current audit shows the same error pattern.\n"
        + "\n".join(case_lines)
        + "\n"
    )


def build_error_section(execution_error: str) -> str:
    if not execution_error:
        return ""
    model_too_large = "Model too large" in execution_error
    compact_warning = (
        "\nThe previous candidate exceeded the model-size limit. "
        "Reformulate with compact direct variables and constraints. "
        "Do not enumerate Cartesian-product configuration options.\n"
        if model_too_large
        else ""
    )
    return f"""
<EXECUTION_ERROR>
The previous repair attempt passed constraint audit but FAILED at runtime.
Fix the runtime error before returning code.
{compact_warning}
Traceback / Error:
{execution_error[:2000]}
</EXECUTION_ERROR>
"""


def build_repair_prompt(
    context: RepairPromptContext,
    *,
    memory_enabled: bool = True,
) -> str:
    hint_block = build_hint_block(context.problem_class, context.hint)
    memory_block = build_memory_block(
        context.historical_cases,
        enabled=memory_enabled,
    )
    error_section = build_error_section(context.execution_error)
    return f"""You are a senior Gurobi engineer. Repair the candidate code for the CURRENT problem.

**STRICT REQUIREMENTS**:
1. Fix the reported error (constraint mismatch OR runtime error).
2. Preserve the problem type and core decision variables. Add auxiliary
   variables or change the objective only when supported by the audit.
3. Do not change unrelated variable domains. Preserve integer/continuous types,
   bounds, and constraints unless the audit explicitly identifies them as wrong.
4. Prefer compact formulations. Do not enumerate large Cartesian-product
   option sets when direct decision variables and constraints are possible.
5. If the problem is lexicographic or has goal priorities, preserve the stated
   priority order with sequential optimization or priority constraints. Do not
   replace it with one weighted or single objective unless the problem says so.
6. Return complete executable Python code only, wrapped in a python code fence.
7. Before returning, verify all names, parentheses, variable domains, and
   Gurobi API calls.
{hint_block}
{memory_block}
<CURRENT_PROBLEM>{context.problem_text}</CURRENT_PROBLEM>
<AUDIT>{context.audit_result}</AUDIT>
<CANDIDATE>{context.original_code}</CANDIDATE>
{error_section}"""


def repair_code(
    original_code: str,
    audit_result: str,
    problem_text: str,
    call_llm_func,
    execution_error: str = "",
    problem_class: str = "unknown",
    hint: str = "",
    historical_cases: list[dict[str, Any]] | None = None,
    memory_enabled: bool = True,
) -> str:
    context = RepairPromptContext(
        original_code=original_code,
        audit_result=audit_result,
        problem_text=problem_text,
        execution_error=execution_error,
        problem_class=problem_class,
        hint=hint,
        historical_cases=tuple(historical_cases or ()),
    )
    prompt = build_repair_prompt(context, memory_enabled=memory_enabled)
    response = str(call_llm_func(prompt))
    return extract_python_code(response) or response.strip()


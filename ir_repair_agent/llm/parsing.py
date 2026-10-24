"""Parsing helpers for LLM responses."""

from __future__ import annotations

import re


def extract_python_code(response: str) -> str:
    """Extract a Python code block, preserving the legacy fallback behavior."""
    response = response or ""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```python\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    code_lines: list[str] = []
    in_code = False
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.startswith("import gurobipy") or stripped.startswith("from gurobipy"):
            in_code = True
        if in_code:
            code_lines.append(line)
    return "\n".join(code_lines) if code_lines else ""


def extract_numeric_answer(response: str) -> str | None:
    """Extract the first explicit answer or objective value."""
    response = response or ""
    match = re.search(
        r"<answer>\s*([\d.eE+\-]+)\s*</answer>",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1)
    match = re.search(
        r"(?:Optimal value|ObjVal)[\s:]+([\d.eE+\-]+)",
        response,
        re.IGNORECASE,
    )
    return match.group(1) if match else None

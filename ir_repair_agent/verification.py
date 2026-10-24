"""Solver-evidence regression checks."""

from __future__ import annotations

from typing import Any


def regression_issues(
    previous_evidence: dict[str, Any],
    candidate_evidence: dict[str, Any],
) -> dict[str, Any]:
    severe: list[str] = []
    warnings: list[str] = []

    prev_status = previous_evidence.get("status_name")
    cand_status = candidate_evidence.get("status_name")
    if prev_status == "OPTIMAL" and cand_status not in (None, "OPTIMAL"):
        severe.append(f"solver_status_regressed:{prev_status}->{cand_status}")

    prev_vio = previous_evidence.get("max_constr_vio")
    cand_vio = candidate_evidence.get("max_constr_vio")
    if isinstance(prev_vio, (int, float)) and isinstance(cand_vio, (int, float)):
        if float(cand_vio) > float(prev_vio) + 1e-6:
            severe.append(
                f"constraint_violation_increased:{prev_vio}->{cand_vio}"
            )

    previous_vars = {
        item.get("name"): item.get("vtype")
        for item in previous_evidence.get("variables", [])
        if item.get("name")
    }
    candidate_vars = {
        item.get("name"): item.get("vtype")
        for item in candidate_evidence.get("variables", [])
        if item.get("name")
    }
    domain_regressions = [
        name for name, vtype in previous_vars.items()
        if vtype == "I" and candidate_vars.get(name) == "C"
    ]
    if domain_regressions:
        severe.append(
            "integer_to_continuous:" + ",".join(domain_regressions[:20])
        )

    previous_constraints = {
        item.get("name")
        for item in previous_evidence.get("constraints", [])
        if item.get("name")
    }
    candidate_constraints = {
        item.get("name")
        for item in candidate_evidence.get("constraints", [])
        if item.get("name")
    }
    disappeared = sorted(previous_constraints - candidate_constraints)
    if (
        len(previous_constraints) >= 5
        and len(disappeared) >= max(2, int(0.2 * len(previous_constraints)))
        and len(candidate_constraints) < len(previous_constraints)
    ):
        warnings.append(
            "constraint_names_disappeared:" + ",".join(disappeared[:20])
        )

    return {
        "ok": not severe,
        "severe": severe,
        "warnings": warnings,
        "disappeared_constraint_count": len(disappeared),
    }

"""Pure acceptance policies for candidate repair patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


GateAction = Literal["accept", "reject", "provisional_accept"]


@dataclass(frozen=True)
class GateDecision:
    action: GateAction
    reason: str
    evidence: dict[str, Any]


def objective_improved(
    direction: str,
    current: float,
    candidate: float,
    tolerance: float,
) -> bool | None:
    direction = (direction or "unknown").lower()
    scale = max(abs(float(current)), 1.0)
    threshold = max(float(tolerance), 1e-9) * scale
    if direction == "min":
        return float(candidate) < float(current) - threshold
    if direction == "max":
        return float(candidate) > float(current) + threshold
    return None


def objective_nochange(
    current: float,
    candidate: float,
    tolerance: float,
) -> bool:
    scale = max(abs(float(current)), 1.0)
    threshold = max(float(tolerance), 1e-9) * scale
    return abs(float(candidate) - float(current)) <= threshold


def assess_candidate(
    *,
    previous_exec_success: bool,
    previous_objective: float | None,
    candidate_objective: float,
    direction_evidence: dict[str, Any],
    regression: dict[str, Any] | None = None,
    tolerance: float = 1e-3,
    oracle_errors: tuple[float, float] | None = None,
) -> GateDecision:
    evidence = {
        "previous_objective": previous_objective,
        "candidate_objective": candidate_objective,
        "direction_evidence": direction_evidence,
        "regression": regression,
        "oracle_errors": oracle_errors,
    }
    if regression is not None and not regression.get("ok", True):
        return GateDecision(
            "reject",
            "solver-level regression detected",
            evidence,
        )
    if not previous_exec_success or previous_objective is None:
        return GateDecision(
            "accept",
            "candidate executes where previous code failed",
            evidence,
        )

    if oracle_errors is not None:
        current_error, candidate_error = oracle_errors
        if candidate_error < current_error:
            return GateDecision(
                "accept",
                "oracle-gated: objective improved",
                evidence,
            )
        return GateDecision(
            "reject",
            "oracle-gated: objective not improved",
            evidence,
        )

    direction = direction_evidence.get("direction", "unknown")
    confidence = direction_evidence.get("confidence", "unknown")
    if confidence == "conflict":
        return GateDecision(
            "reject",
            "conflicting objective direction evidence",
            evidence,
        )

    improved = (
        objective_improved(
            direction,
            previous_objective,
            candidate_objective,
            tolerance,
        )
        if confidence in {"high", "medium"}
        else None
    )
    if improved is True:
        return GateDecision(
            "accept",
            f"objective improved for {direction}",
            evidence,
        )
    if improved is False:
        if objective_nochange(
            previous_objective,
            candidate_objective,
            tolerance,
        ):
            return GateDecision(
                "provisional_accept",
                "objective unchanged within tolerance",
                evidence,
            )
        return GateDecision(
            "reject",
            f"objective not improved for {direction}",
            evidence,
        )
    return GateDecision(
        "provisional_accept",
        "execution evidence only; objective direction evidence insufficient",
        evidence,
    )


def select_final_fallback(
    *,
    current_code: str,
    last_verified_code: str | None,
    execution_baseline_code: str | None,
    final_audit: str,
    oracle_gate: bool,
) -> tuple[str | None, str]:
    # Only a strict PASS is a terminal post-loop audit result. PASS_EQUIVALENT
    # still requires the oracle-free acceptance checks and must not bypass
    # the safe baseline fallback.
    if oracle_gate or final_audit.strip() == "PASS":
        return None, ""
    if (
        last_verified_code is not None
        and last_verified_code.strip() != current_code.strip()
    ):
        return last_verified_code, "last_verified"
    if (
        execution_baseline_code is not None
        and execution_baseline_code.strip() != current_code.strip()
    ):
        return execution_baseline_code, "execution_baseline"
    return None, ""


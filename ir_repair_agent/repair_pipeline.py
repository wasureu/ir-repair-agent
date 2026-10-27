"""Oracle-independent audit-repair loop with verification gates."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

from .repair_gate import GateDecision, assess_candidate, select_final_fallback
from .verification import regression_issues


@dataclass
class RepairPipelineDependencies:
    audit_logic: Callable[..., str]
    repair_code: Callable[..., str]
    execute_code: Callable[[str], Any] | None
    invoke_executor: Callable[[Callable[[str], Any] | None, str], tuple[bool, float | None, str]]
    solver_evidence: Callable[[str], dict[str, Any]]
    problem_direction: Callable[[str], dict[str, Any]]
    fuse_direction: Callable[[dict[str, Any], str, str], dict[str, Any]]
    independent_verify: Callable[[str, str], tuple[str, str]]
    objective_witness: Callable[[str, str], Any]
    log_gate: Callable[..., None]
    memory_error_pattern: Callable[[str], str]
    hint_cache: MutableMapping[str, str]
    normalize_class: Callable[[str], str]
    detect_difficulty: Callable[[str, int], str]
    healing_rounds: Callable[[str, int], int]
    valid_python: Callable[[str], bool]
    attach_error_context: Callable[[str, str], str]
    code_direction: Callable[[str], str]
    return_pass: Callable[..., tuple[str, str, int, str]]
    store_exec_success: Callable[..., None]
    memory_top_score: Callable[[Any, str, list[dict[str, Any]]], float]
    log_memory: Callable[..., None]
    memory_log_enabled: bool
    oracle_gate: bool = False
    assess_candidate: Callable[..., GateDecision] = assess_candidate
    select_final_fallback: Callable[..., tuple[str | None, str]] = select_final_fallback
    regression_check: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] = (
        lambda previous, candidate: regression_issues(previous, candidate)
    )


@dataclass(frozen=True)
class RepairPipelineRequest:
    problem_text: str
    initial_code: str
    initial_modeling_logic: str | None
    exec_output: str
    problem_class: str
    size: int
    question_id: str = ""
    initial_execution_error: str = ""
    ground_truth: float | None = None
    objective_tolerance: float = 1e-6
    max_rounds_override: int | None = None
    memory: Any = None


class RepairPipeline:
    def __init__(self, dependencies: RepairPipelineDependencies):
        self.deps = dependencies

    def _solver_summary(self, code: str) -> dict[str, Any]:
        try:
            evidence = self.deps.solver_evidence(code) or {}
        except Exception:
            return {}
        keys = (
            "status_name", "objective", "objective_sense", "sol_count",
            "max_constr_vio", "sum_constr_vio", "infeasible_constraints", "iis",
        )
        return {key: evidence.get(key) for key in keys}

    def run(self, request: RepairPipelineRequest) -> tuple[str, str, int, str]:
        deps = self.deps
        difficulty = deps.detect_difficulty(request.problem_class, request.size)
        max_rounds = (
            request.max_rounds_override
            or deps.healing_rounds(request.problem_class, request.size)
        )
        print(f" [难度: {difficulty}] 最大修复轮数: {max_rounds}")
        if max(
            len(request.problem_text or ""),
            len(request.initial_code or ""),
            len(request.initial_modeling_logic or ""),
        ) > 12_000:
            return (
                request.initial_code,
                "REVIEW: input too long; refusing unsafe truncation.",
                0,
                difficulty,
            )

        normalized_class = deps.normalize_class(request.problem_class)
        deps.hint_cache.pop(normalized_class, None)
        problem_objective = deps.problem_direction(request.problem_text)
        current_code = request.initial_code
        execution_baseline_code: str | None = None
        if deps.execute_code is not None:
            initial_success, initial_objective, _ = deps.invoke_executor(
                deps.execute_code, request.initial_code
            )
            initial_evidence = deps.solver_evidence(request.initial_code)
            initial_violation = initial_evidence.get("max_constr_vio")
            if (
                initial_success
                and initial_objective is not None
                and (
                    not isinstance(initial_violation, (int, float))
                    or float(initial_violation) <= 1e-5
                )
            ):
                execution_baseline_code = request.initial_code

        report = "REVIEW: no candidate was audited."
        rounds_used = 0
        pending_execution_error = request.initial_execution_error or ""
        first_error_pattern = deps.memory_error_pattern(pending_execution_error)
        last_verified_code: str | None = None
        last_audited_code: str | None = None

        for round_number in range(1, max_rounds + 1):
            rounds_used = round_number
            error_for_repair = ""
            if pending_execution_error:
                report = f"FAIL: runtime error. {pending_execution_error}"
                error_for_repair = deps.attach_error_context(
                    pending_execution_error, current_code
                )
                pending_execution_error = ""
            else:
                report = deps.audit_logic(
                    request.problem_text,
                    current_code,
                    request.exec_output if round_number == 1 else "",
                    request.initial_modeling_logic,
                    request.problem_class,
                )
                last_audited_code = current_code.strip()

                if report.startswith("REVIEW:"):
                    problem_side = (
                        "requirement extraction is uncertain" in report
                        or "problem statement is truncated" in report
                        or "is not explicitly bounded" in report
                    )
                    code_side = (
                        "code extraction is uncertain" in report
                        or "constraint comparison is uncertain" in report
                    )
                    auditor_error = (
                        "constraint comparison failed" in report
                        or "code-constraint extraction failed" in report
                        or "requirement extraction failed" in report
                    )
                    if auditor_error:
                        if round_number < max_rounds:
                            print(f" ⚠️ 第 {round_number} 轮审计器出错，下一轮重试")
                            continue
                        return current_code, report, round_number, difficulty
                    if problem_side and not code_side:
                        if deps.execute_code is not None:
                            success, objective, _ = deps.invoke_executor(
                                deps.execute_code, current_code
                            )
                            if (
                                success
                                and objective is not None
                                and deps.oracle_gate
                                and request.ground_truth is not None
                            ):
                                error = abs(objective - request.ground_truth) / max(
                                    abs(request.ground_truth), 1e-6
                                )
                                if error <= request.objective_tolerance:
                                    deps.log_gate(
                                        request.question_id,
                                        request.problem_class,
                                        round_number,
                                        "review_exec",
                                        "accept",
                                        "oracle-gated: review with correct objective",
                                        {"objective": objective, "relative_error": error},
                                    )
                                    return deps.return_pass(
                                        request.memory,
                                        request.problem_class,
                                        first_error_pattern,
                                        request.initial_code,
                                        current_code,
                                        round_number,
                                        difficulty,
                                        verified_by="review_exec",
                                        confidence=0.8,
                                        question_id=request.question_id,
                                    )
                            deps.log_gate(
                                request.question_id,
                                request.problem_class,
                                round_number,
                                "review_exec",
                                "reject",
                                "problem-side uncertainty with no oracle evidence",
                                {"exec_success": success, "objective": objective},
                            )
                        return current_code, report, round_number, difficulty

                if (
                    not deps.oracle_gate
                    and report == "PASS_EQUIVALENT"
                    and current_code.strip() != request.initial_code.strip()
                ):
                    deps.log_gate(
                        request.question_id,
                        request.problem_class,
                        round_number,
                        "audit_equiv",
                        "reject",
                        "equivalence-only pass rejected without oracle; continue repair",
                        {"code_changed": True, "audit_report": report},
                    )
                    report = (
                        "FAIL: equivalence-only audit passed without oracle.\n"
                        "The candidate changed the model and requires stronger verification."
                    )

                if (
                    not deps.oracle_gate
                    and report == "PASS_EQUIVALENT"
                    and current_code.strip() == request.initial_code.strip()
                ):
                    verdict, reason = deps.independent_verify(
                        request.problem_text, current_code
                    )
                    if verdict == "PASS":
                        witness = deps.objective_witness(
                            request.problem_text, current_code
                        )
                        deps.log_gate(
                            request.question_id,
                            request.problem_class,
                            round_number,
                            "objective_witness",
                            "accept" if witness.verdict == "PASS" else "reject",
                            witness.reason,
                            {
                                "verifier_reason": reason,
                                "solver_evidence": self._solver_summary(current_code),
                                "objective_witness": witness.as_dict(),
                            },
                        )
                        if witness.verdict != "PASS":
                            report = f"FAIL: objective witness {witness.verdict}: {witness.reason}"
                        else:
                            last_verified_code = current_code
                            deps.log_gate(
                                request.question_id,
                                request.problem_class,
                                round_number,
                                "independent_verify",
                                "accept",
                                "unchanged code passed equivalence audit and independent verifier",
                                {
                                    "verifier_reason": reason,
                                    "solver_evidence": self._solver_summary(current_code),
                                },
                            )
                    else:
                        deps.log_gate(
                            request.question_id,
                            request.problem_class,
                            round_number,
                            "independent_verify",
                            "reject",
                            f"independent verifier disagreed: {reason}",
                            {
                                "verdict": verdict,
                                "solver_evidence": self._solver_summary(current_code),
                            },
                        )
                        report = f"FAIL: independent verifier {verdict}: {reason}"

                if (
                    not deps.oracle_gate
                    and report == "PASS"
                    and current_code.strip() != request.initial_code.strip()
                ):
                    direction = deps.fuse_direction(
                        problem_objective, current_code, report
                    )
                    if direction["confidence"] not in {"high", "medium"}:
                        deps.log_gate(
                            request.question_id,
                            request.problem_class,
                            round_number,
                            "audit_pass_direction",
                            "reject",
                            "strict audit passed but direction evidence is insufficient",
                            direction,
                        )
                        report = (
                            "FAIL: strict audit passed but objective direction "
                            "evidence is insufficient for a changed model."
                        )
                    else:
                        last_verified_code = current_code

                if report in {"PASS", "PASS_EQUIVALENT"}:
                    pass_verified_by = (
                        "audit_equiv" if report == "PASS_EQUIVALENT" else "audit_pass"
                    )
                    pass_confidence = 0.9 if report == "PASS_EQUIVALENT" else 1.0
                    if deps.execute_code is None:
                        return deps.return_pass(
                            request.memory,
                            request.problem_class,
                            first_error_pattern,
                            request.initial_code,
                            current_code,
                            round_number,
                            difficulty,
                            verified_by=pass_verified_by,
                            confidence=pass_confidence,
                            question_id=request.question_id,
                        )
                    success, objective, exec_error = deps.invoke_executor(
                        deps.execute_code, current_code
                    )
                    if not success or objective is None:
                        pending_execution_error = (
                            exec_error or "Code executed but returned no objective."
                        )
                        deps.log_gate(
                            request.question_id,
                            request.problem_class,
                            round_number,
                            "audit_pass",
                            "reject",
                            "audit passed but execution failed",
                            {
                                "exec_success": success,
                                "error": pending_execution_error[:200],
                                "solver_evidence": self._solver_summary(current_code),
                            },
                        )
                        continue

                    if not deps.oracle_gate or request.ground_truth is None:
                        if (
                            report == "PASS"
                            and current_code.strip() == request.initial_code.strip()
                        ):
                            witness = deps.objective_witness(
                                request.problem_text, current_code
                            )
                            deps.log_gate(
                                request.question_id,
                                request.problem_class,
                                round_number,
                                "objective_witness",
                                "accept" if witness.verdict == "PASS" else "reject",
                                witness.reason,
                                {"objective_witness": witness.as_dict()},
                            )
                            if witness.verdict != "PASS":
                                report = (
                                    f"FAIL: objective witness {witness.verdict}: "
                                    f"{witness.reason}"
                                )
                                continue
                        last_verified_code = current_code
                        deps.log_gate(
                            request.question_id,
                            request.problem_class,
                            round_number,
                            "audit_pass",
                            "accept",
                            "audit passed and code executes (no oracle)",
                            {
                                "objective": objective,
                                "solver_evidence": self._solver_summary(current_code),
                            },
                        )
                        return deps.return_pass(
                            request.memory,
                            request.problem_class,
                            first_error_pattern,
                            request.initial_code,
                            current_code,
                            round_number,
                            difficulty,
                            verified_by=pass_verified_by,
                            confidence=pass_confidence,
                            question_id=request.question_id,
                        )

                    oracle_error = abs(objective - request.ground_truth) / max(
                        abs(request.ground_truth), 1e-6
                    )
                    if oracle_error <= request.objective_tolerance:
                        deps.log_gate(
                            request.question_id,
                            request.problem_class,
                            round_number,
                            "audit_pass",
                            "accept",
                            "oracle-gated: objective correct",
                            {"objective": objective, "relative_error": oracle_error},
                        )
                        return deps.return_pass(
                            request.memory,
                            request.problem_class,
                            first_error_pattern,
                            request.initial_code,
                            current_code,
                            round_number,
                            difficulty,
                            verified_by=pass_verified_by,
                            confidence=pass_confidence,
                            question_id=request.question_id,
                        )
                    direction = "LOWER" if objective < request.ground_truth else "HIGHER"
                    report = (
                        "FAIL: audit passed but objective differs.\n"
                        f"  ground_truth = {request.ground_truth}\n"
                        f"  code_produces = {objective}\n"
                        f"  relative_error = {oracle_error:.6g}\n"
                        f"  direction: code is {direction} than ground_truth"
                    )
                    deps.log_gate(
                        request.question_id,
                        request.problem_class,
                        round_number,
                        "audit_pass",
                        "reject",
                        "oracle-gated: audit passed but objective differs",
                        {"objective": objective, "relative_error": oracle_error},
                    )

            hint = deps.hint_cache.get(normalized_class, "")
            error_pattern = deps.memory_error_pattern(error_for_repair or report)
            if not first_error_pattern:
                first_error_pattern = error_pattern
            historical_cases: list[dict[str, Any]] = []
            if request.memory is not None and request.memory.enabled:
                historical_cases = request.memory.query(
                    request.problem_class, error_pattern
                )
                if deps.memory_log_enabled:
                    top_score = deps.memory_top_score(
                        request.memory, error_pattern, historical_cases
                    )
                    deps.log_memory(
                        problem_class=request.problem_class,
                        round_number=round_number,
                        bucket_size=request.memory.size(request.problem_class),
                        hits=len(historical_cases),
                        top_score=round(top_score, 4),
                        question_id=request.question_id,
                        selected_verified_by=[
                            case.get("verified_by", "unknown")
                            for case in historical_cases
                        ],
                        selected_confidence=[
                            round(float(case.get("confidence", 1.0)), 3)
                            for case in historical_cases
                        ],
                    )

            repaired = deps.repair_code(
                current_code,
                report,
                request.problem_text,
                execution_error=error_for_repair,
                problem_class=request.problem_class,
                extra_hint=hint,
                historical_cases=historical_cases,
            )
            if not deps.valid_python(repaired):
                pending_execution_error = (
                    "Repaired code has Python syntax error (compile() failed)."
                )
                continue

            old_direction = deps.code_direction(request.initial_code)
            new_direction = deps.code_direction(repaired)
            if (
                old_direction != "unknown"
                and new_direction != "unknown"
                and old_direction != new_direction
            ):
                text_direction = problem_objective.get("direction", "unknown")
                text_confidence = problem_objective.get("confidence", "unknown")
                if not (
                    text_direction == new_direction
                    and text_confidence in {"high", "medium"}
                ):
                    deps.log_gate(
                        request.question_id,
                        request.problem_class,
                        round_number,
                        "objective_switch",
                        "reject",
                        "repair changed objective direction without sufficient evidence",
                        {
                            "initial_code_direction": old_direction,
                            "candidate_code_direction": new_direction,
                            "problem_direction_evidence": problem_objective,
                        },
                    )
                    report = (
                        "FAIL: repair changed the Gurobi objective direction "
                        "without sufficient textual evidence."
                    )
                    continue
                deps.log_gate(
                    request.question_id,
                    request.problem_class,
                    round_number,
                    "objective_switch",
                    "accept",
                    "objective direction change supported by problem text",
                    {
                        "initial_code_direction": old_direction,
                        "candidate_code_direction": new_direction,
                        "problem_direction_evidence": problem_objective,
                    },
                )

            if deps.execute_code is not None:
                current_success, current_objective, _ = deps.invoke_executor(
                    deps.execute_code, current_code
                )
                new_success, new_objective, new_error = deps.invoke_executor(
                    deps.execute_code, repaired
                )
                if not new_success or new_objective is None:
                    deps.log_gate(
                        request.question_id,
                        request.problem_class,
                        round_number,
                        "reverse_validation",
                        "reject",
                        "repaired code cannot execute",
                        {"previous_exec_success": current_success},
                    )
                    pending_execution_error = new_error or "repaired code cannot execute"
                    continue

                previous_evidence = deps.solver_evidence(current_code)
                candidate_evidence = deps.solver_evidence(repaired)
                regression = deps.regression_check(
                    previous_evidence, candidate_evidence
                )
                if not regression["ok"]:
                    deps.log_gate(
                        request.question_id,
                        request.problem_class,
                        round_number,
                        "regression_check",
                        "reject",
                        "solver-level regression detected",
                        regression,
                    )
                    continue

                oracle_errors = None
                if (
                    deps.oracle_gate
                    and request.ground_truth is not None
                    and current_success
                    and current_objective is not None
                ):
                    oracle_errors = (
                        abs(current_objective - request.ground_truth)
                        / max(abs(request.ground_truth), 1e-6),
                        abs(new_objective - request.ground_truth)
                        / max(abs(request.ground_truth), 1e-6),
                    )
                direction = (
                    problem_objective
                    if oracle_errors is not None or not current_success
                    else deps.fuse_direction(problem_objective, repaired, report)
                )
                decision = deps.assess_candidate(
                    previous_exec_success=current_success,
                    previous_objective=current_objective,
                    candidate_objective=new_objective,
                    direction_evidence=direction,
                    regression=regression,
                    tolerance=request.objective_tolerance,
                    oracle_errors=oracle_errors,
                )
                if decision.action == "accept" and regression.get("warnings"):
                    decision = GateDecision(
                        "provisional_accept",
                        "objective improved with structural regression warnings",
                        decision.evidence,
                    )
                evidence = dict(decision.evidence)
                evidence["previous_solver_evidence"] = self._solver_summary(current_code)
                evidence["candidate_solver_evidence"] = self._solver_summary(repaired)
                deps.log_gate(
                    request.question_id,
                    request.problem_class,
                    round_number,
                    "reverse_validation",
                    decision.action,
                    decision.reason,
                    evidence,
                )
                if decision.action == "reject":
                    continue

            current_code = repaired

        if (
            not deps.oracle_gate
            and current_code.strip() != (last_audited_code or "").strip()
        ):
            final_audit = deps.audit_logic(
                request.problem_text,
                current_code,
                "",
                request.initial_modeling_logic,
                request.problem_class,
            )
            last_audited_code = current_code.strip()
            if final_audit == "PASS" and deps.execute_code is not None:
                success, objective, final_error = deps.invoke_executor(
                    deps.execute_code, current_code
                )
                if success and objective is not None:
                    deps.log_gate(
                        request.question_id,
                        request.problem_class,
                        max_rounds + 1,
                        "final_audit",
                        "accept",
                        "post-loop strict audit passed and candidate executes",
                        {"solver_evidence": self._solver_summary(current_code)},
                    )
                    return deps.return_pass(
                        request.memory,
                        request.problem_class,
                        first_error_pattern,
                        request.initial_code,
                        current_code,
                        max_rounds + 1,
                        difficulty,
                        verified_by="audit_pass",
                        confidence=1.0,
                        question_id=request.question_id,
                    )
                final_audit = (
                    "FAIL: post-loop strict audit passed but execution failed. "
                    f"{final_error}"
                )
            deps.log_gate(
                request.question_id,
                request.problem_class,
                max_rounds + 1,
                "final_audit",
                "reject",
                final_audit[:500],
                {"solver_evidence": self._solver_summary(current_code)},
            )
            report = final_audit

        fallback_code, fallback_source = deps.select_final_fallback(
            current_code=current_code,
            last_verified_code=last_verified_code,
            execution_baseline_code=execution_baseline_code,
            final_audit=report,
            oracle_gate=deps.oracle_gate,
        )
        if fallback_code is not None:
            deps.log_gate(
                request.question_id,
                request.problem_class,
                max_rounds,
                "final_fallback",
                "revert_to_verified",
                "final candidate unverified; reverted to safe baseline",
                {
                    "final_audit": report[:300],
                    "current_solver_evidence": self._solver_summary(current_code),
                    "fallback_solver_evidence": self._solver_summary(fallback_code),
                    "fallback_source": fallback_source,
                },
            )
            current_code = fallback_code
            report = (
                "FALLBACK_TO_SAFE_BASELINE: final candidate was not verified; "
                f"previous audit was {report[:300]}"
            )

        deps.hint_cache.pop(normalized_class, None)
        deps.store_exec_success(
            request.memory,
            request.problem_class,
            first_error_pattern,
            request.initial_code,
            current_code,
            deps.execute_code,
            request.ground_truth,
            max(request.objective_tolerance, 5e-3),
            report=report,
            question_id=request.question_id,
        )
        print(f" ⚠️ 达到最大轮数 ({max_rounds})，使用最后版本")
        return current_code, report, max_rounds, difficulty


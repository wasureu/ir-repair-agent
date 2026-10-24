from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .budget import BudgetLedger, BudgetSnapshot
from .contracts import (
    EditAudit,
    FailureDiagnosis,
    IRPatch,
    ModelIR,
    ValidationResult,
    assess_edit,
    diagnosis_from_validation,
)
from .memory import MemoryMatch, SqliteRepairMemory

CompilerFunc = Callable[[ModelIR, str, BudgetLedger], str]
ValidatorFunc = Callable[[ModelIR, str, BudgetLedger], ValidationResult]
PatchGeneratorFunc = Callable[
    ["RepairRequest", tuple[MemoryMatch, ...], BudgetLedger],
    Sequence[IRPatch],
]


@dataclass(frozen=True)
class RepairRequest:
    """Inputs visible to the repair controller.

    There is deliberately no ``ground_truth`` or answer field. Benchmark labels
    must remain in the offline evaluator and must not steer online repair.
    """

    model_ir: ModelIR
    current_code: str
    baseline_validation: ValidationResult
    diagnosis: FailureDiagnosis | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def resolved_diagnosis(self) -> FailureDiagnosis:
        return self.diagnosis or diagnosis_from_validation(self.baseline_validation)


@dataclass(frozen=True)
class RepairPolicy:
    required_checks: tuple[str, ...] = (
        "model_build",
        "constraint_integrity",
        "execution",
    )
    max_memory_candidates: int = 3
    max_generated_candidates: int = 2
    max_wrong_edit_rate: float = 0.0
    store_verified_memory: bool = True
    min_memory_similarity: float = 0.2
    cross_task_only: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.max_wrong_edit_rate <= 1:
            raise ValueError("max_wrong_edit_rate must be in [0, 1]")
        if self.max_memory_candidates < 0 or self.max_generated_candidates < 0:
            raise ValueError("candidate limits must be non-negative")


@dataclass(frozen=True)
class RepairOutcome:
    accepted: bool
    source: str
    original_ir: ModelIR
    final_ir: ModelIR
    final_code: str
    validation: ValidationResult
    edit_audit: EditAudit
    patch: IRPatch | None
    memory_matches: tuple[MemoryMatch, ...]
    patch_generator_calls: int
    budget_before: BudgetSnapshot
    budget_after: BudgetSnapshot
    reason: str = ""

    def as_dict(self, *, include_code: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "accepted": self.accepted,
            "source": self.source,
            "original_fingerprint": self.original_ir.fingerprint(),
            "final_fingerprint": self.final_ir.fingerprint(),
            "validation": self.validation.as_dict(),
            "edit_audit": self.edit_audit.as_dict(),
            "patch": self.patch.as_dict() if self.patch else None,
            "memory_case_ids": [
                match.case.case_id for match in self.memory_matches
            ],
            "patch_generator_calls": self.patch_generator_calls,
            "budget_before": self.budget_before.as_dict(),
            "budget_after": self.budget_after.as_dict(),
            "reason": self.reason,
        }
        if include_code:
            payload["final_code"] = self.final_code
        return payload


class RepairEngine:
    """Try verified memory first, then a constrained IR-patch generator.

    Every candidate is compiled and independently validated before acceptance.
    A cross-task memory hit therefore saves an LLM repair call only when the
    retrieved recipe survives the current problem's hard checks.
    """

    def __init__(
        self,
        *,
        memory: SqliteRepairMemory,
        compiler: CompilerFunc,
        validator: ValidatorFunc,
        patch_generator: PatchGeneratorFunc | None = None,
        policy: RepairPolicy | None = None,
    ) -> None:
        self.memory = memory
        self.compiler = compiler
        self.validator = validator
        self.patch_generator = patch_generator
        self.policy = policy or RepairPolicy()

    def _regressions(
        self,
        before: ValidationResult,
        after: ValidationResult,
    ) -> tuple[str, ...]:
        after_by_name = {check.name: check for check in after.checks}
        regressions: list[str] = []
        for check in before.checks:
            if check.status != "PASS":
                continue
            replacement = after_by_name.get(check.name)
            if replacement is None or replacement.status != "PASS":
                regressions.append(check.name)
        return tuple(sorted(set(regressions)))

    def _evaluate_candidate(
        self,
        *,
        request: RepairRequest,
        diagnosis: FailureDiagnosis,
        patch: IRPatch,
        ledger: BudgetLedger,
    ) -> tuple[ModelIR, str, ValidationResult, EditAudit]:
        candidate_ir = patch.apply(request.model_ir)
        candidate_code = self.compiler(candidate_ir, request.current_code, ledger)
        validation = self.validator(candidate_ir, candidate_code, ledger)
        edit_audit = assess_edit(
            request.model_ir,
            candidate_ir,
            allowed_elements=diagnosis.affected_elements,
            regressions=self._regressions(
                request.baseline_validation,
                validation,
            ),
        )
        return candidate_ir, candidate_code, validation, edit_audit

    def _accepted(
        self,
        validation: ValidationResult,
        edit: EditAudit,
    ) -> bool:
        return (
            validation.hard_pass(
                self.policy.required_checks,
                require_program_evidence=True,
            )
            and not edit.has_regression
            and edit.wrong_edit_rate <= self.policy.max_wrong_edit_rate
        )

    def run(
        self,
        request: RepairRequest,
        ledger: BudgetLedger,
    ) -> RepairOutcome:
        ledger.start()
        budget_before = ledger.snapshot()
        diagnosis = request.resolved_diagnosis()
        if request.baseline_validation.hard_pass(
            self.policy.required_checks,
            require_program_evidence=True,
        ):
            edit = assess_edit(
                request.model_ir,
                request.model_ir,
                allowed_elements=diagnosis.affected_elements,
            )
            return RepairOutcome(
                accepted=True,
                source="baseline",
                original_ir=request.model_ir,
                final_ir=request.model_ir,
                final_code=request.current_code,
                validation=request.baseline_validation,
                edit_audit=edit,
                patch=None,
                memory_matches=(),
                patch_generator_calls=0,
                budget_before=budget_before,
                budget_after=ledger.snapshot(),
                reason="baseline already passed all hard checks",
            )

        matches = tuple(
            self.memory.query(
                model=request.model_ir,
                diagnosis=diagnosis,
                top_k=self.policy.max_memory_candidates,
                min_similarity=self.policy.min_memory_similarity,
                cross_task_only=self.policy.cross_task_only,
            )
        )
        generator_calls = 0
        last_validation = request.baseline_validation
        last_edit = assess_edit(
            request.model_ir,
            request.model_ir,
            allowed_elements=diagnosis.affected_elements,
        )
        last_reason = "no verified memory recipe succeeded"

        for match in matches:
            patch = match.case.bind(diagnosis)
            try:
                candidate_ir, candidate_code, validation, edit = self._evaluate_candidate(
                    request=request,
                    diagnosis=diagnosis,
                    patch=patch,
                    ledger=ledger,
                )
            except Exception as exc:
                self.memory.record_trial(
                    case_id=match.case.case_id,
                    target_problem_id=request.model_ir.problem_id,
                    accepted=False,
                    failure_code=f"CANDIDATE_ERROR:{type(exc).__name__}",
                )
                last_reason = f"memory candidate failed: {exc}"
                continue

            last_validation = validation
            last_edit = edit
            accepted = self._accepted(validation, edit)
            self.memory.record_trial(
                case_id=match.case.case_id,
                target_problem_id=request.model_ir.problem_id,
                accepted=accepted,
                failure_code=(
                    "ACCEPTED"
                    if accepted
                    else validation.failure_code or "VALIDATION_REJECTED"
                ),
            )
            if accepted:
                return RepairOutcome(
                    accepted=True,
                    source="memory",
                    original_ir=request.model_ir,
                    final_ir=candidate_ir,
                    final_code=candidate_code,
                    validation=validation,
                    edit_audit=edit,
                    patch=patch,
                    memory_matches=matches,
                    patch_generator_calls=0,
                    budget_before=budget_before,
                    budget_after=ledger.snapshot(),
                    reason=f"verified memory case {match.case.case_id} passed",
                )

        if self.patch_generator is None:
            return RepairOutcome(
                accepted=False,
                source="none",
                original_ir=request.model_ir,
                final_ir=request.model_ir,
                final_code=request.current_code,
                validation=last_validation,
                edit_audit=last_edit,
                patch=None,
                memory_matches=matches,
                patch_generator_calls=0,
                budget_before=budget_before,
                budget_after=ledger.snapshot(),
                reason=last_reason,
            )

        ledger.ensure_available()
        generator_calls += 1
        generated = tuple(
            self.patch_generator(request, matches, ledger)
        )[: self.policy.max_generated_candidates]
        for patch in generated:
            try:
                candidate_ir, candidate_code, validation, edit = self._evaluate_candidate(
                    request=request,
                    diagnosis=diagnosis,
                    patch=patch,
                    ledger=ledger,
                )
            except Exception as exc:
                last_reason = f"generated candidate failed: {exc}"
                continue
            last_validation = validation
            last_edit = edit
            if self._accepted(validation, edit):
                if self.policy.store_verified_memory:
                    try:
                        self.memory.store_verified(
                            source_model=request.model_ir,
                            diagnosis=diagnosis,
                            patch=patch,
                            validation=validation,
                        )
                    except ValueError:
                        # A successful non-transferable patch may still solve the
                        # current problem, but it cannot enter cross-task memory.
                        pass
                return RepairOutcome(
                    accepted=True,
                    source="generated_ir",
                    original_ir=request.model_ir,
                    final_ir=candidate_ir,
                    final_code=candidate_code,
                    validation=validation,
                    edit_audit=edit,
                    patch=patch,
                    memory_matches=matches,
                    patch_generator_calls=generator_calls,
                    budget_before=budget_before,
                    budget_after=ledger.snapshot(),
                    reason="generated ModelIR patch passed all verification gates",
                )
            last_reason = (
                validation.failure_code
                or "generated patch failed verification or edit-scope policy"
            )

        return RepairOutcome(
            accepted=False,
            source="none",
            original_ir=request.model_ir,
            final_ir=request.model_ir,
            final_code=request.current_code,
            validation=last_validation,
            edit_audit=last_edit,
            patch=None,
            memory_matches=matches,
            patch_generator_calls=generator_calls,
            budget_before=budget_before,
            budget_after=ledger.snapshot(),
            reason=last_reason,
        )


def build_ir_patch_prompt(
    request: RepairRequest,
    matches: Sequence[MemoryMatch],
) -> str:
    """Build a compact prompt for an LLM that may change only ModelIR.

    The hook intentionally returns JSON only. Compilation and execution are
    separate tools, so the repair model does not need to rewrite complete code.
    """
    diagnosis = request.resolved_diagnosis()
    examples = []
    for match in matches:
        examples.append(
            {
                "case_id": match.case.case_id,
                "failure_code": match.case.failure_code,
                "features": list(match.case.features),
                "patch_template": match.case.patch_template.as_dict(),
                "reliability": match.case.reliability,
            }
        )
    payload = {
        "problem_id": request.model_ir.problem_id,
        "problem_class": request.model_ir.problem_class,
        "diagnosis": {
            "failure_code": diagnosis.failure_code,
            "affected_elements": list(diagnosis.affected_elements),
            "features": list(diagnosis.features),
            "summary": diagnosis.summary,
            "allow_new_elements": diagnosis.allow_new_elements,
        },
        "model_ir": request.model_ir.to_dict(),
        "verified_cross_task_recipes": examples,
    }
    return (
        "You are repairing a mathematical optimization model at ModelIR level.\n"
        "Return only JSON with one key `patches`, whose value is a list of patch "
        "objects matching the IRPatch schema. Prefer one patch. Modify only the "
        "diagnosed elements unless the diagnosis explicitly allows new elements. "
        "Do not use benchmark answers. Do not rewrite the full source code.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )

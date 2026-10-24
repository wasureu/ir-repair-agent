"""Evidence-based admission policy for persistent repair memory."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ValidationResult


TRUSTED_PROGRAM_SOURCES = frozenset({"program", "solver", "metamorphic"})


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    verified_checks: tuple[str, ...]
    missing_checks: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    untrusted_checks: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "verified_checks": list(self.verified_checks),
            "missing_checks": list(self.missing_checks),
            "failed_checks": list(self.failed_checks),
            "untrusted_checks": list(self.untrusted_checks),
        }


def assess_memory_admission(
    validation: ValidationResult,
    required_checks: tuple[str, ...],
    *,
    require_program_evidence: bool = True,
) -> AdmissionDecision:
    if validation.status != "PASS":
        return AdmissionDecision(
            admitted=False,
            reason=f"validation status is {validation.status}, not PASS",
            verified_checks=(),
        )

    by_name = {check.name: check for check in validation.checks}
    missing: list[str] = []
    failed: list[str] = []
    untrusted: list[str] = []
    verified: list[str] = []

    for name in required_checks:
        check = by_name.get(name)
        if check is None:
            missing.append(name)
            continue
        if check.status != "PASS":
            failed.append(name)
            continue
        if (
            require_program_evidence
            and check.source not in TRUSTED_PROGRAM_SOURCES
        ):
            untrusted.append(name)
            continue
        verified.append(name)

    hard_failures = tuple(
        check.name
        for check in validation.checks
        if check.hard and check.status == "FAIL"
    )
    failed.extend(name for name in hard_failures if name not in failed)
    admitted = not missing and not failed and not untrusted
    if admitted:
        reason = "all required checks passed with trusted program evidence"
    elif missing:
        reason = "required verification checks are missing"
    elif untrusted:
        reason = "required checks rely on non-program evidence"
    else:
        reason = "required verification checks failed"
    return AdmissionDecision(
        admitted=admitted,
        reason=reason,
        verified_checks=tuple(sorted(verified)),
        missing_checks=tuple(missing),
        failed_checks=tuple(failed),
        untrusted_checks=tuple(untrusted),
    )


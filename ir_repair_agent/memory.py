from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .contracts import (
    FailureDiagnosis,
    IRPatch,
    ModelIR,
    PatchOperation,
    ValidationResult,
)

DEFAULT_VERIFICATION_CHECKS = (
    "model_build",
    "constraint_integrity",
    "execution",
)

TRANSFERABLE_PATCH_OPS = {
    "replace_relation",
    "replace_rhs",
    "replace_objective_sense",
    "replace_variable_domain",
}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_class(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _tokens(values: Iterable[str]) -> set[str]:
    return {value.strip() for value in values if value and value.strip()}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _case_id(
    *,
    source_fingerprint: str,
    failure_signature: str,
    patch: IRPatch,
) -> str:
    payload = _json(
        {
            "source_fingerprint": source_fingerprint,
            "failure_signature": failure_signature,
            "patch": patch.as_dict(),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class RepairCase:
    case_id: str
    problem_class: str
    failure_code: str
    failure_signature: str
    features: tuple[str, ...]
    source_problem_id: str
    source_model_fingerprint: str
    patch_template: IRPatch
    required_checks: tuple[str, ...]
    verified_checks: tuple[str, ...]
    successes: int
    failures: int
    status: str
    created_at: float
    last_used_at: float | None = None

    @property
    def reliability(self) -> float:
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)

    def bind(self, diagnosis: FailureDiagnosis) -> IRPatch:
        return self.patch_template.bind_affected(diagnosis.affected_elements)

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "problem_class": self.problem_class,
            "failure_code": self.failure_code,
            "failure_signature": self.failure_signature,
            "features": list(self.features),
            "source_problem_id": self.source_problem_id,
            "source_model_fingerprint": self.source_model_fingerprint,
            "patch_template": self.patch_template.as_dict(),
            "required_checks": list(self.required_checks),
            "verified_checks": list(self.verified_checks),
            "successes": self.successes,
            "failures": self.failures,
            "status": self.status,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }


@dataclass(frozen=True)
class MemoryMatch:
    case: RepairCase
    score: float
    feature_similarity: float


class SqliteRepairMemory:
    """Persistent cross-task repair recipes with verification-gated admission.

    The database stores only transferable patch recipes and diagnostic
    signatures. It never stores benchmark answers, target objective values, or
    full source statements.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        required_checks: tuple[str, ...] = DEFAULT_VERIFICATION_CHECKS,
        quarantine_after_failures: int = 3,
    ) -> None:
        self.path = str(path)
        self.required_checks = tuple(required_checks)
        self.quarantine_after_failures = max(1, int(quarantine_after_failures))
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_cases (
                    case_id TEXT PRIMARY KEY,
                    problem_class TEXT NOT NULL,
                    failure_code TEXT NOT NULL,
                    failure_signature TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    source_problem_id TEXT NOT NULL,
                    source_model_fingerprint TEXT NOT NULL,
                    patch_template_json TEXT NOT NULL,
                    required_checks_json TEXT NOT NULL,
                    verified_checks_json TEXT NOT NULL,
                    successes INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    last_used_at REAL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_repair_cases_lookup
                ON repair_cases(problem_class, failure_code, status)
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_trials (
                    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    target_problem_id TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    failure_code TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES repair_cases(case_id)
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SqliteRepairMemory":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _make_template(
        self,
        patch: IRPatch,
        diagnosis: FailureDiagnosis,
    ) -> IRPatch:
        if not diagnosis.affected_elements:
            raise ValueError("a verified memory case needs affected elements")
        if any(operation.op not in TRANSFERABLE_PATCH_OPS for operation in patch.operations):
            raise ValueError(
                "only transferable patch operations can enter cross-task memory; "
                f"got {sorted({item.op for item in patch.operations})}"
            )
        target_to_placeholder = {
            element: "{{affected:" + str(index) + "}}"
            for index, element in enumerate(diagnosis.affected_elements)
        }
        template_operations: list[PatchOperation] = []
        for operation in patch.operations:
            if operation.target not in target_to_placeholder:
                raise ValueError(
                    f"patch target {operation.target!r} is outside the diagnosed scope"
                )
            template_operations.append(
                PatchOperation(
                    op=operation.op,
                    target=target_to_placeholder[operation.target],
                    payload=dict(operation.payload),
                )
            )
        return IRPatch(
            operations=tuple(template_operations),
            description=patch.description,
            generator="verified_memory",
            confidence=patch.confidence,
        )

    def store_verified(
        self,
        *,
        source_model: ModelIR,
        diagnosis: FailureDiagnosis,
        patch: IRPatch,
        validation: ValidationResult,
        now: float | None = None,
    ) -> RepairCase | None:
        """Store a successful patch only when program-backed checks pass.

        Invalid, LLM-only, or partially successful evidence is rejected. This
        is an explicit anti-contamination boundary.
        """
        if not validation.hard_pass(
            self.required_checks,
            require_program_evidence=True,
        ):
            raise ValueError("validation does not meet verified-memory admission policy")
        template = self._make_template(patch, diagnosis)
        source_fingerprint = source_model.fingerprint()
        case_id = _case_id(
            source_fingerprint=source_fingerprint,
            failure_signature=diagnosis.signature(),
            patch=template,
        )
        created_at = time.time() if now is None else float(now)
        verified_checks = tuple(
            sorted(
                {
                    check.name
                    for check in validation.checks
                    if check.status == "PASS"
                    and check.source in {"program", "solver", "metamorphic"}
                }
            )
        )
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO repair_cases (
                    case_id, problem_class, failure_code, failure_signature,
                    features_json, source_problem_id, source_model_fingerprint,
                    patch_template_json, required_checks_json,
                    verified_checks_json, successes, failures, status,
                    created_at, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 'active', ?, NULL)
                """,
                (
                    case_id,
                    _normalize_class(source_model.problem_class),
                    diagnosis.failure_code,
                    diagnosis.signature(),
                    _json(sorted(set(diagnosis.features))),
                    source_model.problem_id,
                    source_fingerprint,
                    _json(template.as_dict()),
                    _json(list(self.required_checks)),
                    _json(list(verified_checks)),
                    created_at,
                ),
            )
            inserted = cursor.rowcount > 0
        if not inserted:
            existing = self.get(case_id)
            return existing
        return self.get(case_id)

    def query(
        self,
        *,
        model: ModelIR,
        diagnosis: FailureDiagnosis,
        top_k: int = 3,
        min_similarity: float = 0.2,
        cross_task_only: bool = True,
    ) -> list[MemoryMatch]:
        if top_k <= 0:
            return []
        query_tokens = _tokens(diagnosis.features)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM repair_cases
                WHERE problem_class = ? AND failure_code = ? AND status = 'active'
                """,
                (
                    _normalize_class(model.problem_class),
                    diagnosis.failure_code,
                ),
            ).fetchall()
        current_fingerprint = model.fingerprint()
        matches: list[MemoryMatch] = []
        for row in rows:
            if cross_task_only and row["source_model_fingerprint"] == current_fingerprint:
                continue
            case = self._row_to_case(row)
            similarity = _jaccard(query_tokens, _tokens(case.features))
            if similarity < min_similarity:
                continue
            score = 0.8 * similarity + 0.2 * case.reliability
            matches.append(
                MemoryMatch(
                    case=case,
                    score=score,
                    feature_similarity=similarity,
                )
            )
        matches.sort(
            key=lambda item: (
                item.score,
                item.case.reliability,
                item.case.successes,
                item.case.created_at,
            ),
            reverse=True,
        )
        return matches[:top_k]

    def record_trial(
        self,
        *,
        case_id: str,
        target_problem_id: str,
        accepted: bool,
        failure_code: str,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT successes, failures FROM repair_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if row is None:
                raise KeyError(case_id)
            successes = int(row["successes"]) + int(accepted)
            failures = int(row["failures"]) + int(not accepted)
            status = "active"
            if (
                failures >= self.quarantine_after_failures
                and successes < failures
            ):
                status = "quarantined"
            self._connection.execute(
                """
                UPDATE repair_cases
                SET successes = ?, failures = ?, status = ?, last_used_at = ?
                WHERE case_id = ?
                """,
                (successes, failures, status, timestamp, case_id),
            )
            self._connection.execute(
                """
                INSERT INTO repair_trials (
                    case_id, target_problem_id, accepted, failure_code, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (case_id, target_problem_id, int(accepted), failure_code, timestamp),
            )

    def get(self, case_id: str) -> RepairCase | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM repair_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        return self._row_to_case(row) if row is not None else None

    def all_cases(self) -> list[RepairCase]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM repair_cases ORDER BY created_at, case_id"
            ).fetchall()
        return [self._row_to_case(row) for row in rows]

    @staticmethod
    def _row_to_case(row: sqlite3.Row) -> RepairCase:
        return RepairCase(
            case_id=str(row["case_id"]),
            problem_class=str(row["problem_class"]),
            failure_code=str(row["failure_code"]),
            failure_signature=str(row["failure_signature"]),
            features=tuple(json.loads(row["features_json"])),
            source_problem_id=str(row["source_problem_id"]),
            source_model_fingerprint=str(row["source_model_fingerprint"]),
            patch_template=IRPatch.from_dict(json.loads(row["patch_template_json"])),
            required_checks=tuple(json.loads(row["required_checks_json"])),
            verified_checks=tuple(json.loads(row["verified_checks_json"])),
            successes=int(row["successes"]),
            failures=int(row["failures"]),
            status=str(row["status"]),
            created_at=float(row["created_at"]),
            last_used_at=(
                float(row["last_used_at"])
                if row["last_used_at"] is not None
                else None
            ),
        )

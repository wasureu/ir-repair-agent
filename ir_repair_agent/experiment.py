from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .budget import BudgetLimits
from .contracts import EditAudit, ValidationResult
from .repair import RepairOutcome


def _hash_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    benchmark: str
    arm: str
    model_name: str
    prompt_version: str
    seed: int
    budget_limits: BudgetLimits
    dataset_sha256: str
    temperature: float = 0.0
    evaluator_version: str = "verified_ir_repair_v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fairness_payload(self) -> dict[str, Any]:
        # The arm and experiment id are intentionally excluded. Every other
        # field that can change accuracy or resource expenditure is included.
        return {
            "benchmark": self.benchmark,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "seed": self.seed,
            "budget_limits": self.budget_limits.as_dict(),
            "dataset_sha256": self.dataset_sha256,
            "temperature": self.temperature,
            "evaluator_version": self.evaluator_version,
            "metadata": self.metadata,
        }

    def fairness_hash(self) -> str:
        return _hash_json(self.fairness_payload())

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "arm": self.arm,
            **self.fairness_payload(),
            "fairness_hash": self.fairness_hash(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_id=str(payload["experiment_id"]),
            benchmark=str(payload["benchmark"]),
            arm=str(payload["arm"]),
            model_name=str(payload["model_name"]),
            prompt_version=str(payload["prompt_version"]),
            seed=int(payload["seed"]),
            budget_limits=BudgetLimits(**payload["budget_limits"]),
            dataset_sha256=str(payload["dataset_sha256"]),
            temperature=float(payload.get("temperature", 0.0)),
            evaluator_version=str(
                payload.get("evaluator_version", "verified_ir_repair_v1")
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ExperimentRecord:
    problem_id: str
    config: ExperimentConfig
    correct: bool
    solved: bool
    prompt_tokens: int
    completion_tokens: int
    solver_seconds: float
    wall_seconds: float
    repair_attempts: int = 0
    llm_repair_calls: int = 0
    memory_hits: int = 0
    changed_elements: tuple[str, ...] = ()
    unintended_changes: tuple[str, ...] = ()
    regressions: tuple[str, ...] = ()
    false_pass: bool = False
    validation: ValidationResult | None = None
    edit_audit: EditAudit | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def wrong_edit_rate(self) -> float:
        if not self.changed_elements:
            return 0.0
        return len(self.unintended_changes) / len(self.changed_elements)

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "config": self.config.as_dict(),
            "correct": self.correct,
            "solved": self.solved,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "solver_seconds": self.solver_seconds,
            "wall_seconds": self.wall_seconds,
            "repair_attempts": self.repair_attempts,
            "llm_repair_calls": self.llm_repair_calls,
            "memory_hits": self.memory_hits,
            "changed_elements": list(self.changed_elements),
            "unintended_changes": list(self.unintended_changes),
            "regressions": list(self.regressions),
            "wrong_edit_rate": self.wrong_edit_rate,
            "false_pass": self.false_pass,
            "validation": self.validation.as_dict() if self.validation else None,
            "edit_audit": self.edit_audit.as_dict() if self.edit_audit else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentRecord":
        validation_payload = payload.get("validation")
        validation = (
            ValidationResult.from_dict(validation_payload)
            if isinstance(validation_payload, dict)
            else None
        )
        edit_payload = payload.get("edit_audit")
        edit_audit = (
            EditAudit.from_dict(edit_payload)
            if isinstance(edit_payload, dict)
            else None
        )
        return cls(
            problem_id=str(payload["problem_id"]),
            config=ExperimentConfig.from_dict(payload["config"]),
            correct=bool(payload["correct"]),
            solved=bool(payload["solved"]),
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            completion_tokens=int(payload.get("completion_tokens", 0)),
            solver_seconds=float(payload.get("solver_seconds", 0.0)),
            wall_seconds=float(payload.get("wall_seconds", 0.0)),
            repair_attempts=int(payload.get("repair_attempts", 0)),
            llm_repair_calls=int(payload.get("llm_repair_calls", 0)),
            memory_hits=int(payload.get("memory_hits", 0)),
            changed_elements=tuple(map(str, payload.get("changed_elements", []))),
            unintended_changes=tuple(map(str, payload.get("unintended_changes", []))),
            regressions=tuple(map(str, payload.get("regressions", []))),
            false_pass=bool(payload.get("false_pass", False)),
            validation=validation,
            edit_audit=edit_audit,
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_repair_outcome(
        cls,
        *,
        config: ExperimentConfig,
        problem_id: str,
        correct: bool,
        outcome: RepairOutcome,
        metadata: dict[str, Any] | None = None,
    ) -> "ExperimentRecord":
        snapshot = outcome.budget_after
        return cls(
            problem_id=problem_id,
            config=config,
            correct=correct,
            solved=outcome.validation.status == "PASS",
            prompt_tokens=snapshot.prompt_tokens,
            completion_tokens=snapshot.completion_tokens,
            solver_seconds=snapshot.solver_seconds,
            wall_seconds=snapshot.wall_seconds,
            repair_attempts=outcome.patch_generator_calls,
            llm_repair_calls=outcome.patch_generator_calls,
            memory_hits=1 if outcome.source == "memory" else 0,
            changed_elements=outcome.edit_audit.changed_elements,
            unintended_changes=outcome.edit_audit.extra_changes,
            regressions=outcome.edit_audit.regressions,
            false_pass=(
                outcome.validation.status == "PASS"
                and not correct
            ),
            validation=outcome.validation,
            edit_audit=outcome.edit_audit,
            metadata=metadata or {},
        )

    @classmethod
    def from_legacy_result(
        cls,
        *,
        config: ExperimentConfig,
        problem_id: str,
        result: dict[str, Any],
    ) -> "ExperimentRecord":
        """Normalize the current evaluator's JSONL shape without changing it."""
        usage = result.get("token_usage", {}) or {}
        return cls(
            problem_id=problem_id,
            config=config,
            correct=bool(result.get("tag", 0)),
            solved=result.get("predicted_answer") is not None,
            prompt_tokens=int(usage.get("prompt", 0) or 0),
            completion_tokens=int(usage.get("completion", 0) or 0),
            solver_seconds=float(result.get("solver_seconds", 0.0) or 0.0),
            wall_seconds=float(result.get("elapsed_seconds", 0.0) or 0.0),
            repair_attempts=int(result.get("rounds_used", 0) or 0),
            llm_repair_calls=int(
                (result.get("token_usage_healing", {}) or {}).get("calls", 0) or 0
            ),
            metadata={
                key: result[key]
                for key in ("problem_class", "difficulty", "size")
                if key in result
            },
        )


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    iterations: int = 2000,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    return means[int(0.025 * (iterations - 1))], means[int(0.975 * (iterations - 1))]


def _aggregate(records: Sequence[ExperimentRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty run")
    total_changes = sum(len(record.changed_elements) for record in records)
    total_unintended = sum(len(record.unintended_changes) for record in records)
    return {
        "n": len(records),
        "correct": sum(record.correct for record in records),
        "accuracy": sum(record.correct for record in records) / len(records),
        "false_passes": sum(record.false_pass for record in records),
        "total_tokens": sum(record.total_tokens for record in records),
        "mean_tokens": statistics.fmean(record.total_tokens for record in records),
        "total_solver_seconds": sum(record.solver_seconds for record in records),
        "mean_wall_seconds": statistics.fmean(record.wall_seconds for record in records),
        "repair_calls": sum(record.llm_repair_calls for record in records),
        "memory_hits": sum(record.memory_hits for record in records),
        "changed_elements": total_changes,
        "unintended_changes": total_unintended,
        "wrong_edit_rate": (
            total_unintended / total_changes if total_changes else 0.0
        ),
        "regressions": sum(len(record.regressions) for record in records),
    }


def compare_runs(
    baseline: Sequence[ExperimentRecord],
    treatment: Sequence[ExperimentRecord],
    *,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    if not baseline or not treatment:
        raise ValueError("both runs must contain records")
    baseline_config = baseline[0].config
    treatment_config = treatment[0].config
    if baseline_config.fairness_hash() != treatment_config.fairness_hash():
        raise ValueError(
            "runs are not budget/model fair: "
            f"{baseline_config.fairness_payload()} != "
            f"{treatment_config.fairness_payload()}"
        )
    left = {record.problem_id: record for record in baseline}
    right = {record.problem_id: record for record in treatment}
    if len(left) != len(baseline) or len(right) != len(treatment):
        raise ValueError("duplicate problem ids in a run")
    if set(left) != set(right):
        raise ValueError("runs do not cover the same problem ids")
    ids = sorted(left)
    correctness_delta = [
        float(right[problem_id].correct) - float(left[problem_id].correct)
        for problem_id in ids
    ]
    token_delta = [
        right[problem_id].total_tokens - left[problem_id].total_tokens
        for problem_id in ids
    ]
    accuracy_ci = _bootstrap_mean_ci(
        correctness_delta,
        seed=treatment_config.seed,
        iterations=bootstrap_iterations,
    )
    token_ci = _bootstrap_mean_ci(
        token_delta,
        seed=treatment_config.seed + 1,
        iterations=bootstrap_iterations,
    )
    baseline_metrics = _aggregate(baseline)
    treatment_metrics = _aggregate(treatment)
    delta = {
        "accuracy": treatment_metrics["accuracy"] - baseline_metrics["accuracy"],
        "accuracy_ci95": list(accuracy_ci),
        "total_tokens": (
            treatment_metrics["total_tokens"] - baseline_metrics["total_tokens"]
        ),
        "mean_tokens_ci95": list(token_ci),
        "wrong_edit_rate": (
            treatment_metrics["wrong_edit_rate"]
            - baseline_metrics["wrong_edit_rate"]
        ),
        "false_passes": (
            treatment_metrics["false_passes"] - baseline_metrics["false_passes"]
        ),
        "regressions": (
            treatment_metrics["regressions"] - baseline_metrics["regressions"]
        ),
    }
    return {
        "fairness_hash": baseline_config.fairness_hash(),
        "baseline": {
            "experiment_id": baseline_config.experiment_id,
            "arm": baseline_config.arm,
            "metrics": baseline_metrics,
        },
        "treatment": {
            "experiment_id": treatment_config.experiment_id,
            "arm": treatment_config.arm,
            "metrics": treatment_metrics,
        },
        "delta": delta,
        "bootstrap_iterations": bootstrap_iterations,
    }


def load_jsonl_records(path: str | Path) -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ExperimentRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid record: {exc}") from exc
    return records


def write_jsonl_records(
    path: str | Path,
    records: Iterable[ExperimentRecord],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

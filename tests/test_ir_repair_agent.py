from __future__ import annotations

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from ir_repair_agent import (
    BudgetExceeded,
    BudgetLedger,
    BudgetLimits,
    CheckEvidence,
    ConstraintSpec,
    ExperimentConfig,
    ExperimentRecord,
    FailureDiagnosis,
    IRPatch,
    LinearExpression,
    LinearTerm,
    ModelIR,
    ObjectiveSpec,
    PatchOperation,
    QuadraticTerm,
    RepairEngine,
    RepairRequest,
    SqliteRepairMemory,
    ValidationResult,
    VariableSpec,
    assess_edit,
    compare_runs,
)


def make_ir(
    problem_id: str,
    *,
    problem_class: str = "knapsack",
    relation: str = "le",
    rhs: float = 10.0,
) -> ModelIR:
    return ModelIR(
        problem_id=problem_id,
        problem_class=problem_class,
        variables=(VariableSpec("x", "integer", lb=0),),
        objective=ObjectiveSpec(
            "max",
            LinearExpression((LinearTerm("x", 1),)),
        ),
        constraints=(
            ConstraintSpec(
                id="c1",
                family="capacity",
                lhs=LinearExpression((LinearTerm("x", 2),)),
                relation=relation,  # type: ignore[arg-type]
                rhs=rhs,
            ),
        ),
    )


def passed_validation() -> ValidationResult:
    return ValidationResult(
        "PASS",
        checks=(
            CheckEvidence("model_build", "PASS", "program"),
            CheckEvidence("constraint_integrity", "PASS", "program"),
            CheckEvidence("execution", "PASS", "solver"),
        ),
        solver_status="OPTIMAL",
        objective=5.0,
    )


def failed_validation() -> ValidationResult:
    return ValidationResult(
        "FAIL",
        checks=(
            CheckEvidence("model_build", "PASS", "program"),
            CheckEvidence(
                "constraint_integrity",
                "FAIL",
                "program",
                detail="capacity relation is reversed",
            ),
            CheckEvidence("execution", "SKIP", "solver"),
        ),
        failure_code="RELATION_WRONG",
        affected_elements=("constraint:c1",),
    )


def repair_patch() -> IRPatch:
    return IRPatch(
        operations=(
            PatchOperation(
                "replace_relation",
                "constraint:c1",
                {"relation": "ge"},
            ),
        ),
        description="Reverse the diagnosed capacity inequality.",
        generator="llm",
        confidence=0.9,
    )


class ModelIRContractTests(unittest.TestCase):
    def test_fingerprint_is_order_independent(self) -> None:
        left = ModelIR(
            problem_id="p1",
            problem_class="knapsack",
            variables=(VariableSpec("x"), VariableSpec("y")),
            objective=ObjectiveSpec(
                "min",
                LinearExpression(
                    (LinearTerm("x", 1), LinearTerm("y", 2))
                ),
            ),
            constraints=(),
        )
        right = ModelIR(
            problem_id="p2",
            problem_class="knapsack",
            variables=(VariableSpec("y"), VariableSpec("x")),
            objective=ObjectiveSpec(
                "min",
                LinearExpression(
                    (LinearTerm("y", 2), LinearTerm("x", 1))
                ),
            ),
            constraints=(),
        )
        self.assertEqual(left.fingerprint(), right.fingerprint())

    def test_patch_rejects_unbound_placeholders(self) -> None:
        patch = IRPatch(
            (
                PatchOperation(
                    "replace_relation",
                    "{{affected:0}}",
                    {"relation": "ge"},
                ),
            )
        )
        with self.assertRaises(ValueError):
            patch.apply(make_ir("p1"))

    def test_quadratic_objective_is_canonicalized(self) -> None:
        left = ModelIR(
            "p1",
            "portfolio",
            (VariableSpec("x"), VariableSpec("y")),
            ObjectiveSpec(
                "min",
                LinearExpression(),
                quadratic_terms=(QuadraticTerm("x", "y", 2),),
            ),
            (),
        )
        right = ModelIR(
            "p2",
            "portfolio",
            (VariableSpec("y"), VariableSpec("x")),
            ObjectiveSpec(
                "min",
                LinearExpression(),
                quadratic_terms=(QuadraticTerm("y", "x", 2),),
            ),
            (),
        )
        self.assertEqual(left.fingerprint(), right.fingerprint())

    def test_edit_audit_detects_out_of_scope_change(self) -> None:
        before = make_ir("before")
        after = IRPatch(
            (
                PatchOperation(
                    "replace_relation",
                    "constraint:c1",
                    {"relation": "ge"},
                ),
                PatchOperation(
                    "replace_objective_sense",
                    "objective",
                    {"sense": "min"},
                ),
            )
        ).apply(before)
        audit = assess_edit(
            before,
            after,
            allowed_elements=("constraint:c1",),
        )
        self.assertEqual(audit.changed_elements, ("objective", "constraint:c1"))
        self.assertEqual(audit.extra_changes, ("objective",))
        self.assertAlmostEqual(audit.wrong_edit_rate, 0.5)

    def test_validation_serialization_round_trip(self) -> None:
        original = passed_validation()
        restored = ValidationResult.from_dict(original.as_dict())
        self.assertEqual(restored.status, "PASS")
        self.assertTrue(
            restored.hard_pass(
                ("model_build", "constraint_integrity", "execution")
            )
        )

    def test_repair_request_has_no_answer_field(self) -> None:
        names = {item.name for item in fields(RepairRequest)}
        self.assertNotIn("ground_truth", names)
        self.assertNotIn("answer", names)


class BudgetTests(unittest.TestCase):
    def test_llm_call_limit_is_enforced(self) -> None:
        ledger = BudgetLedger(BudgetLimits(max_llm_calls=1))
        ledger.charge_llm(prompt_tokens=2, completion_tokens=3)
        with self.assertRaises(BudgetExceeded):
            ledger.ensure_available()

    def test_solver_limit_is_enforced(self) -> None:
        ledger = BudgetLedger(BudgetLimits(max_solver_seconds=1.0))
        with self.assertRaises(BudgetExceeded):
            ledger.charge_solver(1.1)


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory = SqliteRepairMemory(Path(self.temp_dir.name) / "memory.sqlite")
        self.diagnosis = FailureDiagnosis(
            failure_code="RELATION_WRONG",
            affected_elements=("constraint:c1",),
            features=("family:capacity", "relation:le"),
        )

    def tearDown(self) -> None:
        self.memory.close()
        self.temp_dir.cleanup()

    def test_memory_rejects_llm_only_evidence(self) -> None:
        llm_only = ValidationResult(
            "PASS",
            checks=(
                CheckEvidence("model_build", "PASS", "llm"),
                CheckEvidence("constraint_integrity", "PASS", "llm"),
                CheckEvidence("execution", "PASS", "llm"),
            ),
        )
        with self.assertRaises(ValueError):
            self.memory.store_verified(
                source_model=make_ir("source"),
                diagnosis=self.diagnosis,
                patch=repair_patch(),
                validation=llm_only,
            )

    def test_verified_case_is_bound_for_a_different_task(self) -> None:
        stored = self.memory.store_verified(
            source_model=make_ir("source"),
            diagnosis=self.diagnosis,
            patch=repair_patch(),
            validation=passed_validation(),
            now=1.0,
        )
        self.assertIsNotNone(stored)
        target = make_ir("target", rhs=12)
        matches = self.memory.query(model=target, diagnosis=self.diagnosis)
        self.assertEqual(len(matches), 1)
        bound = matches[0].case.bind(self.diagnosis)
        repaired = bound.apply(target)
        self.assertEqual(repaired.constraint("c1").relation, "ge")

    def test_failed_memory_trials_can_quarantine_recipe(self) -> None:
        stored = self.memory.store_verified(
            source_model=make_ir("source"),
            diagnosis=self.diagnosis,
            patch=repair_patch(),
            validation=passed_validation(),
        )
        assert stored is not None
        for index in range(3):
            self.memory.record_trial(
                case_id=stored.case_id,
                target_problem_id=f"target-{index}",
                accepted=False,
                failure_code="NEGATIVE_TRANSFER",
            )
        refreshed = self.memory.get(stored.case_id)
        assert refreshed is not None
        self.assertEqual(refreshed.status, "quarantined")


class RepairEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory = SqliteRepairMemory(Path(self.temp_dir.name) / "memory.sqlite")
        self.diagnosis = FailureDiagnosis(
            failure_code="RELATION_WRONG",
            affected_elements=("constraint:c1",),
            features=("family:capacity", "relation:le"),
        )
        self.memory.store_verified(
            source_model=make_ir("source"),
            diagnosis=self.diagnosis,
            patch=repair_patch(),
            validation=passed_validation(),
        )

    def tearDown(self) -> None:
        self.memory.close()
        self.temp_dir.cleanup()

    def test_memory_hit_avoids_llm_repair_call(self) -> None:
        def compiler(model: ModelIR, _code: str, _ledger: BudgetLedger) -> str:
            return model.canonical_json()

        def validator(model: ModelIR, _code: str, _ledger: BudgetLedger) -> ValidationResult:
            return (
                passed_validation()
                if model.constraint("c1").relation == "ge"
                else failed_validation()
            )

        def forbidden_generator(*_args: object) -> tuple[IRPatch, ...]:
            raise AssertionError("LLM patch generator must not be called on a memory hit")

        engine = RepairEngine(
            memory=self.memory,
            compiler=compiler,
            validator=validator,
            patch_generator=forbidden_generator,
        )
        request = RepairRequest(
            model_ir=make_ir("target", rhs=12),
            current_code="legacy code",
            baseline_validation=failed_validation(),
            diagnosis=self.diagnosis,
        )
        outcome = engine.run(
            request,
            BudgetLedger(BudgetLimits(max_llm_calls=0)),
        )
        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.source, "memory")
        self.assertEqual(outcome.patch_generator_calls, 0)
        self.assertEqual(outcome.budget_after.llm_calls, 0)


class ExperimentTests(unittest.TestCase):
    def _config(self, arm: str, experiment_id: str) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id=experiment_id,
            benchmark="BWOR",
            arm=arm,
            model_name="same-model",
            prompt_version="v1",
            seed=7,
            budget_limits=BudgetLimits(
                max_total_tokens=1000,
                max_wall_seconds=60,
            ),
            dataset_sha256="abc",
        )

    def test_fairness_hash_ignores_arm_name(self) -> None:
        baseline = self._config("baseline", "run-a")
        treatment = self._config("memory", "run-b")
        self.assertEqual(baseline.fairness_hash(), treatment.fairness_hash())

    def test_compare_runs_reports_accuracy_and_token_delta(self) -> None:
        baseline_config = self._config("baseline", "run-a")
        treatment_config = self._config("memory", "run-b")
        baseline = [
            ExperimentRecord(
                "p1",
                baseline_config,
                correct=True,
                solved=True,
                prompt_tokens=10,
                completion_tokens=10,
                solver_seconds=1,
                wall_seconds=2,
            ),
            ExperimentRecord(
                "p2",
                baseline_config,
                correct=False,
                solved=False,
                prompt_tokens=20,
                completion_tokens=20,
                solver_seconds=1,
                wall_seconds=2,
            ),
        ]
        treatment = [
            ExperimentRecord(
                "p1",
                treatment_config,
                correct=True,
                solved=True,
                prompt_tokens=5,
                completion_tokens=5,
                solver_seconds=1,
                wall_seconds=2,
                memory_hits=1,
            ),
            ExperimentRecord(
                "p2",
                treatment_config,
                correct=True,
                solved=True,
                prompt_tokens=5,
                completion_tokens=5,
                solver_seconds=1,
                wall_seconds=2,
                memory_hits=1,
            ),
        ]
        report = compare_runs(
            baseline,
            treatment,
            bootstrap_iterations=100,
        )
        self.assertAlmostEqual(report["delta"]["accuracy"], 0.5)
        self.assertEqual(report["delta"]["total_tokens"], -40)

    def test_compare_runs_rejects_budget_mismatch(self) -> None:
        baseline_config = self._config("baseline", "run-a")
        treatment_config = ExperimentConfig(
            experiment_id="run-b",
            benchmark="BWOR",
            arm="memory",
            model_name="same-model",
            prompt_version="v1",
            seed=7,
            budget_limits=BudgetLimits(max_total_tokens=2000),
            dataset_sha256="abc",
        )
        left = [
            ExperimentRecord(
                "p1",
                baseline_config,
                correct=True,
                solved=True,
                prompt_tokens=1,
                completion_tokens=1,
                solver_seconds=1,
                wall_seconds=1,
            )
        ]
        right = [
            ExperimentRecord(
                "p1",
                treatment_config,
                correct=True,
                solved=True,
                prompt_tokens=1,
                completion_tokens=1,
                solver_seconds=1,
                wall_seconds=1,
            )
        ]
        with self.assertRaises(ValueError):
            compare_runs(left, right)


if __name__ == "__main__":
    unittest.main()

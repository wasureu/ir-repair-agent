import io
import unittest
from contextlib import redirect_stdout

from ir_repair_agent.repair_gate import GateDecision
from ir_repair_agent.repair_pipeline import (
    RepairPipeline,
    RepairPipelineDependencies,
    RepairPipelineRequest,
)


EVIDENCE = {
    "status_name": "OPTIMAL",
    "objective": 9.0,
    "objective_sense": 1,
    "sol_count": 1,
    "max_constr_vio": 0.0,
    "sum_constr_vio": 0.0,
    "infeasible_constraints": 0,
    "variables": [],
    "constraints": [],
}


class Witness:
    verdict = "PASS"
    reason = "objective symbols match"

    def as_dict(self):
        return {"verdict": self.verdict, "reason": self.reason}


class PassReturn:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        code = args[4]
        return code, "PASS", args[5] if len(args) > 5 else 1, args[6] if len(args) > 6 else "easy"


class RepairPipelineTests(unittest.TestCase):
    def make_deps(self, audit_results):
        audits = iter(audit_results)
        return RepairPipelineDependencies(
            audit_logic=lambda *_args: next(audits),
            repair_code=lambda *_args, **_kwargs: "repaired",
            execute_code=lambda code: (True, 10.0 if code == "initial" else 9.0, ""),
            invoke_executor=lambda _func, code: (True, 10.0 if code == "initial" else 9.0, ""),
            solver_evidence=lambda _code: EVIDENCE,
            problem_direction=lambda _text: {"direction": "min", "confidence": "high"},
            fuse_direction=lambda *_args: {
                "direction": "min",
                "confidence": "high",
            },
            independent_verify=lambda *_args: ("PASS", "ok"),
            objective_witness=lambda *_args: Witness(),
            log_gate=lambda *_args, **_kwargs: None,
            memory_error_pattern=lambda text: (text or "")[:200],
            hint_cache={},
            normalize_class=str.lower,
            detect_difficulty=lambda *_args: "easy",
            healing_rounds=lambda *_args: 2,
            valid_python=lambda _code: True,
            attach_error_context=lambda error, _code: error,
            code_direction=lambda _code: "min",
            return_pass=PassReturn(),
            store_exec_success=lambda *_args, **_kwargs: None,
            memory_top_score=lambda *_args: 0.0,
            log_memory=lambda *_args, **_kwargs: None,
            memory_log_enabled=False,
        )

    def test_first_round_audit_pass_is_final(self):
        deps = self.make_deps(["PASS"])
        pipeline = RepairPipeline(deps)
        with redirect_stdout(io.StringIO()):
            result = pipeline.run(
                RepairPipelineRequest(
                    problem_text="problem",
                    initial_code="initial",
                    initial_modeling_logic="",
                    exec_output="",
                    problem_class="toy",
                    size=3,
                    question_id="q1",
                )
            )
        self.assertEqual(result[0], "initial")
        self.assertEqual(result[1], "PASS")

    def test_last_round_candidate_gets_final_audit(self):
        deps = self.make_deps(["FAIL", "PASS"])
        pipeline = RepairPipeline(deps)
        with redirect_stdout(io.StringIO()):
            result = pipeline.run(
                RepairPipelineRequest(
                    problem_text="problem",
                    initial_code="initial",
                    initial_modeling_logic="",
                    exec_output="",
                    problem_class="toy",
                    size=3,
                    question_id="q1",
                    max_rounds_override=1,
                )
            )
        self.assertEqual(result[0], "repaired")
        self.assertEqual(result[1], "PASS")


if __name__ == "__main__":
    unittest.main()


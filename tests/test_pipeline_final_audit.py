import sys
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


CODE_DIR = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

import constraint_auditor_v3 as auditor  # noqa: E402


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


def direction_evidence():
    return {
        "direction": "min",
        "confidence": "high",
        "sources": {
            "rule": {"direction": "min"},
            "llm": {"direction": "min"},
            "code": {"direction": "min"},
            "audit": {"direction": "min"},
        },
    }


class PipelineFinalAuditTests(unittest.TestCase):
    def _run(self, audit_results):
        def execute(code):
            return True, 10.0 if code == "initial" else 9.0, ""

        with (
            mock.patch.object(
                auditor,
                "audit_logic_hybrid",
                side_effect=audit_results,
            ),
            mock.patch.object(auditor, "repair_code", return_value="repaired"),
            mock.patch.object(auditor, "_valid_python", return_value=True),
            mock.patch.object(
                auditor,
                "_problem_direction_evidence",
                return_value=direction_evidence(),
            ),
            mock.patch.object(
                auditor,
                "_fuse_direction_evidence",
                return_value=direction_evidence(),
            ),
        ):
            with redirect_stdout(io.StringIO()):
                return auditor.run_healing_pipeline(
                    problem_text="problem",
                    initial_code="initial",
                    initial_modeling_logic="",
                    exec_output="",
                    problem_class="toy",
                    size=3,
                    call_llm_func=lambda *_args, **_kwargs: "",
                    max_rounds_override=1,
                    execute_code_func=execute,
                    execution_evidence_func=lambda _code: EVIDENCE,
                    question_id="q1",
                )

    def test_last_candidate_receives_final_audit(self):
        code, report, rounds, _difficulty = self._run(["FAIL", "PASS"])
        self.assertEqual(code, "repaired")
        self.assertEqual(report, "PASS")
        self.assertEqual(rounds, 2)

    def test_failed_final_audit_falls_back_to_initial(self):
        code, report, _rounds, _difficulty = self._run(["FAIL", "FAIL"])
        self.assertEqual(code, "initial")
        self.assertIn("FALLBACK_TO_SAFE_BASELINE", report)

    def test_equivalent_final_audit_falls_back_to_initial(self):
        code, report, _rounds, _difficulty = self._run(
            ["FAIL", "PASS_EQUIVALENT"]
        )
        self.assertEqual(code, "initial")
        self.assertIn("FALLBACK_TO_SAFE_BASELINE", report)


if __name__ == "__main__":
    unittest.main()


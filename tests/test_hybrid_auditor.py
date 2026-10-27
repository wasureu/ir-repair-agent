import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from ir_repair_agent.audit.hybrid_auditor import (
    AuditDependencies,
    audit_logic_hybrid,
    safe_pass,
    validate_constraints,
)


def constraint(cid="c1", expression="x <= 1"):
    return {"id": cid, "type": "constraint", "expression": expression}


class HybridAuditorTests(unittest.TestCase):
    def test_validate_constraints_rejects_duplicates(self):
        error = validate_constraints(
            {"constraints": [constraint(), constraint()], "notes": []},
            "code extraction",
        )
        self.assertIn("duplicate id", error)

    def test_safe_pass_requires_full_bidirectional_coverage(self):
        expected = [constraint()]
        implemented = [constraint()]
        audit = {
            "verdict": "PASS",
            "differences": [],
            "expected_results": [
                {
                    "expected_id": "c1",
                    "implemented_id": "c1",
                    "status": "MATCHED",
                }
            ],
            "implemented_results": [
                {
                    "implemented_id": "c1",
                    "expected_id": "c1",
                    "status": "MATCHED",
                }
            ],
        }
        self.assertTrue(safe_pass(audit, expected, implemented))
        audit["implemented_results"] = []
        self.assertFalse(safe_pass(audit, expected, implemented))

    def test_three_stage_audit_pass_and_hint_injection(self):
        prompts = []
        implemented = {"constraints": [constraint()], "notes": []}
        expected = {"constraints": [constraint()], "notes": []}
        comparison = {
            "verdict": "PASS",
            "differences": [],
            "expected_results": [
                {
                    "expected_id": "c1",
                    "implemented_id": "c1",
                    "status": "MATCHED",
                }
            ],
            "implemented_results": [
                {
                    "implemented_id": "c1",
                    "expected_id": "c1",
                    "status": "MATCHED",
                }
            ],
            "notes": [],
        }

        def ask(prompt):
            prompts.append(prompt)
            return [implemented, expected, comparison][len(prompts) - 1]

        deps = AuditDependencies(
            hints={"toy": "check capacity"},
            ask_json_retry=ask,
            normalize_class=str.lower,
        )
        result = audit_logic_hybrid(
            "problem text",
            "import gurobipy as gp\n" + "x" * 60,
            "",
            "",
            deps,
            problem_class="toy",
        )
        self.assertEqual(result, "PASS")
        self.assertIn("check capacity", prompts[0])
        self.assertEqual(len(prompts), 3)

    def test_differences_can_pass_equivalence(self):
        implemented = {"constraints": [constraint(expression="x + y <= 1")], "notes": []}
        expected = {"constraints": [constraint(expression="x <= 1 - y")], "notes": []}
        comparison = {
            "verdict": "FAIL",
            "differences": [
                {
                    "status": "MISMATCH",
                    "reason": "algebraic form",
                    "expected_constraint": "x <= 1-y",
                    "implemented_constraint": "x+y <= 1",
                }
            ],
            "expected_results": [],
            "implemented_results": [],
            "notes": [],
        }
        calls = iter([implemented, expected, comparison])
        deps = AuditDependencies(
            ask_json_retry=lambda _prompt: next(calls),
            normalize_class=str.lower,
            verify_equivalence=lambda **_kwargs: {"all_equivalent": True},
            equivalence_ok=lambda _result, _diffs: True,
        )
        with redirect_stdout(io.StringIO()):
            result = audit_logic_hybrid(
                "problem",
                "import gurobipy as gp\n" + "x" * 60,
                "",
                "",
                deps,
                problem_class="toy",
            )
        self.assertEqual(result, "PASS_EQUIVALENT")

    def test_self_diagnosis_switch_skips_consensus_calls(self):
        implemented = {"constraints": [constraint()], "notes": []}
        expected = {"constraints": [constraint()], "notes": []}
        comparison = {
            "verdict": "PASS",
            "differences": [],
            "expected_results": [
                {
                    "expected_id": "c1",
                    "implemented_id": "c1",
                    "status": "MATCHED",
                }
            ],
            "implemented_results": [
                {
                    "implemented_id": "c1",
                    "expected_id": "c1",
                    "status": "MATCHED",
                }
            ],
            "notes": [],
        }
        calls = iter([implemented, expected, comparison])
        self_diagnose = mock.Mock(return_value="hint")
        deps = AuditDependencies(
            ask_json_retry=lambda _prompt: next(calls),
            normalize_class=str.lower,
            self_diagnose=self_diagnose,
            verify_self_diagnosis=lambda *_args: {"all_valid": True},
        )
        with redirect_stdout(io.StringIO()):
            result = audit_logic_hybrid(
                "problem",
                "import gurobipy as gp\n" + "x" * 60,
                "",
                "",
                deps,
                problem_class="toy",
                self_diagnosis_enabled=False,
            )
        self.assertEqual(result, "PASS")
        self_diagnose.assert_not_called()


if __name__ == "__main__":
    unittest.main()


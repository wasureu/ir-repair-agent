import unittest

from ir_repair_agent.repair_gate import (
    assess_candidate,
    objective_improved,
    objective_nochange,
    select_final_fallback,
)


class RepairGateTests(unittest.TestCase):
    def test_objective_helpers(self):
        self.assertTrue(objective_improved("min", 10, 9, 1e-3))
        self.assertFalse(objective_improved("max", 10, 9, 1e-3))
        self.assertTrue(objective_nochange(10, 10.004, 1e-3))
        self.assertFalse(objective_nochange(10, 10.5, 1e-3))

    def test_candidate_decisions(self):
        direction = {"direction": "min", "confidence": "high"}
        self.assertEqual(
            assess_candidate(
                previous_exec_success=True,
                previous_objective=10,
                candidate_objective=9,
                direction_evidence=direction,
            ).action,
            "accept",
        )
        self.assertEqual(
            assess_candidate(
                previous_exec_success=True,
                previous_objective=10,
                candidate_objective=11,
                direction_evidence=direction,
            ).action,
            "reject",
        )
        self.assertEqual(
            assess_candidate(
                previous_exec_success=True,
                previous_objective=10,
                candidate_objective=10.001,
                direction_evidence=direction,
            ).action,
            "provisional_accept",
        )

    def test_regression_blocks_acceptance(self):
        result = assess_candidate(
            previous_exec_success=True,
            previous_objective=10,
            candidate_objective=9,
            direction_evidence={"direction": "min", "confidence": "high"},
            regression={"ok": False, "severe": ["domain"]},
        )
        self.assertEqual(result.action, "reject")

    def test_final_fallback(self):
        fallback, source = select_final_fallback(
            current_code="bad",
            last_verified_code="verified",
            execution_baseline_code="initial",
            final_audit="FAIL",
            oracle_gate=False,
        )
        self.assertEqual(fallback, "verified")
        self.assertEqual(source, "last_verified")
        self.assertEqual(
            select_final_fallback(
                current_code="bad",
                last_verified_code=None,
                execution_baseline_code="initial",
                final_audit="FAIL",
                oracle_gate=False,
            )[0],
            "initial",
        )


if __name__ == "__main__":
    unittest.main()


import unittest

from ir_repair_agent.execution_evidence import (
    ConstraintEvidence,
    SolverEvidence,
    VariableEvidence,
    safe_float,
)


class ExecutionEvidenceTests(unittest.TestCase):
    def test_safe_float_handles_non_finite_values(self):
        self.assertEqual(safe_float(float("inf")), "inf")
        self.assertEqual(safe_float(float("-inf")), "-inf")
        self.assertIsNone(safe_float(float("nan")))
        self.assertEqual(safe_float(3), 3.0)

    def test_solver_evidence_round_trip(self):
        evidence = SolverEvidence(
            status=2,
            status_name="OPTIMAL",
            solution_count=1,
            objective=1.0,
            objective_sense=1,
            model_constraint_violation=0.0,
            max_constraint_violation=0.0,
            sum_constraint_violation=0.0,
            infeasible_constraints=0,
            variables=(VariableEvidence("x", "C", 0.0, "inf", 1.0),),
            constraints=(ConstraintEvidence("c", ">", 1.0, 0.0, 0.0),),
        )
        payload = evidence.to_dict()
        self.assertEqual(payload["status_name"], "OPTIMAL")
        self.assertEqual(payload["variables"][0]["name"], "x")
        self.assertEqual(payload["constraints"][0]["violation"], 0.0)


if __name__ == "__main__":
    unittest.main()

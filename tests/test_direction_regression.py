import unittest

from ir_repair_agent.audit.direction import (
    code_objective_direction,
    fuse_direction_evidence,
    report_objective_direction,
    rule_direction_evidence,
)
from ir_repair_agent.verification import regression_issues


class DirectionRegressionTests(unittest.TestCase):
    def test_direction_sources(self):
        self.assertEqual(
            rule_direction_evidence("The objective is to minimize total cost.")["direction"],
            "min",
        )
        self.assertEqual(
            code_objective_direction("model.setObjective(cost, GRB.MAXIMIZE)"),
            "max",
        )
        self.assertEqual(report_objective_direction("MINIMIZE total cost"), "min")

    def test_fused_direction(self):
        base = {
            "direction": "max",
            "confidence": "high",
            "sources": {
                "rule": {"direction": "max"},
                "llm": {"direction": "max"},
            },
        }
        fused = fuse_direction_evidence(
            base,
            "model.setObjective(profit, GRB.MAXIMIZE)",
            "MAXIMIZE profit",
        )
        self.assertEqual(fused["direction"], "max")
        self.assertEqual(fused["confidence"], "high")

    def test_regression_detection(self):
        result = regression_issues(
            {
                "status_name": "OPTIMAL",
                "max_constr_vio": 0.0,
                "variables": [{"name": "x", "vtype": "I"}],
                "constraints": [{"name": f"c{i}"} for i in range(5)],
            },
            {
                "status_name": "INFEASIBLE",
                "max_constr_vio": 1.0,
                "variables": [{"name": "x", "vtype": "C"}],
                "constraints": [],
            },
        )
        self.assertFalse(result["ok"])
        self.assertTrue(any("solver_status_regressed" in x for x in result["severe"]))
        self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()

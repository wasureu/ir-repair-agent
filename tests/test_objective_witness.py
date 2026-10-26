import unittest

from ir_repair_agent.audit.objective_witness import (
    extract_objective_witness,
    verify_objective_witness,
)


class ObjectiveWitnessTests(unittest.TestCase):
    def test_extracts_objective_symbols(self):
        code = """
model.setObjective(
    gp.quicksum(distance[i, j] * y[i, j] for i, j in arcs),
    GRB.MINIMIZE,
)
"""
        text, symbols = extract_objective_witness(code)
        self.assertIn("distance", symbols)
        self.assertIn("y", symbols)
        self.assertIn("quicksum", text)

    def test_required_symbol_mismatch_fails(self):
        code = """
model.setObjective(
    gp.quicksum(distance[i, j] * y[i, j] for i, j in arcs),
    GRB.MINIMIZE,
)
"""
        witness = verify_objective_witness(
            "The transportation decision variable is x.",
            code,
            lambda _: {
                "verdict": "PASS",
                "required_symbols": ["x"],
                "forbidden_symbols": ["y"],
                "reason": "objective must use x",
                "confidence": 0.9,
            },
        )
        self.assertTrue(witness.hard_fail)
        self.assertEqual(witness.verdict, "FAIL")

    def test_matching_symbol_passes(self):
        code = """
model.setObjective(
    gp.quicksum(distance[i, j] * x[i, j] for i, j in arcs),
    GRB.MINIMIZE,
)
"""
        witness = verify_objective_witness(
            "The transportation decision variable is x.",
            code,
            lambda _: {
                "verdict": "PASS",
                "required_symbols": ["x"],
                "forbidden_symbols": ["y"],
                "reason": "objective uses x",
                "confidence": 0.9,
            },
        )
        self.assertEqual(witness.verdict, "PASS")


if __name__ == "__main__":
    unittest.main()


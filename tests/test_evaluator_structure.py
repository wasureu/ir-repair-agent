import ast
import unittest
from pathlib import Path


EVALUATOR = (
    Path(__file__).resolve().parent.parent
    / "code"
    / "evaluation_tir_gurobi_robust.py"
)


class EvaluatorStructureTests(unittest.TestCase):
    def test_prompt_template_is_a_class_constant(self):
        tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
        evaluator = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SIRGurobiRobustEvaluator"
        )

        self.assertIn(
            "PROMPT_TEMPLATE",
            {
                target.id
                for node in evaluator.body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            },
        )


if __name__ == "__main__":
    unittest.main()

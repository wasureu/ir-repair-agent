import unittest

from ir_repair_agent.repair_patcher import (
    RepairPromptContext,
    build_repair_prompt,
    repair_code,
)


class RepairPatcherTests(unittest.TestCase):
    def test_prompt_contains_compact_and_goal_guidance(self):
        prompt = build_repair_prompt(
            RepairPromptContext(
                original_code="model = gp.Model()",
                audit_result="FAIL: [MISSING] priority goal",
                problem_text="Use lexicographic priorities",
                problem_class="industryor",
            )
        )
        self.assertIn("Cartesian-product", prompt)
        self.assertIn("lexicographic", prompt)
        self.assertIn("Preserve integer/continuous", prompt)

    def test_model_too_large_adds_warning(self):
        prompt = build_repair_prompt(
            RepairPromptContext(
                original_code="code",
                audit_result="FAIL",
                problem_text="problem",
                execution_error="GurobiError: Model too large for license",
            )
        )
        self.assertIn("exceeded the model-size limit", prompt)

    def test_repair_code_parses_fence(self):
        result = repair_code(
            "old",
            "audit",
            "problem",
            lambda _: "```python\nimport gurobipy as gp\n```",
        )
        self.assertIn("import gurobipy", result)


if __name__ == "__main__":
    unittest.main()


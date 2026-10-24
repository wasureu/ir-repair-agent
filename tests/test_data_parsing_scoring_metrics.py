import json
import tempfile
import unittest
from pathlib import Path

from ir_repair_agent.data import load_frozen_codes, load_problems
from ir_repair_agent.evaluation import aggregate_token_usage, difference_usage, is_correct
from ir_repair_agent.llm import extract_numeric_answer, extract_python_code


class MigrationModuleTests(unittest.TestCase):
    def test_parse_code_and_answer(self):
        response = """Explanation
```python
import gurobipy as gp
model = gp.Model()
```
<answer>12.5</answer>"""
        self.assertIn("gp.Model", extract_python_code(response))
        self.assertEqual(extract_numeric_answer(response), "12.5")

    def test_load_problem_and_frozen_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "en_question": "question",
                        "en_answer": "42",
                        "problem_class": "toy",
                        "size": 3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            frozen = root / "codes.json"
            frozen.write_text(json.dumps({"question": "code"}), encoding="utf-8")

            problems = load_problems(data)
            self.assertEqual(problems[0].question, "question")
            self.assertEqual(problems[0].answer, 42.0)
            self.assertEqual(load_frozen_codes(frozen), {"question": "code"})

    def test_scoring_and_usage(self):
        self.assertTrue(is_correct(99.95, 100.0))
        self.assertFalse(is_correct(98.0, 100.0))
        self.assertEqual(
            difference_usage(
                {"prompt": 5, "completion": 3, "total": 8, "calls": 1},
                {"prompt": 2, "completion": 1, "total": 3, "calls": 1},
            ),
            {"prompt": 3, "completion": 2, "total": 5, "calls": 0},
        )
        result = aggregate_token_usage(
            [
                {
                    "token_usage": {"prompt": 2, "completion": 3, "total": 5, "calls": 1},
                    "token_usage_normal": {"prompt": 1, "completion": 1, "total": 2, "calls": 1},
                    "token_usage_healing": {"prompt": 1, "completion": 2, "total": 3, "calls": 0},
                }
            ]
        )
        self.assertEqual(result["n_samples"], 1)
        self.assertEqual(result["total"]["total"], 5)


if __name__ == "__main__":
    unittest.main()

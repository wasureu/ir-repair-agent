import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ir_repair_agent.data.dataset import ProblemRecord
from ir_repair_agent.evaluation.runner import (
    EvaluationRunner,
    HealingOutcome,
    TokenUsageTracker,
)


def problem():
    return {
        "question": "maximize x",
        "answer": 2.0,
        "problem_class": "toy",
        "size": 3,
    }


def record():
    return ProblemRecord(
        question="maximize x",
        answer=2.0,
        problem_class="toy",
        size=3,
        raw=problem(),
    )


def usage(prompt=2, completion=3):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


class RunnerTests(unittest.TestCase):
    def make_runner(self, *, initial_codes=None, enable_healing=False):
        tracker = TokenUsageTracker()

        def call_llm(_prompt):
            tracker.record(usage())
            return "generated"

        return EvaluationRunner(
            input_path="input.jsonl",
            output_path="output.jsonl",
            model="toy",
            max_workers=1,
            prompt_template="{question}",
            usage_tracker=tracker,
            call_llm=call_llm,
            extract_code=lambda value: f"code:{value}",
            execute_code=lambda code: (True, 2.0, ""),
            initial_codes=initial_codes,
            enable_healing=enable_healing,
        )

    def test_generated_initial_result_records_usage_and_score(self):
        result = self.make_runner().evaluate_one(problem())
        self.assertEqual(result["code"], "code:generated")
        self.assertEqual(result["tag"], 1)
        self.assertEqual(result["token_usage"]["total"], 5)
        self.assertEqual(result["token_usage_healing"]["total"], 0)

    def test_frozen_code_skips_llm_and_is_executed(self):
        executed = []
        tracker = TokenUsageTracker()
        runner = EvaluationRunner(
            input_path="input.jsonl",
            output_path="output.jsonl",
            model="toy",
            max_workers=1,
            prompt_template="{question}",
            usage_tracker=tracker,
            call_llm=lambda _prompt: self.fail("LLM should not be called"),
            extract_code=lambda value: value,
            execute_code=lambda code: (
                executed.append(code) or (True, 2.0, "")
            ),
            initial_codes={"maximize x": "frozen"},
        )
        result = runner.initial_result(problem())
        self.assertEqual(result["code"], "frozen")
        self.assertEqual(executed, ["frozen"])
        self.assertEqual(result["token_usage"]["calls"], 0)

    def test_healing_falls_back_when_candidate_cannot_execute(self):
        tracker = TokenUsageTracker()

        def call_llm(_prompt):
            tracker.record(usage())
            return "generated"

        def execute(code):
            if code == "candidate":
                return False, None, "boom"
            return True, 2.0, ""

        runner = EvaluationRunner(
            input_path="input.jsonl",
            output_path="output.jsonl",
            model="toy",
            max_workers=1,
            prompt_template="{question}",
            usage_tracker=tracker,
            call_llm=call_llm,
            extract_code=lambda value: f"code:{value}",
            execute_code=execute,
            initial_codes={"maximize x": "initial"},
            enable_healing=True,
            healing_callback=lambda _request: HealingOutcome(
                code="candidate",
                audit_report="FAIL",
                rounds_used=2,
                difficulty="easy",
            ),
        )
        result = runner.evaluate_one_with_healing(problem())
        self.assertEqual(result["code"], "initial")
        self.assertEqual(result["tag"], 1)
        self.assertEqual(result["audit_result"], "FALLBACK: repair failed to execute")

    def test_run_writes_incremental_rows_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            tracker = TokenUsageTracker()
            runner = EvaluationRunner(
                input_path="input.jsonl",
                output_path=output,
                model="toy",
                max_workers=1,
                prompt_template="{question}",
                usage_tracker=tracker,
                call_llm=lambda _prompt: "generated",
                extract_code=lambda value: value,
                execute_code=lambda _code: (True, 2.0, ""),
                initial_codes={"maximize x": "frozen"},
                enable_healing=False,
                problem_loader=lambda _path: [record()],
            )
            metrics = runner.run()
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertTrue(Path(f"{output}.metrics.json").exists())


if __name__ == "__main__":
    unittest.main()

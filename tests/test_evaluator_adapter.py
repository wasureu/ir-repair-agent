import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from ir_repair_agent.evaluation.runner import (
    EvaluationRunner,
    HealingRequest,
)


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "code" / "evaluation_tir_gurobi_robust.py"
AUDITOR = ROOT / "code" / "constraint_auditor_v3.py"


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def load_evaluator():
    gurobipy = types.ModuleType("gurobipy")
    gurobipy.GRB = types.SimpleNamespace()
    openai = types.ModuleType("openai")
    openai.OpenAI = FakeOpenAI
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, **_kwargs: iterable
    auditor = types.ModuleType("constraint_auditor_v3")
    auditor.run_healing_pipeline = mock.Mock()
    auditor.repair_code = mock.Mock()
    auditor.audit_logic_hybrid = mock.Mock()
    auditor.RepairMemory = lambda: mock.Mock()

    with mock.patch.dict(
        sys.modules,
        {
            "gurobipy": gurobipy,
            "openai": openai,
            "tqdm": tqdm,
            "constraint_auditor_v3": auditor,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "evaluator_under_test",
            EVALUATOR,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def load_auditor():
    with mock.patch.object(sys, "path", [str(ROOT / "code"), *sys.path]):
        spec = importlib.util.spec_from_file_location(
            "constraint_auditor_default_test",
            AUDITOR,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class EvaluatorAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator()

    def make_evaluator(self):
        return self.module.SIRGurobiRobustEvaluator(
            input_path="input.jsonl",
            output_path="output.jsonl",
            api_key="test",
            max_workers=1,
        )

    def test_evaluator_constructs_runner_with_prompt_template(self):
        evaluator = self.make_evaluator()
        self.assertIsInstance(evaluator.runner, EvaluationRunner)
        self.assertEqual(
            evaluator.runner.prompt_template,
            evaluator.PROMPT_TEMPLATE,
        )

    def test_self_diagnosis_hint_fallback_defaults_enabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SELF_DIAGNOSE", None)
            module = load_auditor()
        self.assertTrue(module._SELF_DIAGNOSE)

    def test_healing_adapter_hides_answer_without_oracle_gate(self):
        evaluator = self.make_evaluator()
        pipeline = self.module.run_healing_pipeline
        pipeline.return_value = ("candidate", "PASS", 1, "easy")
        request = HealingRequest(
            problem_text="problem",
            initial_code="initial",
            initial_modeling_logic="",
            exec_output="",
            problem_class="toy",
            size=3,
            max_rounds_override=None,
            execution_error="",
            question_id="q1",
            oracle_value=None,
        )

        outcome = evaluator._heal(request)

        self.assertEqual(outcome.code, "candidate")
        kwargs = pipeline.call_args.kwargs
        self.assertIsNone(kwargs["ground_truth"])
        self.assertEqual(kwargs["execute_code_func"], evaluator.execute_code_with_gurobi)


if __name__ == "__main__":
    unittest.main()

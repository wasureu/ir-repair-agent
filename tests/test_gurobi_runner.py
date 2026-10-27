import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from ir_repair_agent.execution.gurobi_runner import GurobiRunner


class CountingRunner(GurobiRunner):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def _execute_uncached(self, code):
        self.calls += 1
        return (True, 1.0, ""), {"status_name": "OPTIMAL", "code": code}


class GurobiRunnerTests(unittest.TestCase):
    def test_result_and_evidence_cache(self):
        runner = CountingRunner()
        first = runner.execute("code")
        second = runner.execute("code")
        self.assertEqual(first, (True, 1.0, ""))
        self.assertEqual(second, first)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(runner.evidence("code")["status_name"], "OPTIMAL")

    def test_empty_code(self):
        runner = GurobiRunner()
        result = runner.execute("")
        self.assertFalse(result[0])
        self.assertIn("No code", result[2])
        self.assertEqual(runner.evidence(""), {})

    def test_real_execution_path_disposes_environment(self):
        events = []

        class FakeEnv:
            def setParam(self, name, value):
                events.append(("env_param", name, value))

            def start(self):
                events.append(("start",))

            def dispose(self):
                events.append(("dispose",))

        class FakeModel:
            Status = 2
            SolCount = 1
            ObjVal = 1.0

            def setParam(self, name, value):
                events.append(("model_param", name, value))

            def optimize(self):
                events.append(("optimize",))

        grb = SimpleNamespace(OPTIMAL=2, TIME_LIMIT=3)
        gurobipy = types.ModuleType("gurobipy")
        gurobipy.Env = lambda **_kwargs: FakeEnv()
        gurobipy.GRB = grb
        gurobipy.make_model = FakeModel

        evidence = SimpleNamespace(
            to_dict=lambda: {"status_name": "OPTIMAL"}
        )
        runner = GurobiRunner(time_limit=30)
        with (
            mock.patch.dict(sys.modules, {"gurobipy": gurobipy}),
            mock.patch(
                "ir_repair_agent.execution.gurobi_runner.collect_gurobi_evidence",
                return_value=evidence,
            ),
        ):
            result, collected = runner._execute_uncached(
                "model = gp.make_model()",
            )

        self.assertEqual(result, (True, 1.0, ""))
        self.assertEqual(collected, {"status_name": "OPTIMAL"})
        self.assertIn(("dispose",), events)


if __name__ == "__main__":
    unittest.main()


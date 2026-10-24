import unittest

from ir_repair_agent.audit.self_diagnosis import (
    build_self_diagnosis_hint,
    consensus_missing_constraints,
    verify_self_diagnosis,
)


class SelfDiagnosisTests(unittest.TestCase):
    def test_consensus_requires_all_runs(self):
        results = [
            {
                "problem_type": "inventory",
                "constraints": [
                    {
                        "name": "inventory balance",
                        "formula": "I_t = I_t-1 + production - demand",
                        "type": "TYPE_A",
                        "in_code": False,
                    }
                ],
            },
            {
                "problem_type": "inventory",
                "constraints": [
                    {
                        "name": "inventory balance",
                        "formula": "balance equation",
                        "type": "TYPE_A",
                        "in_code": False,
                    }
                ],
            },
            {
                "problem_type": "inventory",
                "constraints": [
                    {
                        "name": "inventory balance",
                        "formula": "stock flow equation",
                        "type": "TYPE_A",
                        "in_code": False,
                    }
                ],
            },
        ]
        missing = consensus_missing_constraints(results)
        self.assertEqual(len(missing), 1)
        self.assertIn("inventory balance", missing[0]["name"])
        hint = build_self_diagnosis_hint(results, missing)
        self.assertIn("inventory", hint)

    def test_one_disagreeing_run_breaks_consensus(self):
        base = {
            "problem_type": "inventory",
            "constraints": [
                {
                    "name": "inventory balance",
                    "formula": "stock flow",
                    "type": "TYPE_A",
                    "in_code": False,
                }
            ],
        }
        disagreeing = {
            "problem_type": "inventory",
            "constraints": [
                {
                    "name": "capacity",
                    "formula": "storage limit",
                    "type": "TYPE_A",
                    "in_code": False,
                }
            ],
        }
        self.assertEqual(
            consensus_missing_constraints([base, base, disagreeing]),
            [],
        )

    def test_verification_error_is_rejected(self):
        result = verify_self_diagnosis(
            "problem",
            "code",
            "hint",
            lambda _: {"error": "timeout"},
        )
        self.assertFalse(result["all_valid"])


if __name__ == "__main__":
    unittest.main()


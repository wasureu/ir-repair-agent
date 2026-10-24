import unittest

from ir_repair_agent.audit import equivalence_ok


class EquivalenceTests(unittest.TestCase):
    def test_mapping_category_passes(self):
        result = {
            "all_equivalent": True,
            "confidence": 0.9,
            "items": [
                {
                    "difference_index": 0,
                    "decision": "EQUIVALENT",
                    "category": "VARIABLE_RENAME",
                    "reason": "same quantity",
                    "mapping": "old_x -> new_x",
                }
            ],
        }
        self.assertTrue(equivalence_ok(result, [{"status": "MISMATCH"}]))

    def test_mapping_is_required(self):
        result = {
            "all_equivalent": True,
            "confidence": 0.9,
            "items": [
                {
                    "difference_index": 0,
                    "decision": "EQUIVALENT",
                    "category": "VARIABLE_RENAME",
                    "reason": "same quantity",
                }
            ],
        }
        self.assertFalse(equivalence_ok(result, [{"status": "MISMATCH"}]))

    def test_low_confidence_or_true_error_fails(self):
        differences = [{"status": "MISSING"}]
        self.assertFalse(
            equivalence_ok(
                {"all_equivalent": True, "confidence": 0.7, "items": []},
                differences,
            )
        )
        self.assertFalse(
            equivalence_ok(
                {
                    "all_equivalent": False,
                    "confidence": 0.99,
                    "items": [
                        {
                            "difference_index": 0,
                            "decision": "TRUE_ERROR",
                            "category": "REDUNDANT_CONSTRAINT",
                            "reason": "missing",
                        }
                    ],
                },
                differences,
            )
        )


if __name__ == "__main__":
    unittest.main()


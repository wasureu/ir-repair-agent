import unittest

from ir_repair_agent.memory_features import (
    constraint_skeleton,
    memory_error_pattern,
    structural_skeletons,
    structural_tokens,
)


class MemoryFeatureTests(unittest.TestCase):
    def test_error_pattern_prefers_last_exception(self):
        text = "ValueError: first\nTraceback\nGurobiError: second"
        self.assertEqual(memory_error_pattern(text), "second")

    def test_error_pattern_extracts_constraint_diff(self):
        text = (
            "FAIL [MISSING] balance constraint\n"
            "[MISMATCH] capacity coefficient"
        )
        result = memory_error_pattern(text)
        self.assertIn("MISSING: balance", result)
        self.assertIn("MISMATCH: capacity", result)

    def test_structural_skeleton_normalizes_names_and_numbers(self):
        first = constraint_skeleton("inv[t] == inv[t-1] + prod[t] - 10")
        second = constraint_skeleton("stock[t] == stock[t-1] + make[t] - 25")
        self.assertEqual(first, second)
        self.assertIn("[prev]", first)
        self.assertIn("[cur]", first)

    def test_structural_tokens_and_skeletons(self):
        pattern = (
            "expected=sum(x[t]) <= 10 "
            "| implemented=x[t] == x[t-1] + 2"
        )
        self.assertIn("SUM", structural_tokens(pattern))
        self.assertIn("RECUR", structural_tokens(pattern))
        self.assertEqual(len(structural_skeletons(pattern)), 2)


if __name__ == "__main__":
    unittest.main()


import unittest

from ir_repair_agent.contracts import CheckEvidence, ValidationResult
from ir_repair_agent.memory_admission import assess_memory_admission


class MemoryAdmissionTests(unittest.TestCase):
    def test_program_evidence_is_admitted(self):
        validation = ValidationResult(
            status="PASS",
            checks=(
                CheckEvidence("model_build", "PASS", source="program"),
                CheckEvidence("execution", "PASS", source="solver"),
            ),
        )
        decision = assess_memory_admission(
            validation,
            ("model_build", "execution"),
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.verified_checks, ("execution", "model_build"))

    def test_llm_only_evidence_is_rejected(self):
        validation = ValidationResult(
            status="PASS",
            checks=(
                CheckEvidence("model_build", "PASS", source="llm"),
            ),
        )
        decision = assess_memory_admission(validation, ("model_build",))
        self.assertFalse(decision.admitted)
        self.assertIn("non-program", decision.reason)

    def test_missing_or_failed_checks_are_reported(self):
        validation = ValidationResult(
            status="PASS",
            checks=(
                CheckEvidence("execution", "FAIL", source="solver"),
            ),
        )
        decision = assess_memory_admission(
            validation,
            ("model_build", "execution"),
        )
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.missing_checks, ("model_build",))
        self.assertIn("execution", decision.failed_checks)


if __name__ == "__main__":
    unittest.main()


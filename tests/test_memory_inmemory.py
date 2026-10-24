import unittest

from ir_repair_agent.memory_inmemory import RepairMemory


class InMemoryRepairMemoryTests(unittest.TestCase):
    def test_add_query_and_tier_limit(self):
        memory = RepairMemory(
            feature_match=2,
            min_similarity=0.0,
            min_confidence=0.0,
            tier_b_max=0,
            enabled=True,
        )
        memory.add(
            "inventory",
            "MISSING: balance inv[t] = inv[t-1] + production - demand",
            "old",
            "new",
            verified_by="exec_correct",
            confidence=0.9,
        )
        self.assertEqual(
            memory.query(
                "inventory",
                "MISSING: balance stock[t] = stock[t-1] + make - demand",
            ),
            [],
        )
        memory.add(
            "inventory",
            "MISSING: balance inv[t] = inv[t-1] + production - demand",
            "old2",
            "new2",
            verified_by="audit_equiv",
            confidence=0.9,
        )
        hits = memory.query(
            "inventory",
            "MISSING: balance stock[t] = stock[t-1] + make - demand",
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["verified_by"], "audit_equiv")

    def test_disabled_and_logger(self):
        events = []
        memory = RepairMemory(
            enabled=True,
            feature_match=0,
            min_similarity=0.0,
            store_logger=lambda *args: events.append(args),
        )
        memory.add("toy", "MISSING: balance", "old", "new")
        self.assertEqual(memory.size("toy"), 1)
        self.assertEqual(len(events), 1)

        disabled = RepairMemory(enabled=False)
        disabled.add("toy", "MISSING: balance", "old", "new")
        self.assertEqual(disabled.size("toy"), 0)

    def test_export_and_load(self):
        memory = RepairMemory(enabled=True, feature_match=0)
        memory.add("toy", "MISSING: balance", "old", "new")
        exported = memory.export_cases()
        restored = RepairMemory(enabled=True, feature_match=0)
        restored.load_cases(exported)
        self.assertEqual(restored.size("toy"), 1)


if __name__ == "__main__":
    unittest.main()


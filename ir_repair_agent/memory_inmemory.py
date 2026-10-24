"""Thread-safe in-process repair memory used by online ablation runs."""

from __future__ import annotations

import copy
import os
import re
import threading
from collections.abc import Callable
from typing import Any

from .memory_features import structural_skeletons, structural_tokens


StoreLogger = Callable[[str, int, str, float, str, str], None]


def normalize_class(problem_class: str) -> str:
    value = str(problem_class or "unknown").lower().replace(" ", "").replace("-", "_")
    aliases = {"binpacking": "bin_packing"}
    return aliases.get(value, value)


class RepairMemory:
    """Bucketed repair memory with deterministic feature matching."""

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "for", "to", "in", "on", "at", "by", "with", "from", "as",
        "and", "or", "not", "no", "all", "each", "every", "must", "should",
        "would", "could", "this", "that", "these", "those", "it", "its",
        "have", "has", "had", "do", "does", "did", "but", "if",
        "then", "else", "than", "into", "over", "under", "between", "about",
        "constraint", "constraints", "variable", "variables",
        "code", "model", "problem",
        "meal", "meals", "diet", "dietary", "nutrition", "nutritional",
        "food", "foods", "vitamin", "vitamins", "protein", "fiber",
        "carbohydrate", "carbohydrates", "calorie", "calories",
        "nutrient", "nutrients",
    }

    def __init__(
        self,
        max_cases: int = 200,
        top_k: int = 3,
        min_similarity: float = 0.1,
        snippet_limit: int = 1600,
        require_type_match: bool | None = None,
        feature_match: int | None = None,
        min_confidence: float | None = None,
        tier_b_max: int | None = None,
        enabled: bool | None = None,
        store_logger: StoreLogger | None = None,
    ):
        self._enabled = (
            not (os.environ.get("DISABLE_MEMORY", "0") == "1")
            if enabled is None
            else bool(enabled)
        )
        self.max_cases = max(1, int(max_cases))
        self.top_k = min(max(1, int(top_k)), 3)
        self.min_similarity = float(
            os.environ.get("MEMORY_MIN_SIMILARITY", str(min_similarity))
        )
        if require_type_match is None:
            require_type_match = os.environ.get("MEMORY_REQUIRE_TYPE", "1") == "1"
        self.require_type_match = bool(require_type_match)
        if feature_match is None:
            feature_match = int(os.environ.get("MEMORY_FEATURE_MATCH", "0"))
        self.feature_match = int(feature_match)
        if min_confidence is None:
            min_confidence = float(os.environ.get("MEMORY_MIN_CONFIDENCE", "0.5"))
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        if tier_b_max is None:
            tier_b_max = int(os.environ.get("MEMORY_TIER_B_MAX", "1"))
        self.tier_b_max = max(0, int(tier_b_max))
        self.snippet_limit = max(1, int(snippet_limit))
        self.store_logger = store_logger
        self._buckets: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", (text or "").lower())
        return {word for word in words if word not in cls._STOPWORDS}

    @staticmethod
    def _similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    @staticmethod
    def _error_type(pattern: str) -> str | None:
        match = re.match(
            r"(MISSING|EXTRA|MISMATCH)\b",
            (pattern or "").strip(),
            re.I,
        )
        return match.group(1).upper() if match else None

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = (text or "").strip()
        return text if len(text) <= limit else text[:limit] + "\n…[truncated]"

    def add(
        self,
        problem_class: str,
        error_pattern: str,
        original_snippet: str,
        repaired_snippet: str,
        verified_by: str = "audit_pass",
        confidence: float = 1.0,
        audit_diff_count: int = 0,
        objective_rel_error: float | None = None,
        question_id: str = "",
    ) -> None:
        if not self._enabled:
            return
        pattern = (error_pattern or "").strip()
        original = self._truncate(original_snippet, self.snippet_limit)
        repaired = self._truncate(repaired_snippet, self.snippet_limit)
        if not pattern or not original or not repaired:
            return
        key = normalize_class(problem_class)
        record = {
            "question_id": question_id,
            "problem_class": key,
            "error_pattern": pattern,
            "error_tokens": self._tokens(pattern),
            "error_type": self._error_type(pattern),
            "structural_tokens": structural_tokens(pattern),
            "structural_skeletons": structural_skeletons(pattern),
            "original_snippet": original,
            "repaired_snippet": repaired,
            "verified_by": verified_by,
            "confidence": min(max(float(confidence), 0.0), 1.0),
            "audit_diff_count": int(audit_diff_count),
            "objective_rel_error": objective_rel_error,
        }
        with self._lock:
            bucket = self._buckets.setdefault(key, [])
            bucket[:] = [
                old for old in bucket
                if not (
                    old["error_pattern"] == pattern
                    and old["original_snippet"] == original
                )
            ]
            bucket.append(record)
            if len(bucket) > self.max_cases:
                del bucket[: len(bucket) - self.max_cases]
            bucket_size = len(bucket)
        if self.store_logger is not None:
            self.store_logger(
                key,
                bucket_size,
                verified_by,
                record["confidence"],
                pattern,
                question_id,
            )

    def query(
        self,
        problem_class: str,
        error_pattern: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        k = min(max(1, int(top_k if top_k is not None else self.top_k)), 3)
        key = normalize_class(problem_class)
        query_type = self._error_type(error_pattern)
        if self.feature_match == 1:
            query_tokens = structural_tokens(error_pattern)
        elif self.feature_match == 2:
            query_tokens = structural_skeletons(error_pattern)
        else:
            query_tokens = self._tokens(error_pattern)
        if not query_tokens:
            return []
        with self._lock:
            bucket = list(self._buckets.get(key, []))
        if not bucket:
            return []
        scored = []
        for index, record in enumerate(bucket):
            if self.require_type_match and query_type is not None:
                record_type = record.get("error_type")
                if record_type is not None and record_type != query_type:
                    continue
            if self.feature_match == 1:
                record_tokens = record.get("structural_tokens")
            elif self.feature_match == 2:
                record_tokens = record.get("structural_skeletons")
            else:
                record_tokens = record["error_tokens"]
            scored.append((self._similarity(query_tokens, record_tokens), index, record))
        scored = [
            item for item in scored
            if item[0] >= self.min_similarity
            and float(item[2].get("confidence", 1.0)) >= self.min_confidence
        ]

        def tier(record: dict[str, Any]) -> int:
            verified = record.get("verified_by")
            if verified == "audit_pass":
                return 2
            if verified == "audit_equiv":
                return 1
            return 0

        scored.sort(
            key=lambda item: (
                tier(item[2]),
                item[0],
                float(item[2].get("confidence", 1.0)),
                item[1],
            ),
            reverse=True,
        )
        selected = []
        tier_b_used = 0
        for _, _, record in scored:
            if tier(record) == 0:
                if tier_b_used >= self.tier_b_max:
                    continue
                tier_b_used += 1
            selected.append(record)
            if len(selected) >= k:
                break
        return selected

    def size(self, problem_class: str) -> int:
        key = normalize_class(problem_class)
        with self._lock:
            return len(self._buckets.get(key, []))

    def export_cases(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return copy.deepcopy(self._buckets)

    def load_cases(self, cases: dict[str, list[dict[str, Any]]]) -> None:
        with self._lock:
            self._buckets = copy.deepcopy(cases)


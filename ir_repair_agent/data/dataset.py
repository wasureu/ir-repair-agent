"""Dataset loading and field normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProblemRecord:
    question: str
    answer: float | None
    problem_class: str
    size: int
    raw: dict[str, Any]


def question_text(item: dict[str, Any]) -> str | None:
    value = item.get("question") or item.get("augmented") or item.get("en_question")
    return str(value) if value is not None else None


def true_answer(item: dict[str, Any]) -> float | None:
    value = item.get("answer") or item.get("en_answer")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_problems(path: str | Path) -> list[ProblemRecord]:
    records: list[ProblemRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            question = question_text(item)
            if question is None:
                continue
            records.append(
                ProblemRecord(
                    question=question,
                    answer=true_answer(item),
                    problem_class=str(item.get("problem_class", "unknown")),
                    size=int(item.get("size", 0) or 0),
                    raw=item,
                )
            )
    return records

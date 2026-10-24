"""Frozen initial-code loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_frozen_codes(path: str | Path) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Frozen-code file must contain a JSON object")
    return {str(question): str(code or "") for question, code in data.items()}


def subset_frozen_codes(
    codes: dict[str, str],
    questions: Iterable[str],
) -> dict[str, str]:
    wanted = set(questions)
    return {question: codes[question] for question in wanted if question in codes}

"""Token-usage aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


USAGE_KEYS = ("prompt", "completion", "total", "calls")


def difference_usage(total: dict[str, Any], base: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(total.get(key, 0) or 0) - int(base.get(key, 0) or 0)
        for key in USAGE_KEYS
    }


def aggregate_token_usage(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    total_sum = {key: 0 for key in USAGE_KEYS}
    normal_sum = {key: 0 for key in USAGE_KEYS}
    healing_sum = {key: 0 for key in USAGE_KEYS}
    per_sample: list[int] = []

    for row in rows:
        total = row.get("token_usage", {}) or {}
        normal = row.get("token_usage_normal", {}) or {}
        healing = row.get("token_usage_healing", {}) or {}
        for key in USAGE_KEYS:
            total_sum[key] += int(total.get(key, 0) or 0)
            normal_sum[key] += int(normal.get(key, 0) or 0)
            healing_sum[key] += int(healing.get(key, 0) or 0)
        per_sample.append(int(total.get("total", 0) or 0))

    count = len(rows)
    median = sorted(per_sample)[count // 2] if count else 0
    average = (
        {key: round(total_sum[key] / count, 1) for key in USAGE_KEYS}
        if count
        else {key: 0 for key in USAGE_KEYS}
    )
    return {
        "n_samples": count,
        "total": total_sum,
        "avg_per_sample": average,
        "median_total_per_sample": median,
        "normal": normal_sum,
        "healing": healing_sum,
    }

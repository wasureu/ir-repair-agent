"""Answer scoring for optimization outputs."""

from __future__ import annotations


def relative_error(predicted: float | None, truth: float | None) -> float | None:
    if predicted is None or truth is None:
        return None
    return abs(float(predicted) - float(truth)) / (abs(float(truth)) + 1e-6)


def is_correct(
    predicted: float | None,
    truth: float | None,
    tolerance: float = 1e-3,
) -> bool:
    error = relative_error(predicted, truth)
    return error is not None and error < float(tolerance)

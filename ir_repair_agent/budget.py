from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable


class BudgetExceeded(RuntimeError):
    def __init__(self, resource: str, used: float, limit: float) -> None:
        self.resource = resource
        self.used = used
        self.limit = limit
        super().__init__(f"{resource} budget exceeded: used={used:g}, limit={limit:g}")


@dataclass(frozen=True)
class BudgetLimits:
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
    max_llm_calls: int | None = None
    max_wall_seconds: float | None = None
    max_solver_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_prompt_tokens", self.max_prompt_tokens),
            ("max_completion_tokens", self.max_completion_tokens),
            ("max_total_tokens", self.max_total_tokens),
            ("max_llm_calls", self.max_llm_calls),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("max_wall_seconds", self.max_wall_seconds),
            ("max_solver_seconds", self.max_solver_seconds),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_llm_calls": self.max_llm_calls,
            "max_wall_seconds": self.max_wall_seconds,
            "max_solver_seconds": self.max_solver_seconds,
        }


@dataclass(frozen=True)
class BudgetSnapshot:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_calls: int
    solver_seconds: float
    wall_seconds: float
    stage_usage: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "solver_seconds": self.solver_seconds,
            "wall_seconds": self.wall_seconds,
            "stage_usage": self.stage_usage,
        }


class BudgetLedger:
    """Thread-safe shared budget ledger for every model call in one sample.

    The same ledger should be shared by initial generation, audit, repair, and
    solver execution. This prevents a treatment arm from receiving extra repair
    calls and makes token/time comparisons meaningful.
    """

    def __init__(
        self,
        limits: BudgetLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = limits
        self._clock = clock
        self._lock = threading.RLock()
        self._started_at: float | None = None
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._llm_calls = 0
        self._solver_seconds = 0.0
        self._stage_usage: dict[str, dict[str, float | int]] = {}

    def start(self) -> None:
        with self._lock:
            if self._started_at is None:
                self._started_at = self._clock()

    def _wall_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._started_at)

    def _assert_within_wall_budget(self) -> None:
        elapsed = self._wall_seconds()
        limit = self.limits.max_wall_seconds
        if limit is not None and elapsed > limit:
            raise BudgetExceeded("wall_seconds", elapsed, limit)

    def ensure_available(self) -> None:
        """Call before a tool invocation so an already exhausted run stops."""
        with self._lock:
            self.start()
            self._assert_within_wall_budget()
            max_calls = self.limits.max_llm_calls
            if max_calls is not None and self._llm_calls >= max_calls:
                raise BudgetExceeded("llm_calls", self._llm_calls + 1, max_calls)
            max_total = self.limits.max_total_tokens
            if max_total is not None and self._total_tokens >= max_total:
                raise BudgetExceeded("total_tokens", self._total_tokens + 1, max_total)

    def _stage(self, stage: str) -> dict[str, float | int]:
        if stage not in self._stage_usage:
            self._stage_usage[stage] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "llm_calls": 0,
                "solver_seconds": 0.0,
            }
        return self._stage_usage[stage]

    def charge_llm(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        stage: str = "llm",
        calls: int = 1,
    ) -> BudgetSnapshot:
        if min(prompt_tokens, completion_tokens, calls) < 0:
            raise ValueError("token and call counts must be non-negative")
        total_tokens = prompt_tokens + completion_tokens
        with self._lock:
            self.start()
            stage_usage = self._stage(stage)
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._total_tokens += total_tokens
            self._llm_calls += calls
            stage_usage["prompt_tokens"] += prompt_tokens
            stage_usage["completion_tokens"] += completion_tokens
            stage_usage["total_tokens"] += total_tokens
            stage_usage["llm_calls"] += calls
            snapshot = self.snapshot()

        self._raise_if_exceeded(
            "prompt_tokens",
            self._prompt_tokens,
            self.limits.max_prompt_tokens,
        )
        self._raise_if_exceeded(
            "completion_tokens",
            self._completion_tokens,
            self.limits.max_completion_tokens,
        )
        self._raise_if_exceeded("total_tokens", self._total_tokens, self.limits.max_total_tokens)
        self._raise_if_exceeded("llm_calls", self._llm_calls, self.limits.max_llm_calls)
        self._raise_if_exceeded("wall_seconds", snapshot.wall_seconds, self.limits.max_wall_seconds)
        return snapshot

    def charge_solver(
        self,
        seconds: float,
        *,
        stage: str = "solver",
    ) -> BudgetSnapshot:
        if seconds < 0:
            raise ValueError("solver seconds must be non-negative")
        with self._lock:
            self.start()
            self._solver_seconds += float(seconds)
            self._stage(stage)["solver_seconds"] += float(seconds)
            snapshot = self.snapshot()
        self._raise_if_exceeded(
            "solver_seconds",
            self._solver_seconds,
            self.limits.max_solver_seconds,
        )
        self._raise_if_exceeded("wall_seconds", snapshot.wall_seconds, self.limits.max_wall_seconds)
        return snapshot

    @staticmethod
    def _raise_if_exceeded(resource: str, used: float, limit: float | int | None) -> None:
        if limit is not None and used > limit:
            raise BudgetExceeded(resource, used, float(limit))

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                total_tokens=self._total_tokens,
                llm_calls=self._llm_calls,
                solver_seconds=self._solver_seconds,
                wall_seconds=self._wall_seconds(),
                stage_usage={
                    name: dict(values) for name, values in self._stage_usage.items()
                },
            )

    def remaining(self) -> dict[str, float | int | None]:
        with self._lock:
            elapsed = self._wall_seconds()
            return {
                "prompt_tokens": self._remaining(
                    self.limits.max_prompt_tokens, self._prompt_tokens
                ),
                "completion_tokens": self._remaining(
                    self.limits.max_completion_tokens, self._completion_tokens
                ),
                "total_tokens": self._remaining(
                    self.limits.max_total_tokens, self._total_tokens
                ),
                "llm_calls": self._remaining(self.limits.max_llm_calls, self._llm_calls),
                "wall_seconds": self._remaining(
                    self.limits.max_wall_seconds, elapsed
                ),
                "solver_seconds": self._remaining(
                    self.limits.max_solver_seconds, self._solver_seconds
                ),
            }

    @staticmethod
    def _remaining(limit: float | int | None, used: float) -> float | int | None:
        if limit is None:
            return None
        return max(0, limit - used)

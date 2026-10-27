"""Problem-level evaluation orchestration.

The runner owns prompt-to-code evaluation, frozen initial-code reuse,
incremental JSONL output, and metrics. Solver and repair implementations are
injected so this module remains deterministic and testable without network or
Gurobi dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from ..data.dataset import ProblemRecord, load_problems, question_text, true_answer
from .metrics import aggregate_token_usage, difference_usage
from .scoring import is_correct


USAGE_KEYS = ("prompt", "completion", "total", "calls")
EXECUTION_RESULT = tuple[bool, float | None, str]


def question_id(question: str) -> str:
    return hashlib.md5((question or "").encode("utf-8")).hexdigest()[:12]


def difficulty_for_size(size: int) -> str:
    if size <= 5:
        return "easy"
    if size <= 10:
        return "medium"
    return "hard"


class UsageSnapshot(Protocol):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenUsageTracker:
    """Thread-local LLM usage for concurrent problem evaluations."""

    def __init__(self) -> None:
        self._local = threading.local()

    @staticmethod
    def _empty() -> dict[str, int]:
        return {key: 0 for key in USAGE_KEYS}

    def reset(self) -> None:
        self._local.total = self._empty()

    def record(self, usage: UsageSnapshot | None) -> None:
        if usage is None:
            return
        if not hasattr(self._local, "total"):
            self._local.total = self._empty()
        print(
            f" [tokens] prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} "
            f"total={usage.total_tokens}"
        )
        self._local.total["prompt"] += int(usage.prompt_tokens)
        self._local.total["completion"] += int(usage.completion_tokens)
        self._local.total["total"] += int(usage.total_tokens)
        self._local.total["calls"] += 1

    def snapshot(self) -> dict[str, int]:
        total = getattr(self._local, "total", None)
        return dict(total or self._empty())


@dataclass(frozen=True)
class HealingRequest:
    problem_text: str
    initial_code: str
    initial_modeling_logic: str
    exec_output: str
    problem_class: str
    size: int
    max_rounds_override: int | None
    execution_error: str
    question_id: str
    oracle_value: float | None


@dataclass(frozen=True)
class HealingOutcome:
    code: str
    audit_report: Any
    rounds_used: int
    difficulty: str


ProblemLoader = Callable[[str | Path], list[ProblemRecord]]
LLMCall = Callable[..., str]
CodeExtractor = Callable[[str], str]
CodeExecutor = Callable[..., EXECUTION_RESULT]
HealingCallback = Callable[[HealingRequest], HealingOutcome]


class EvaluationRunner:
    def __init__(
        self,
        *,
        input_path: str | Path,
        output_path: str | Path,
        model: str,
        max_workers: int,
        prompt_template: str,
        usage_tracker: TokenUsageTracker,
        call_llm: LLMCall,
        extract_code: CodeExtractor,
        execute_code: CodeExecutor,
        seed: str = "",
        enable_healing: bool = True,
        initial_codes: dict[str, str] | None = None,
        dump_initial_codes: str | Path | None = None,
        healing_callback: HealingCallback | None = None,
        problem_loader: ProblemLoader = load_problems,
        oracle_gate_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.model = str(model)
        self.max_workers = int(max_workers)
        self.prompt_template = str(prompt_template)
        self.usage_tracker = usage_tracker
        self.call_llm = call_llm
        self.extract_code = extract_code
        self.execute_code = execute_code
        self.seed = str(seed)
        self.enable_healing = bool(enable_healing)
        self.initial_codes = dict(initial_codes or {})
        self.has_initial_codes = initial_codes is not None
        self.dump_initial_codes_path = (
            Path(dump_initial_codes) if dump_initial_codes else None
        )
        self.healing_callback = healing_callback
        self.problem_loader = problem_loader
        self.oracle_gate_enabled = oracle_gate_enabled or (
            lambda: os.environ.get("ORACLE_GATE", "0") == "1"
        )

    def _base_result(
        self,
        *,
        item: dict[str, Any],
        question: str,
        code: str,
        response: str,
        execution_error: str,
        predicted_answer: float | None,
        true_value: float | None,
        started_at: float,
        normal_usage: dict[str, int],
        healing_usage: dict[str, int],
    ) -> dict[str, Any]:
        size = int(item.get("size", 0) or 0)
        usage = self.usage_tracker.snapshot()
        return {
            "question": question,
            "question_id": question_id(question),
            "seed": self.seed,
            "true_answer": true_value,
            "predicted_answer": predicted_answer,
            "tag": int(is_correct(predicted_answer, true_value, tolerance=1e-3)),
            "response": response,
            "code": code,
            "execution_error": execution_error,
            "problem_class": item.get("problem_class", "unknown"),
            "size": size,
            "difficulty": difficulty_for_size(size),
            "token_usage": usage,
            "token_usage_normal": dict(normal_usage),
            "token_usage_healing": dict(healing_usage),
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }

    def evaluate_one(self, problem_obj: dict[str, Any]) -> dict[str, Any] | None:
        started_at = time.perf_counter()
        self.usage_tracker.reset()
        item = deepcopy(problem_obj)
        question = question_text(item)
        if question is None:
            return None

        print(">>> [evaluate_one] preparing LLM call")
        prompt = self.prompt_template.format(question=question)
        response = self.call_llm(prompt)
        print(f">>> [evaluate_one] LLM returned, length={len(response)}")
        code = self.extract_code(response)
        print(f">>> [evaluate_one] extracted code, length={len(code)}")

        predicted_answer, execution_error = self._execute(code)
        normal_usage = self.usage_tracker.snapshot()
        return self._base_result(
            item=item,
            question=question,
            code=code,
            response=response,
            execution_error=execution_error,
            predicted_answer=predicted_answer,
            true_value=true_answer(item),
            started_at=started_at,
            normal_usage=normal_usage,
            healing_usage={key: 0 for key in USAGE_KEYS},
        )

    def initial_result(
        self,
        problem_obj: dict[str, Any],
    ) -> dict[str, Any] | None:
        started_at = time.perf_counter()
        item = deepcopy(problem_obj)
        question = question_text(item)
        if question is None:
            return None
        if not self.has_initial_codes or question not in self.initial_codes:
            return self.evaluate_one(problem_obj)

        self.usage_tracker.reset()
        code = self.initial_codes[question] or ""
        predicted_answer, execution_error = self._execute(code)
        normal_usage = self.usage_tracker.snapshot()
        return self._base_result(
            item=item,
            question=question,
            code=code,
            response="",
            execution_error=execution_error,
            predicted_answer=predicted_answer,
            true_value=true_answer(item),
            started_at=started_at,
            normal_usage=normal_usage,
            healing_usage={key: 0 for key in USAGE_KEYS},
        )

    def evaluate_one_with_healing(
        self,
        problem_obj: dict[str, Any],
        max_rounds_override: int | None = None,
    ) -> dict[str, Any] | None:
        print(">>> entering evaluate_one_with_healing")
        started_at = time.perf_counter()
        if not self.enable_healing:
            return self.initial_result(problem_obj)

        normal_result = self.initial_result(problem_obj)
        if not normal_result:
            return None
        normal_usage = dict(normal_result.get("token_usage", {}) or {})

        item = deepcopy(problem_obj)
        question = question_text(item)
        if question is None:
            return normal_result
        initial_code = normal_result.get("code")
        if not initial_code:
            return normal_result
        if self.healing_callback is None:
            raise RuntimeError("healing is enabled but no healing callback is configured")

        true_value = true_answer(item)
        request = HealingRequest(
            problem_text=question,
            initial_code=initial_code,
            initial_modeling_logic="",
            exec_output=normal_result.get("response", ""),
            problem_class=str(item.get("problem_class", "unknown")),
            size=int(item.get("size", 0) or 0),
            max_rounds_override=max_rounds_override,
            execution_error=normal_result.get("execution_error", "") or "",
            question_id=question_id(question),
            oracle_value=(
                true_value if self.oracle_gate_enabled() else None
            ),
        )
        outcome = self.healing_callback(request)

        success, predicted_answer, final_error = self.execute_code(outcome.code)
        initial_success, initial_predicted, _ = self.execute_code(initial_code)
        final_code = outcome.code
        audit_report = outcome.audit_report
        if not success or predicted_answer is None:
            fallback_reason = "repair failed to execute"
            print(f" [warn] falling back to initial code: {fallback_reason}")
            if initial_success and initial_predicted is not None:
                final_code = initial_code
                predicted_answer = initial_predicted
                audit_report = f"FALLBACK: {fallback_reason}"
                final_error = ""
            else:
                predicted_answer = None

        total_usage = self.usage_tracker.snapshot()
        healing_usage = difference_usage(total_usage, normal_usage)
        result = self._base_result(
            item=item,
            question=question,
            code=final_code,
            response=normal_result.get("response", ""),
            execution_error="",
            predicted_answer=predicted_answer,
            true_value=true_value,
            started_at=started_at,
            normal_usage=normal_usage,
            healing_usage=healing_usage,
        )
        result.update(
            {
                "oracle_gate": self.oracle_gate_enabled(),
                "audit_result": audit_report,
                "rounds_used": int(outcome.rounds_used),
                "original_code": initial_code,
            }
        )
        return result

    def _execute(self, code: str) -> tuple[float | None, str]:
        if not code:
            return None, ""
        print(">>> executing code")
        success, predicted_answer, execution_error = self.execute_code(code)
        print(
            f">>> execution finished success={success}, "
            f"predicted={predicted_answer}"
        )
        return (predicted_answer if success else None), execution_error

    def dump_initial_codes(self) -> None:
        if self.dump_initial_codes_path is None:
            raise ValueError("dump_initial_codes path is not configured")
        started_at = time.perf_counter()
        self.usage_tracker.reset()
        problems = [record.raw for record in self.problem_loader(self.input_path)]
        codes: dict[str, str] = {}
        details: list[dict[str, Any]] = []
        for index, item in enumerate(problems, start=1):
            question = question_text(item)
            if question is None:
                continue
            before = self.usage_tracker.snapshot()
            question_started = time.perf_counter()
            response = self.call_llm(
                self.prompt_template.format(question=question)
            )
            code = self.extract_code(response)
            codes[question] = code
            after = self.usage_tracker.snapshot()
            details.append(
                {
                    "question": question,
                    "code_len": len(code),
                    "elapsed_seconds": round(
                        time.perf_counter() - question_started,
                        3,
                    ),
                    "token_usage": difference_usage(after, before),
                }
            )
            print(
                f"[dump] {index}/{len(problems)} "
                f"code_len={len(code)}"
            )

        target = self.dump_initial_codes_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(codes, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        elapsed = time.perf_counter() - started_at
        count = max(1, len(codes))
        usage = self.usage_tracker.snapshot()
        metrics = {
            "n_samples": len(codes),
            "elapsed_seconds_total": round(elapsed, 1),
            "elapsed_per_problem_mean": round(elapsed / count, 2),
            "token_usage": {
                "total": usage,
                "avg_per_sample": {
                    key: round(float(usage.get(key, 0)) / count, 1)
                    for key in USAGE_KEYS
                },
            },
            "source": "dump_initial_codes",
            "input_path": str(self.input_path),
            "model": self.model,
        }
        metrics_path = Path(f"{target}.metrics.json")
        details_path = Path(f"{target}.details.jsonl")
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with details_path.open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            "initial-code tokens: "
            f"prompt={usage.get('prompt', 0):,} "
            f"completion={usage.get('completion', 0):,} "
            f"total={usage.get('total', 0):,} "
            f"calls={usage.get('calls', 0):,}"
        )
        print(
            f"initial-code elapsed: {elapsed:.1f}s, "
            f"mean={elapsed / count:.2f}s"
        )
        print(f"initial codes saved to {target} ({len(codes)} items)")
        print(f"dump metrics saved to {metrics_path}")
        print(f"dump details saved to {details_path}")

    def run(self) -> dict[str, Any] | None:
        started_at = time.perf_counter()
        if self.dump_initial_codes_path is not None:
            self.dump_initial_codes()
            return None
        problems = [record.raw for record in self.problem_loader(self.input_path)]
        print(
            "Start SIR (Gurobi) Robust Evaluation, "
            f"{len(problems)} questions in total"
        )
        print(f" - Healing: {'ON' if self.enable_healing else 'OFF'}")
        print(" - Writing: incremental after each sample")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.evaluate_one_with_healing, problem)
                for problem in problems
            ]
            try:
                from tqdm import tqdm

                completed = tqdm(
                    as_completed(futures),
                    total=len(problems),
                    desc="Evaluating",
                )
            except ImportError:
                completed = as_completed(futures)
            for future in completed:
                try:
                    result = future.result()
                except Exception as error:
                    print(f"[warn] task failed; skipping sample: {error}")
                    traceback.print_exc()
                    continue
                if result:
                    with self.output_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")

        results = []
        with self.output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    results.append(json.loads(line))
        accuracy = (
            sum(int(row["tag"]) for row in results) / len(results)
            if results
            else 0.0
        )
        usage = aggregate_token_usage(results)
        print(f"Evaluation finished | accuracy rate: {accuracy:.2%}")
        print("Token usage summary:")
        print(f"  total tokens       : {usage['total']['total']:,}")
        print(f"  avg per problem    : {int(usage['avg_per_sample']['total']):,}")
        print(
            "  median per problem : "
            f"{usage['median_total_per_sample']:,}"
        )
        print(
            "  prompt / completion / calls: "
            f"{usage['total']['prompt']:,} / "
            f"{usage['total']['completion']:,} / "
            f"{usage['total']['calls']:,}"
        )
        print(f"  healing tokens     : {usage['healing']['total']:,}")
        print(f"results saved to: {self.output_path}")

        elapsed_values = sorted(
            float(row.get("elapsed_seconds", 0) or 0)
            for row in results
            if row.get("elapsed_seconds")
        )
        elapsed_mean = (
            round(sum(elapsed_values) / len(elapsed_values), 2)
            if elapsed_values
            else 0.0
        )
        elapsed_median = (
            elapsed_values[len(elapsed_values) // 2]
            if elapsed_values
            else 0.0
        )
        elapsed_total = round(time.perf_counter() - started_at, 1)
        print(
            "elapsed: "
            f"total={elapsed_total}s  "
            f"per-problem mean={elapsed_mean}s  "
            f"median={elapsed_median}s"
        )

        metrics = {
            "n_samples": usage["n_samples"],
            "accuracy": accuracy,
            "seed": self.seed,
            "oracle_gate": self.oracle_gate_enabled(),
            "elapsed_seconds_total": elapsed_total,
            "elapsed_per_problem_mean": elapsed_mean,
            "elapsed_per_problem_median": elapsed_median,
            "token_usage": {
                "total": usage["total"],
                "avg_per_sample": usage["avg_per_sample"],
                "median_total_per_sample": usage["median_total_per_sample"],
                "normal": usage["normal"],
                "healing": usage["healing"],
            },
        }
        metrics_path = Path(f"{self.output_path}.metrics.json")
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"metrics saved to: {metrics_path}")
        return metrics

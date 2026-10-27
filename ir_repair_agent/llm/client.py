"""OpenAI-compatible chat client used by the evaluator."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class LLMClient:
    SYSTEM_PROMPT = (
        "You are an expert in operations research and Gurobi solver. "
        "Always generate correct, runnable Python code."
    )

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = 32768,
        usage_callback: Callable[[Any], None] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = int(max_tokens)
        self.usage_callback = usage_callback

    def _request(
        self,
        prompt: str,
        *,
        temperature: float,
        timeout_s: float,
    ) -> Any:
        try:
            import httpx
            timeout = httpx.Timeout(
                timeout_s,
                connect=10.0,
                read=timeout_s,
                write=10.0,
            )
        except ImportError:
            timeout = timeout_s
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=self.max_tokens,
            timeout=timeout,
        )

    def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        timeout_s: float = 300.0,
    ) -> str:
        try:
            response = self._request(
                prompt,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            self._record_usage(response)
            message = response.choices[0].message
            content = message.content or ""
            reasoning = getattr(message, "reasoning_content", None) or ""
            if not content and reasoning:
                print(f"[warn] content is empty; reasoning_len={len(reasoning)}")
                if "```python" in reasoning or "import gurobipy" in reasoning:
                    print("[fallback] extracting code from reasoning")
                    return reasoning
            if content:
                return content

            print("[warn] empty response; retrying in 3 seconds...")
            time.sleep(3)
            try:
                retry = self._request(
                    prompt,
                    temperature=temperature,
                    timeout_s=timeout_s,
                )
                self._record_usage(retry)
                return retry.choices[0].message.content or ""
            except Exception as retry_error:
                print(f"[error] retry failed: {retry_error}")
                return ""
        except Exception as error:
            print(f"[error] LLM API error: {error}")
            return ""

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None and self.usage_callback is not None:
            self.usage_callback(usage)


"""Cached Gurobi execution runner with structured evidence."""

from __future__ import annotations

import hashlib
import threading
import traceback
from typing import Any

from ..execution_evidence import collect_gurobi_evidence


class GurobiRunner:
    def __init__(self, time_limit: int = 100):
        self.time_limit = int(time_limit)
        self._result_cache: dict[str, tuple[bool, float | None, str]] = {}
        self._evidence_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _hash(code: str) -> str:
        return hashlib.md5(code.encode("utf-8")).hexdigest()

    def execute(
        self,
        code: str,
        use_cache: bool = True,
    ) -> tuple[bool, float | None, str]:
        if not code:
            return False, None, "No code provided."
        code_hash = self._hash(code)
        if use_cache:
            with self._lock:
                cached = self._result_cache.get(code_hash)
            if cached is not None:
                return cached
        result, evidence = self._execute_uncached(code)
        if use_cache:
            with self._lock:
                self._result_cache[code_hash] = result
                self._evidence_cache[code_hash] = evidence
        return result

    def evidence(self, code: str) -> dict[str, Any]:
        if not code:
            return {}
        with self._lock:
            return dict(self._evidence_cache.get(self._hash(code)) or {})

    def clear_cache(self) -> None:
        with self._lock:
            self._result_cache.clear()
            self._evidence_cache.clear()

    def _execute_uncached(
        self,
        code: str,
    ) -> tuple[tuple[bool, float | None, str], dict[str, Any]]:
        import gurobipy as gp
        from gurobipy import GRB

        env = None
        result: tuple[bool, float | None, str] = (
            False,
            None,
            "Unknown error",
        )
        evidence: dict[str, Any] = {}
        try:
            env = gp.Env(empty=True)
            env.setParam("TimeLimit", self.time_limit)
            env.start()
            namespace = {
                "gp": gp,
                "GRB": GRB,
                "env": env,
                "__builtins__": __builtins__,
            }
            exec(code, namespace, namespace)
            if "model" not in namespace:
                result = (False, None, "No 'model' object found in namespace.")
            else:
                model = namespace["model"]
                model.setParam("TimeLimit", self.time_limit)
                model.optimize()
                try:
                    evidence = collect_gurobi_evidence(model).to_dict()
                except Exception as evidence_error:
                    evidence = {
                        "status": "evidence_error",
                        "error": str(evidence_error)[:500],
                    }
                if model.Status == GRB.OPTIMAL:
                    result = (True, model.ObjVal, "")
                elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
                    result = (True, model.ObjVal, "")
                elif model.Status == GRB.TIME_LIMIT:
                    result = (False, None, "Time limit reached, no feasible solution.")
                else:
                    result = (False, None, f"Gurobi status: {model.Status}")
        except Exception as error:
            print(f"[error] code execution failed: {error}")
            result = (False, None, traceback.format_exc())
        finally:
            if env is not None:
                try:
                    env.dispose()
                except Exception:
                    pass
        return result, evidence


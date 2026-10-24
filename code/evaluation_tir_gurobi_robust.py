#!/usr/bin/env python3
"""
SIR (Solver-Integrated Reasoning) Evaluation Pipeline with Gurobi
Enhanced with Robust Self-Healing Mechanism (4-Layer Protection)

Features:
1. Problem signature extraction (prevent problem swapping)
2. Constraint-level repair (not full code rewrite)
3. Enhanced constraint audit (magnitude & direction check)
4. Fallback mechanism (return original on any anomaly)
5. Incremental writing (results saved immediately after each sample)
6. Gurobi time limit: 100 seconds

Usage:
    python pipeline/evaluation_tir_gurobi_robust.py --input_path test.jsonl --output_path ./results.json --model deepseek-chat
"""
import httpx
import os
import json
import argparse
import re
import time
import traceback
import threading
import hashlib
import math
import sys
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ir_repair_agent.data import load_frozen_codes, load_problems
from ir_repair_agent.evaluation import (
    aggregate_token_usage,
    difference_usage,
    is_correct,
)
from ir_repair_agent.llm import extract_numeric_answer, extract_python_code

# Import black-box healing modules
try:
    from constraint_auditor_v3 import (
        run_healing_pipeline,
        repair_code,
        audit_logic_hybrid,
        RepairMemory,
    )
    # 如果仍需要旧版本的函数（用于其他模块），可以保留旧的导入
    # 但 v3 版本不包含 extract_constraints_from_text 等函数
    HEALING_AVAILABLE = True
except ImportError:
    HEALING_AVAILABLE = False
    print("⚠️ constraint_auditor_v3 not found, healing disabled")

# 文件写入锁（多线程安全）
file_write_lock = threading.Lock()


def _get_difficulty(size):
    if size <= 5:
        return "easy"
    elif size <= 10:
        return "medium"
    else:
        return "hard"
    
class SIRGurobiRobustEvaluator:
    def __init__(self, input_path, output_path, api_key,
                 api_base="https://api.deepseek.com/v1", model="deepseek-chat",
                 max_workers=16, enable_healing=True,
                 initial_codes_file=None, dump_initial_codes=None, seed=""):
        self.input_path = input_path
        self.output_path = output_path
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.max_workers = max_workers
        self.seed = str(seed or os.environ.get("SEED", ""))
        self.enable_healing = enable_healing and HEALING_AVAILABLE
        self.client = OpenAI(api_key=api_key, base_url=api_base)  
        # 执行结果缓存：{code_hash: (success, objective, error)}
        # 避免同一份代码重复执行
        import threading
        self._exec_cache = {}
        self._exec_cache_lock = threading.Lock()
        self._exec_evidence = {}
        self._exec_evidence_lock = threading.Lock()
        # 跨题记忆：所有 worker 共享同一实例（内部有锁，线程安全）
        self.memory = RepairMemory()
        # 冻结初版：让两个消融臂从同一份初版开始修复，隔离初版生成方差
        self.initial_codes_file = initial_codes_file
        self.initial_codes = None
        if initial_codes_file:
            self.initial_codes = load_frozen_codes(initial_codes_file)
        self.dump_initial_codes = dump_initial_codes

    @staticmethod
    def _question_id(question: str) -> str:
        """稳定的题目 ID：与打分无关，只用于日志关联。"""
        return hashlib.md5((question or "").encode("utf-8")).hexdigest()[:12]
        # Prompt template for Gurobi
        self.prompt_template = """
You are an operations research and Gurobi solver expert. Below is an operations research question:

{question}

Carefully analyze the problem to identify the core elements such as Decision Variables, Objective Function, and Constraints. Determine whether the variable is an integer or a continuous variable.

Build a mathematical model and corresponding Gurobi Python code.

**Required code structure**:
```python
import gurobipy as gp
from gurobipy import GRB

# Create model
model = gp.Model("optimization_model")

# Add variables
x = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x")

# Set objective
model.setObjective(expression, GRB.MAXIMIZE)

# Add constraints
model.addConstr(expression <= rhs, name="constr_name")

# Optimize
model.optimize()

# Print result
if model.Status == GRB.OPTIMAL:
    print(f"Optimal value: {{model.ObjVal}}")
else:
    print("No optimal solution found")
Important:

Use the exact data from the problem statement.

Make sure the model variable is named 'model'.

Output only the Python code block, no explanations.
**IMPORTANT**:
- Set `model.setParam('TimeLimit', 100)` BEFORE calling `model.optimize()`.
- Use `model = gp.Model(env=env)` if an `env` variable is provided.
"""

    def _record_usage(self, usage):
        """把一次 LLM 调用的 token 用量累加到当前线程的统计里。"""
        if usage is None:
            return
        print(
            f" [tokens] prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} "
            f"total={usage.total_tokens}"
        )
        if not hasattr(self, '_thread_local_tokens'):
            self._thread_local_tokens = threading.local()
        if not hasattr(self._thread_local_tokens, 'total'):
            self._thread_local_tokens.total = {
                "prompt": 0, "completion": 0, "total": 0, "calls": 0
            }
        self._thread_local_tokens.total["prompt"] += usage.prompt_tokens
        self._thread_local_tokens.total["completion"] += usage.completion_tokens
        self._thread_local_tokens.total["total"] += usage.total_tokens
        self._thread_local_tokens.total["calls"] += 1

    def call_llm(self, prompt, temperature=0.3, timeout_s=300.0):
        """Call LLM API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in operations research and Gurobi solver. Always generate correct, runnable Python code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=32768,
                timeout=httpx.Timeout(timeout_s, connect=10.0, read=timeout_s, write=10.0),
            )
            self._record_usage(response.usage)
            
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, 'reasoning_content', None) or ""

            if not content and reasoning:
                print(f" ⚠️ content 为空，reasoning_len={len(reasoning)}")
                if '```python' in reasoning or 'import gurobipy' in reasoning:
                    print(f"    → 从 reasoning 提取代码")
                    return reasoning

            # content 和 reasoning 都空 → 重试一次
            if not content:
                print(f" ⚠️ 空响应，3 秒后重试...")
                import time
                time.sleep(3)
                try:
                    response2 = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert in operations research and Gurobi solver. Always generate correct, runnable Python code."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=32768,
                        timeout=httpx.Timeout(timeout_s, connect=10.0, read=timeout_s, write=10.0),
                    )
                    self._record_usage(response2.usage)
                    return response2.choices[0].message.content or ""
                except Exception as e2:
                    print(f"重试失败: {e2}")
                    return ""

            return content    
        
        except Exception as e:
            print(f"LLM API error: {e}")
            return ""

    def extract_code(self, response):
        return extract_python_code(response)

    def extract_answer_from_response(self, response):
        """Extract numerical answer from response"""
        return extract_numeric_answer(response)

    def execute_code_with_gurobi(self, code, use_cache=True):
        """..."""
        if not code:
            return False, None, "No code provided."

        # 查缓存（用 md5 避免 hash 碰撞）
        code_hash = None
        if use_cache:
            import hashlib
            code_hash = hashlib.md5(code.encode('utf-8')).hexdigest()
            with self._exec_cache_lock:
                if code_hash in self._exec_cache:
                    return self._exec_cache[code_hash]

        env = None
        result = None
        evidence = None
        try:
            env = gp.Env(empty=True)
            env.setParam('TimeLimit', 100)
            env.start()

            namespace = {
                'gp': gp,
                'GRB': GRB,
                'env': env,
                '__builtins__': __builtins__,
            }

            exec(code, namespace, namespace)

            if 'model' not in namespace:
                result = (False, None, "No 'model' object found in namespace.")
            else:
                model = namespace['model']
                model.setParam('TimeLimit', 100)
                model.optimize()
                try:
                    evidence = self._collect_model_evidence(model)
                except Exception as evidence_exc:
                    evidence = {
                        "status": "evidence_error",
                        "error": str(evidence_exc)[:500],
                    }

                if model.Status == GRB.OPTIMAL:
                    result = (True, model.ObjVal, "")
                elif model.Status == GRB.TIME_LIMIT:
                    if model.SolCount > 0:
                        result = (True, model.ObjVal, "")
                    else:
                        result = (False, None, "Time limit reached, no feasible solution.")
                else:
                    result = (False, None, f"Gurobi status: {model.Status}")
        except Exception as e:
            print(f"❌ 代码执行错误: {e}")
            result = (False, None, traceback.format_exc())
        finally:
            if env is not None:
                try:
                    env.dispose()
                except Exception:
                    pass

        # 写缓存并返回
        if use_cache and result is not None:
            with self._exec_cache_lock:
                self._exec_cache[code_hash] = result
            if evidence is not None:
                with self._exec_evidence_lock:
                    self._exec_evidence[code_hash] = evidence
        return result if result is not None else (False, None, "Unknown error")

    @staticmethod
    def _safe_float(value):
        """把 Gurobi 数值转成 JSON 安全的 Python 值。"""
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "-inf" if number < 0 else "inf"
        return number

    @staticmethod
    def _collect_model_evidence(model) -> dict:
        """读取求解器级证据：状态、解、约束违反与 IIS。"""
        status_names = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
        }
        status = int(model.Status)
        evidence = {
            "status": status,
            "status_name": status_names.get(status, f"STATUS_{status}"),
            "sol_count": int(model.SolCount),
            "objective": float(model.ObjVal) if model.SolCount > 0 else None,
            "objective_sense": getattr(model, "ModelSense", None),
            "variables": [],
            "constraints": [],
            "max_constr_vio": 0.0,
            "sum_constr_vio": 0.0,
            "infeasible_constraints": 0,
            "model_constr_vio": None,
            "iis": None,
        }
        try:
            evidence["model_constr_vio"] = SIRGurobiRobustEvaluator._safe_float(
                model.getAttr(GRB.Attr.ConstrVio)
            )
        except Exception:
            evidence["model_constr_vio"] = None

        for var in model.getVars():
            if len(evidence["variables"]) >= 500:
                break
            try:
                x_value = float(var.X)
            except Exception:
                x_value = None
            evidence["variables"].append(
                {
                    "name": var.VarName,
                    "vtype": str(var.VType),
                    "lb": SIRGurobiRobustEvaluator._safe_float(var.LB),
                    "ub": SIRGurobiRobustEvaluator._safe_float(var.UB),
                    "x": SIRGurobiRobustEvaluator._safe_float(x_value),
                }
            )

        constraints = model.getConstrs()
        vio_values = []
        try:
            vio_values = model.getAttr(GRB.Attr.ConstrVio, constraints)
        except Exception:
            vio_values = []
        try:
            slack_values = model.getAttr(GRB.Attr.Slack, constraints)
        except Exception:
            slack_values = []

        for index, constr in enumerate(constraints):
            if len(evidence["constraints"]) >= 1000:
                break
            vio = (
                SIRGurobiRobustEvaluator._safe_float(vio_values[index])
                if index < len(vio_values) and vio_values[index] is not None
                else None
            )
            slack = (
                SIRGurobiRobustEvaluator._safe_float(slack_values[index])
                if index < len(slack_values) and slack_values[index] is not None
                else None
            )
            rhs_value = SIRGurobiRobustEvaluator._safe_float(constr.RHS)
            if vio is None and slack is not None:
                sense = str(constr.Sense)
                lhs = None
                try:
                    lhs = SIRGurobiRobustEvaluator._safe_float(
                        model.getRow(constr).getValue()
                    )
                except Exception:
                    lhs = None
                if lhs is not None and rhs_value is not None:
                    if sense == "<":
                        vio = max(0.0, lhs - rhs_value)
                    elif sense == ">":
                        vio = max(0.0, rhs_value - lhs)
                    elif sense == "=":
                        vio = abs(lhs - rhs_value)
            if vio is not None and vio > 0.0:
                evidence["infeasible_constraints"] += 1
                evidence["max_constr_vio"] = max(evidence["max_constr_vio"], vio)
                evidence["sum_constr_vio"] += vio
            evidence["constraints"].append(
                {
                    "name": constr.ConstrName,
                    "sense": str(constr.Sense),
                    "rhs": rhs_value,
                    "slack": slack,
                    "vio": vio,
                }
            )

        if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            try:
                model.computeIIS()
                iis_constraints = [c.ConstrName for c in model.getConstrs()
                                   if getattr(c, "IISConstr", 0) > 0]
                iis_vars = [v.VarName for v in model.getVars()
                            if getattr(v, "IISLB", 0) > 0
                            or getattr(v, "IISUB", 0) > 0]
                evidence["iis"] = {
                    "constraints": iis_constraints[:200],
                    "variables": iis_vars[:200],
                }
            except Exception:
                evidence["iis"] = {"error": "computeIIS failed"}

        return evidence

    def get_execution_evidence(self, code: str) -> dict:
        """返回最近一次执行该代码收集的求解器证据。"""
        if not code:
            return {}
        import hashlib
        code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()
        with self._exec_evidence_lock:
            return dict(self._exec_evidence.get(code_hash) or {})
                    
    def evaluate_one(self, problem_obj):
        """Evaluate single problem (no healing)"""
        _t0 = time.perf_counter()
        import threading
        if not hasattr(self, '_thread_local_tokens'):
            self._thread_local_tokens = threading.local()
        self._thread_local_tokens.total = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
        
        item = deepcopy(problem_obj)
        true_answer = item.get("answer") or item.get("en_answer")
        question = item.get("question") or item.get("augmented") or item.get("en_question")
        if question is None:
            return None
        
        print(">>> [evaluate_one] 准备调用 LLM") 
        prompt = self.prompt_template.format(question=question)

        response = self.call_llm(prompt)
        print(f">>> [evaluate_one] LLM 返回，长度={len(response)}") 

        code = self.extract_code(response)
        print(f">>> [evaluate_one] 提取代码，长度={len(code)}") 

        pred_value = None
        exec_error = ""
        if code:
            print(">>> [evaluate_one] 开始执行代码")
            success, pred_value, exec_error = self.execute_code_with_gurobi(code)
            print(f">>> [evaluate_one] 执行结束 success={success}, pred={pred_value}")
            if not success:
                pred_value = None

        try:
            true_value = float(true_answer)
        except Exception:
            true_value = None

        tag = int(is_correct(pred_value, true_value, tolerance=1e-3))

        return {
            "question": question,
            "question_id": self._question_id(question),
            "seed": self.seed,
            "true_answer": true_value,
            "predicted_answer": pred_value,
            "tag": tag,
            "response": response,
            "code": code,
            "execution_error": exec_error,
            "problem_class": item.get('problem_class', 'unknown'),
            "size": item.get('size', 0),
            "difficulty": _get_difficulty(item.get('size', 0)),
            "token_usage": dict(getattr(self._thread_local_tokens, 'total', {}) or {}),
            "token_usage_normal": dict(getattr(self._thread_local_tokens, 'total', {}) or {}),
            "token_usage_healing": {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
            "elapsed_seconds": round(time.perf_counter() - _t0, 3),
        }

    def _initial_result(self, problem_obj):
        """返回与 evaluate_one 同构的初版结果；命中冻结初版时跳过重新生成。"""
        _t0 = time.perf_counter()
        item = deepcopy(problem_obj)
        question = item.get("question") or item.get("augmented") or item.get("en_question")
        if question is None:
            return None
        if self.initial_codes is None or question not in self.initial_codes:
            return self.evaluate_one(problem_obj)

        if not hasattr(self, '_thread_local_tokens'):
            self._thread_local_tokens = threading.local()
        self._thread_local_tokens.total = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}

        code = self.initial_codes[question] or ""
        pred_value = None
        exec_error = ""
        if code:
            success, pred_value, exec_error = self.execute_code_with_gurobi(code)
            if not success:
                pred_value = None

        try:
            true_value = float(item.get("answer") or item.get("en_answer"))
        except Exception:
            true_value = None

        tag = int(is_correct(pred_value, true_value, tolerance=1e-3))

        return {
            "question": question,
            "question_id": self._question_id(question),
            "seed": self.seed,
            "true_answer": true_value,
            "predicted_answer": pred_value,
            "tag": tag,
            "response": "",
            "code": code,
            "execution_error": exec_error,
            "problem_class": item.get('problem_class', 'unknown'),
            "size": item.get('size', 0),
            "difficulty": _get_difficulty(item.get('size', 0)),
            "token_usage": dict(getattr(self._thread_local_tokens, 'total', {}) or {}),
            "token_usage_normal": dict(getattr(self._thread_local_tokens, 'total', {}) or {}),
            "token_usage_healing": {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
            "elapsed_seconds": round(time.perf_counter() - _t0, 3),
        }

    def evaluate_one_with_healing(self, problem_obj, max_rounds_override=None):
        print(">>> 进入 evaluate_one_with_healing")
        _t0 = time.perf_counter()
        if not self.enable_healing:
            return self._initial_result(problem_obj)

    # 先跑普通 SIR
        normal_result = self._initial_result(problem_obj)
        normal_usage = dict(normal_result.get("token_usage", {}) or {}) if normal_result else {}

        if not normal_result:
            return None

    # 提取必要信息
        item = deepcopy(problem_obj)
        question = item.get("question") or item.get("augmented") or item.get("en_question")
        true_answer = item.get("answer") or item.get("en_answer")
        if question is None:
            return normal_result

        initial_code = normal_result.get('code')
        if not initial_code:
            return normal_result

    # 需要获取 initial_modeling_logic（来自 evaluate_one 的 response 中的建模逻辑部分）
    # 由于 evaluate_one 没有返回 modeling_logic，我们需要从 normal_result 中提取
    # 简单起见，我们直接让 LLM 重新生成一次 modeling logic，或者在 evaluate_one 中返回它。
    # 为了简化，这里我们暂时使用空字符串，但更好的做法是修改 evaluate_one 返回建模逻辑。
    # 但为了快速使用新脚本，我们可以用 audit_logic_hybrid 中的 phase2 来提取，它不需要 initial_modeling_logic。
    # 实际上 audit_logic_hybrid 中 initial_modeling_logic 是可选的，如果为空，它会忽略。
    # 所以我们传入 ""。

        initial_modeling_logic = ""
        initial_execution_error = normal_result.get('execution_error', '') or ''

    # 获取初始执行输出（从 normal_result 中获取 response 作为执行日志）
        exec_output = normal_result.get('response', '')

    # 提取问题类型和 size
        problem_class = item.get('problem_class', 'unknown')
        size = item.get('size', 0)
        try:
            true_value = float(true_answer)
        except (ValueError, TypeError):
            true_value = None

        question_id = self._question_id(question)
        # 正式实验：修复循环不接触标准答案；仅当显式 ORACLE_GATE=1 时
        # 才把答案传入，用于诊断性 oracle 上界。
        oracle_value = (
            true_value if os.environ.get("ORACLE_GATE", "0") == "1" else None
        )

    # 调用统一修复流水线
                # 调用统一修复流水线
        final_code, audit_report, rounds_used, difficulty = run_healing_pipeline(
            problem_text=question,
            initial_code=initial_code,
            initial_modeling_logic=initial_modeling_logic,
            exec_output=exec_output,
            problem_class=problem_class,
            size=size,
            call_llm_func=self.call_llm,
            max_rounds_override=max_rounds_override,
            execute_code_func=self.execute_code_with_gurobi,   # ← 新增：传入执行器
            ground_truth=oracle_value,                          # 仅 oracle 诊断时非空
            objective_tolerance=1e-3,  
            initial_execution_error=initial_execution_error,
            memory=self.memory,   # ← 新增：跨题记忆
            question_id=question_id,
            execution_evidence_func=self.get_execution_evidence,
        )
    # 执行修复后的代码
        success, pred_value, _ = self.execute_code_with_gurobi(final_code)
        
        # 先获取初始代码的目标值（用于对比）
        init_success, init_pred, _ = self.execute_code_with_gurobi(initial_code)
        
        # 判断是否需要回退
        need_fallback = False
        fallback_reason = ""
        
        if not success or pred_value is None:
            need_fallback = True
            fallback_reason = "repair failed to execute"
        
        if need_fallback:
            print(f" ⚠️ 回退到初始代码: {fallback_reason}")
            if init_success and init_pred is not None:
                final_code = initial_code
                pred_value = init_pred
                audit_report = f"FALLBACK: {fallback_reason}"
            else:
                pred_value = None
            
        tag = int(is_correct(pred_value, true_value, tolerance=1e-3))

        return {
            "question": question,
            "question_id": question_id,
            "seed": self.seed,
            "oracle_gate": os.environ.get("ORACLE_GATE", "0") == "1",
            "true_answer": true_value,
            "predicted_answer": pred_value,
            "tag": tag,
            "response": normal_result.get('response', ''),
            "code": final_code,
            "audit_result": audit_report,
            "rounds_used": rounds_used,
            "difficulty": difficulty,
            "original_code": initial_code,
            "problem_class": problem_class,
            "size": size,
            "difficulty": _get_difficulty(size),
            "execution_error": "",
            "token_usage": dict(getattr(self._thread_local_tokens, 'total', {}) or {}),
            "token_usage_normal": normal_usage,
            "token_usage_healing": self._sub_usage(
                getattr(self._thread_local_tokens, 'total', {}) or {}, normal_usage
            ),
            "elapsed_seconds": round(time.perf_counter() - _t0, 3),
        }

    @staticmethod
    def _sub_usage(total, base):
        """total - base（逐字段相减），用于拆分 healing 阶段的 token。"""
        return difference_usage(total, base)

    @staticmethod
    def _aggregate_token_usage(results):
        """聚合所有样本的 token 消耗统计。"""
        return aggregate_token_usage(results)

    def _dump_initial_codes(self):
        """为每道题生成一次初版代码并保存 {question: code}（不执行、不修复）。"""
        _dump_t0 = time.perf_counter()
        if not hasattr(self, "_thread_local_tokens"):
            self._thread_local_tokens = threading.local()
        self._thread_local_tokens.total = {
            "prompt": 0, "completion": 0, "total": 0, "calls": 0
        }
        problems = [record.raw for record in load_problems(self.input_path)]
        out = {}
        details = []
        for i, item in enumerate(problems, 1):
            question = item.get("question") or item.get("augmented") or item.get("en_question")
            if question is None:
                continue
            before = dict(self._thread_local_tokens.total)
            q_t0 = time.perf_counter()
            response = self.call_llm(self.prompt_template.format(question=question))
            code = self.extract_code(response)
            out[question] = code
            after = dict(self._thread_local_tokens.total)
            delta = {
                k: int(after.get(k, 0)) - int(before.get(k, 0))
                for k in ("prompt", "completion", "total", "calls")
            }
            details.append({
                "question": question,
                "code_len": len(code),
                "elapsed_seconds": round(time.perf_counter() - q_t0, 3),
                "token_usage": delta,
            })
            print(f"[dump] {i}/{len(problems)} code_len={len(out[question])}")
        with open(self.dump_initial_codes, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

        elapsed = time.perf_counter() - _dump_t0
        n = max(1, len(out))
        usage = dict(getattr(self._thread_local_tokens, "total", {}) or {})
        metrics = {
            "n_samples": len(out),
            "elapsed_seconds_total": round(elapsed, 1),
            "elapsed_per_problem_mean": round(elapsed / n, 2),
            "token_usage": {
                "total": usage,
                "avg_per_sample": {
                    k: round(float(usage.get(k, 0)) / n, 1)
                    for k in ("prompt", "completion", "total", "calls")
                },
            },
            "source": "dump_initial_codes",
            "input_path": self.input_path,
            "model": self.model,
        }
        metrics_path = self.dump_initial_codes + ".metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        details_path = self.dump_initial_codes + ".details.jsonl"
        with open(details_path, "w", encoding="utf-8") as f:
            for row in details:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            "initial-code tokens: "
            f"prompt={usage.get('prompt', 0):,} "
            f"completion={usage.get('completion', 0):,} "
            f"total={usage.get('total', 0):,} "
            f"calls={usage.get('calls', 0):,}"
        )
        print(f"initial-code elapsed: {elapsed:.1f}s, mean={elapsed / n:.2f}s")
        print(f"initial codes saved to {self.dump_initial_codes} ({len(out)} items)")
        print(f"dump metrics saved to {metrics_path}")
        print(f"dump details saved to {details_path}")

    def run(self):
        """运行批量评估（边运行边写入）"""
        _run_t0 = time.perf_counter()
        if self.dump_initial_codes:
            self._dump_initial_codes()
            return
        problems = [record.raw for record in load_problems(self.input_path)]

        print(f"Start SIR (Gurobi) Robust Evaluation, {len(problems)} questions in total")
        print(f" - Healing: {'ON' if self.enable_healing else 'OFF'}")
        print(f" - Writing: incremental (每完成一个样本立即写入)")
        print(f" - Gurobi TimeLimit: 100 seconds")

        # 清空输出文件（从头开始）
        with open(self.output_path, "w", encoding="utf-8") as f:
            pass

        # 统一走 evaluate_one_with_healing：它在 enable_healing=False 时会转 _initial_result，
        # 从而让“无修复”也复用冻结初版（没有冻结文件时才退回 evaluate_one）。
        evaluate_fn = self.evaluate_one_with_healing
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(evaluate_fn, p) for p in problems]
            for future in tqdm(as_completed(futures), total=len(problems), desc="Evaluating"):
                try:
                    res = future.result()
                except Exception as e:
                    print(f"⚠️ 任务异常，跳过: {e}")
                    traceback.print_exc()
                    continue

                if res:
                    # 立即追加写入（线程安全）
                    with file_write_lock:
                        with open(self.output_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    completed += 1

        # 统计准确率
        results = []
        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        acc = sum([r["tag"] for r in results]) / len(results) if results else 0

        # 统计 token 消耗
        ts = self._aggregate_token_usage(results)
        print("Evaluation finished | accuracy rate: {:.2%}".format(acc))
        print("Token usage summary:")
        print("  total tokens       : {:,}".format(ts["total"]["total"]))
        print("  avg per problem    : {:,}".format(int(ts["avg_per_sample"]["total"])))
        print("  median per problem : {:,}".format(ts["median_total_per_sample"]))
        print("  prompt / completion / calls: {:,} / {:,} / {:,}".format(
            ts["total"]["prompt"], ts["total"]["completion"], ts["total"]["calls"]))
        print("  healing tokens     : {:,}".format(ts["healing"]["total"]))
        print(f"results saved to: {self.output_path}")

        el = sorted(r.get("elapsed_seconds", 0) for r in results if r.get("elapsed_seconds"))
        el_mean = round(sum(el) / len(el), 2) if el else 0.0
        el_median = el[len(el) // 2] if el else 0.0
        elapsed_total = round(time.perf_counter() - _run_t0, 1)
        print("elapsed: total={}s  per-problem mean={}s  median={}s".format(
            elapsed_total, el_mean, el_median))

        # 写入指标汇总文件，便于有/无记忆消融对比
        metrics = {
            "n_samples": ts["n_samples"],
            "accuracy": acc,
            "seed": self.seed,
            "oracle_gate": os.environ.get("ORACLE_GATE", "0") == "1",
            "elapsed_seconds_total": elapsed_total,
            "elapsed_per_problem_mean": el_mean,
            "elapsed_per_problem_median": el_median,
            "token_usage": {
                "total": ts["total"],
                "avg_per_sample": ts["avg_per_sample"],
                "median_total_per_sample": ts["median_total_per_sample"],
                "normal": ts["normal"],
                "healing": ts["healing"],
            },
        }
        metrics_path = self.output_path + ".metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="SIR (Gurobi) Robust Evaluation Pipeline")

    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL")

    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                        help="API key (or set env OPENAI_API_KEY)")
    parser.add_argument("--api_base", type=str, default="https://api.deepseek.com/v1", help="Base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name")

    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子标识（写入结果与日志，便于复现）")
    parser.add_argument("--no-healing", action="store_true", help="Disable self-healing")
    parser.add_argument("--dump_initial_codes", type=str, default=None,
                        help="生成初版代码并保存到该 JSON 文件（跳过评测）")
    parser.add_argument("--initial_codes_file", type=str, default=None,
                        help="从该 JSON 文件加载初版代码，修复阶段跳过重新生成")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("api_key is empty. Pass --api_key or set env OPENAI_API_KEY")

    if args.seed is not None:
        os.environ["SEED"] = str(args.seed)

    evaluator = SIRGurobiRobustEvaluator(
        input_path=args.input_path,
        output_path=args.output_path,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        max_workers=args.max_workers,
        enable_healing=not args.no_healing,
        initial_codes_file=args.initial_codes_file,
        dump_initial_codes=args.dump_initial_codes,
        seed=os.environ.get("SEED", ""),
    )
    evaluator.run()


if __name__ == "__main__":
    main()

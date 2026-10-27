# Migration Acceptance Checklist

These checks must run in the WSL environment that has Gurobi and the live LLM
credentials.

## 1. Package and evaluator imports

```bash
cd /mnt/c/Users/lenovo/Documents/LLM_Optimization_Modeling
export PYTHONPATH="$PWD:$PWD/code"

python3 -m unittest discover -s tests -q

python3 - <<'PY'
from ir_repair_agent.evaluation import EvaluationRunner
from ir_repair_agent.execution import GurobiRunner
from ir_repair_agent.llm import LLMClient
import evaluation_tir_gurobi_robust

print("migration imports OK")
print(EvaluationRunner, GurobiRunner, LLMClient)
PY
```

## 2. Real Gurobi execution evidence

```bash
python3 - <<'PY'
from ir_repair_agent.execution import GurobiRunner

code = """
import gurobipy as gp
from gurobipy import GRB

model = gp.Model(env=env)
x = model.addVar(lb=0, name="x")
model.addConstr(x >= 1, name="at_least_one")
model.setObjective(x, GRB.MINIMIZE)
model.optimize()
"""

runner = GurobiRunner(time_limit=30)
success, objective, error = runner.execute(code, use_cache=True)
evidence = runner.evidence(code)

assert success, error
assert abs(objective - 1.0) < 1e-6, objective
assert evidence["status_name"] == "OPTIMAL", evidence
assert evidence["max_constr_vio"] == 0.0, evidence
print("GurobiRunner smoke OK", objective, evidence["status_name"])
PY
```

## 3. Six-case golden set

```bash
cd ~/OPTEngine
STAMP=$(date +%Y%m%d_%H%M%S)

SEED=1 \
AUDIT_EQUIV=1 \
SELF_DIAGNOSE=1 \
DISABLE_MEMORY=1 \
GATE_LOG=1 \
GATE_LOG_PATH="logs/industryor_cases6_gate_${STAMP}.jsonl" \
python3 -u /mnt/c/Users/lenovo/Documents/LLM_Optimization_Modeling/code/evaluation_tir_gurobi_robust.py \
  --input_path logs/industryor_cases6.jsonl \
  --output_path "logs/industryor_cases6_${STAMP}.jsonl" \
  --initial_codes_file logs/industryor_cases6_initial_codes.json \
  --model deepseek-v4-flash \
  --max_workers 2 \
  > "logs/industryor_cases6_${STAMP}.out.txt" 2>&1

PILOT="logs/industryor_cases6_${STAMP}.jsonl"
python3 /mnt/c/Users/lenovo/Documents/Codex/2026-10-11/llm-gurobi-self-healing-1-constraint-7/outputs/compare_industryor_pilot.py \
  --pilot "$PILOT"
```

Acceptance target for the golden set:

- all six problems produce output rows;
- `fixed >= 1`;
- `broke = 0`;
- `pilot oracle_gate values: [False]`.

Only after these checks pass should the temporary legacy evaluator methods be
removed.

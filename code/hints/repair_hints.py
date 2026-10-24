"""修复阶段的领域知识提示词（按问题类型）"""

REPAIR_HINTS = {
    "bin_packing": """
COMMON ERRORS FOR BIN PACKING:
1. Missing linking constraint x[i,j] <= y[j]. Without it, items can be
   assigned to a bin that is not opened.
2. Capacity must use y[j] as a multiplier:
   sum_i weight[i] * x[i,j] <= capacity * y[j].
3. Each item must be in exactly one bin: sum_j x[i,j] == 1.
4. Objective must be sum_j y[j] (minimize), not sum over items.
""",
    "inventory": """
COMMON ERRORS FOR INVENTORY:
1. Balance reversed. Correct: I[t] == I[t-1] + order[t] - demand[t].
   Common mistake: I[t] == I[t-1] + demand[t] - order[t].
2. Initial inventory I[0] is a data parameter, not a decision variable.
3. ONLY IF the problem explicitly states a fixed ordering cost, add binary 
   y[t] and linking order[t] <= M * y[t]. 
   If NOT mentioned, do NOT add them.
4. Missing non-negativity on I[t] or order[t].
""",
    "jobshop": """
COMMON ERRORS FOR JOB SHOP:
1. Precedence reversed. Correct: start[next_op] >= finish[prev_op].
2. Disjunctive constraints missing on machines (no two operations overlap).
   Big-M form: start[j] >= finish[i] - M * (1 - y[i,j]).
3. Each operation assigned to exactly one machine:
   sum_k y[op,k] == 1.
4. Makespan Cmax must be >= finish time of every operation.
""",
    "knapsack": """
COMMON ERRORS FOR KNAPSACK:
1. Capacity direction must be <=, not >=.
2. Variable domain must be GRB.BINARY, not GRB.CONTINUOUS.
3. Objective maximizes total value; check coefficient signs.
4. Multi-dimensional knapsack: one capacity constraint per dimension.
""",
    "networkflow": """
COMMON ERRORS FOR NETWORK FLOW:
1. Flow conservation at intermediate nodes: inflow == outflow.
2. Capacity bound: 0 <= flow[e] <= capacity[e].
3. Source/sink balance: not both zero.
4. Arc direction must match variable definition.
""",
    "pollution": """
COMMON ERRORS FOR POLLUTION:
1. Emission cap direction: sum emission <= cap, not >=.
2. Abatement variables must be non-negative and bounded by max abatement.
3. Objective: minimize total treatment cost.
4. Multi-period: missing carry-over of accumulated pollution.
""",
    "portfolio": """
COMMON ERRORS FOR PORTFOLIO:
1. Budget: sum w[i] == 1 (or <= 1) missing.
2. Risk term (variance) must use covariance matrix:
   variance = sum_i sum_j w[i] * w[j] * cov[i,j].
3. Return target must be >=, not <=.
4. If short-selling disallowed, missing bounds w[i] >= 0.
""",
    "production": """
COMMON ERRORS FOR PRODUCTION:
1. Demand satisfaction direction:
   production[t] + inventory[t-1] >= demand[t].
2. Missing setup linking: production[t] <= M * y[t].
3. Inventory balance uses previous period's inventory.
4. Non-negativity of production, inventory, and setup binary.
""",
    "tsp": """
COMMON ERRORS FOR TSP:
1. Missing subtour elimination.
   MTZ form for i != j, i,j != depot:
       u[i] - u[j] + n * x[i,j] <= n - 1
   Common mistake: coefficient 1 instead of n.
2. Degree constraints must be equality == 2, not <= 2:
   sum_j x[i,j] + sum_j x[j,i] == 2.
3. u[i] bounds must be 1 <= u[i] <= n-1 for i != depot.
4. If x is symmetric (only i<j), ensure constraints account for both directions.
""",
    "transportation": """
COMMON ERRORS FOR TRANSPORTATION:
1. Supply: sum_j x[i,j] <= supply[i].
2. Demand: sum_i x[i,j] >= demand[j].
3. If balanced, both can be equalities.
4. Missing non-negativity x[i,j] >= 0.
""",
}

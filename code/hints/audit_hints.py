"""审计阶段的领域知识提示词（按问题类型）"""

AUDIT_HINTS = {
    "bin_packing": """
Focus especially on:
- Each item assigned to exactly one bin: sum_j x[i,j] == 1
- Bin capacity: sum_i weight[i] * x[i,j] <= capacity * y[j]
- Linking constraint: x[i,j] <= y[j]
- y[j] binary (bin used or not)
- Objective: minimize number of bins: sum_j y[j]
- If the problem does not explicitly state the number of available bins/pallets,
  use any sufficiently large upper bound (e.g., number of items n).
  Do NOT mark this as uncertainty — it does not affect the optimal value.
""",

    "inventory": """
Focus especially on:
- Inventory balance: I[t] == I[t-1] + order[t] - demand[t]
- Initial inventory I[0] defined from data
- Holding cost and ordering cost in objective
- Non-negativity of inventory and order quantities
- Order capacity or fixed-charge linking: 
  CHECK ONLY IF the problem explicitly mentions a fixed ordering cost 
  or an order quantity range. 
  Otherwise, do NOT expect them and do NOT report them as MISSING.
""",

    "jobshop": """
Focus especially on:
- Each operation assigned to exactly one machine
- Precedence constraints: start of successor >= finish of predecessor
- No overlap on the same machine (disjunctive, big-M)
- Machine capacity = 1 operation at a time
- Makespan definition: Cmax >= finish of all operations
""",
    "knapsack": """
Focus especially on:
- Capacity constraint direction: sum weight[i] * x[i] <= capacity
- Binary variable domains (GRB.BINARY)
- Objective: maximize total value
- Multi-dimensional knapsack: one capacity constraint per dimension
""",
    "networkflow": """
Focus especially on:
- Flow conservation at intermediate nodes: inflow == outflow
- Capacity: 0 <= flow[e] <= capacity[e]
- Source/sink balance
- Continuous or integer flow domains
""",
    "pollution": """
Focus especially on:
- Emission limits: sum of pollutants <= allowed cap
- Abatement / treatment efficiency constraints
- Cost minimization of treatment
- Non-negativity of emission and treatment variables
- If multi-period, temporal accumulation balance
""",
    "portfolio": """
Focus especially on:
- Budget constraint: sum of weights == 1 (or <= 1)
- Non-negativity or bounds on weights if short-selling forbidden
- Return target: sum expected_return[i] * w[i] >= target
- Risk term (variance) in objective if mean-variance formulation
- Cardinality or integer lot constraints if present
""",
    "production": """
Focus especially on:
- Demand satisfaction: production[t] + inventory[t-1] >= demand[t]
- Production capacity limits
- Inventory balance across periods
- Setup binary and linking: production[t] <= M * y[t]
- Cost terms: production, setup, holding
""",
    "tsp": """
Focus especially on:
- Degree constraints: sum_j x[i,j] + sum_j x[j,i] == 2 for every node
- Subtour elimination: MTZ (u[i] - u[j] + n*x[i,j] <= n-1) or DFJ
- Binary edge variables
- Missing subtour elimination is a hard FAIL, not a warning
""",
    "transportation": """
Focus especially on:
- Supply constraint: sum_j x[i,j] <= supply[i]
- Demand constraint: sum_i x[i,j] >= demand[j]
- Non-negativity of flow variables
- Objective: minimize total transportation cost
""",
}

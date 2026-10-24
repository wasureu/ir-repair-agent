"""MAMO 数据集专用 hints（基于对修正版 30 道错题的实际分析）"""

MAMO_COMPLEXLP_AUDIT = """
This is an MILP problem from MAMO. We have observed that **73% of remaining 
errors in this dataset** are variable type mismatches — the code uses 
GRB.CONTINUOUS where GRB.INTEGER is required.

**VARIABLE TYPE CHECK (HIGHEST PRIORITY)**:

When reading the CODE:
- Report the LITERAL vtype string from each addVar / addVars call:
  - GRB.CONTINUOUS → report as "continuous"
  - GRB.INTEGER    → report as "integer"
  - GRB.BINARY     → report as "binary"
- Do NOT interpret or guess. Read the exact string.

When reading the PROBLEM TEXT:
- If the problem asks for a COUNT of discrete items (people, vehicles, units,
  boxes, packages, machines, workers) → the variable MUST be integer
- If the problem asks for a yes/no decision (select, choose, whether, open) 
  → the variable MUST be binary
- Only amounts, fractions, weights, and budgets may be continuous

**REPORTING RULES**:
- ALWAYS report IMPLEMENTED as what the CODE literally says
- ALWAYS report EXPECTED as what the PROBLEM requires
- Do NOT swap these two

**COMMON FAILURE MODE IN THIS DATASET**:
The code models "how many X" questions with GRB.CONTINUOUS. When the problem 
asks for a count, the code should use GRB.INTEGER. Report a MISMATCH if it 
does not.

Do NOT skip variable domain comparison. It is the most important check.
"""

MAMO_COMPLEXLP_REPAIR = """
**THIS DATASET'S #1 ERROR: wrong variable type (73% of remaining errors)**

If the audit report indicates variables are CONTINUOUS but should be INTEGER
(because the problem asks for a count of discrete items):

STEP 1: Find every `model.addVar(...)` or `model.addVars(...)` in the code 
        that represents a count of discrete items (items, people, vehicles, 
        packages, etc.)

STEP 2: For those variables, change `vtype=GRB.CONTINUOUS` to 
        `vtype=GRB.INTEGER`. If BINARY semantics apply (yes/no), use 
        `GRB.BINARY`.

STEP 3: Keep all other variables, constraints, objective, and coefficients 
        UNCHANGED.

STEP 4: Verify that the bounds (lb/ub) are still consistent with the new 
        integer type.

**DO NOT**:
- Change the problem structure
- Change constraint coefficients or directions
- Change the objective function
- Add or remove constraints

**ONLY** change vtype from CONTINUOUS to INTEGER (or BINARY) where the 
problem semantics require discrete counts.
"""

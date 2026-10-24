"""Canonical solver-level execution evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def safe_float(value: Any) -> float | str | None:
    """Return a finite Python float or a JSON-safe representation."""
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


@dataclass(frozen=True)
class VariableEvidence:
    name: str
    vtype: str
    lb: float | str | None
    ub: float | str | None
    value: float | str | None


@dataclass(frozen=True)
class ConstraintEvidence:
    name: str
    sense: str
    rhs: float | str | None
    slack: float | str | None
    violation: float | None


@dataclass(frozen=True)
class SolverEvidence:
    status: int | None
    status_name: str
    solution_count: int | None
    objective: float | str | None
    objective_sense: int | None
    model_constraint_violation: float | None
    max_constraint_violation: float
    sum_constraint_violation: float
    infeasible_constraints: int
    variables: tuple[VariableEvidence, ...] = field(default_factory=tuple)
    constraints: tuple[ConstraintEvidence, ...] = field(default_factory=tuple)
    iis: dict[str, list[str]] | dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_name": self.status_name,
            "sol_count": self.solution_count,
            "objective": self.objective,
            "objective_sense": self.objective_sense,
            "model_constr_vio": self.model_constraint_violation,
            "max_constr_vio": self.max_constraint_violation,
            "sum_constr_vio": self.sum_constraint_violation,
            "infeasible_constraints": self.infeasible_constraints,
            "variables": [vars(item) for item in self.variables],
            "constraints": [vars(item) for item in self.constraints],
            "iis": self.iis,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "status_name": self.status_name,
            "objective": self.objective,
            "objective_sense": self.objective_sense,
            "sol_count": self.solution_count,
            "max_constr_vio": self.max_constraint_violation,
            "sum_constr_vio": self.sum_constraint_violation,
            "infeasible_constraints": self.infeasible_constraints,
            "iis": self.iis,
        }


def collect_gurobi_evidence(
    model: Any,
    max_variables: int = 500,
    max_constraints: int = 1000,
) -> SolverEvidence:
    """Extract status, solution, slack, violation, and IIS from a Gurobi model."""
    import gurobipy as gp

    status_names = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        gp.GRB.NODE_LIMIT: "NODE_LIMIT",
        gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
    }
    status = int(model.Status)
    objective = safe_float(model.ObjVal) if model.SolCount > 0 else None

    model_vio = None
    try:
        model_vio = safe_float(model.getAttr(gp.GRB.Attr.ConstrVio))
    except Exception:
        pass

    variables: list[VariableEvidence] = []
    for var in model.getVars():
        if len(variables) >= max_variables:
            break
        try:
            value = var.X
        except Exception:
            value = None
        variables.append(
            VariableEvidence(
                name=var.VarName,
                vtype=str(var.VType),
                lb=safe_float(var.LB),
                ub=safe_float(var.UB),
                value=safe_float(value),
            )
        )

    constraints_list = model.getConstrs()
    slack_values: list[Any] = []
    try:
        slack_values = model.getAttr(gp.GRB.Attr.Slack, constraints_list)
    except Exception:
        pass

    constraints: list[ConstraintEvidence] = []
    violations: list[float] = []
    for index, constr in enumerate(constraints_list):
        if len(constraints) >= max_constraints:
            break
        rhs = safe_float(constr.RHS)
        slack = (
            safe_float(slack_values[index])
            if index < len(slack_values)
            else None
        )
        lhs = None
        try:
            lhs = safe_float(model.getRow(constr).getValue())
        except Exception:
            pass

        violation = None
        if lhs is not None and rhs is not None:
            if constr.Sense == "<":
                violation = max(0.0, float(lhs) - float(rhs))
            elif constr.Sense == ">":
                violation = max(0.0, float(rhs) - float(lhs))
            elif constr.Sense == "=":
                violation = abs(float(lhs) - float(rhs))
        if violation is not None and violation > 0:
            violations.append(violation)
        constraints.append(
            ConstraintEvidence(
                name=constr.ConstrName,
                sense=str(constr.Sense),
                rhs=rhs,
                slack=slack,
                violation=violation,
            )
        )

    iis = None
    if status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
        try:
            model.computeIIS()
            iis = {
                "constraints": [
                    c.ConstrName for c in model.getConstrs()
                    if getattr(c, "IISConstr", 0) > 0
                ][:200],
                "variables": [
                    v.VarName for v in model.getVars()
                    if getattr(v, "IISLB", 0) > 0
                    or getattr(v, "IISUB", 0) > 0
                ][:200],
            }
        except Exception:
            iis = {"error": "computeIIS failed"}

    return SolverEvidence(
        status=status,
        status_name=status_names.get(status, f"STATUS_{status}"),
        solution_count=int(model.SolCount),
        objective=objective,
        objective_sense=getattr(model, "ModelSense", None),
        model_constraint_violation=model_vio,
        max_constraint_violation=max(violations, default=0.0),
        sum_constraint_violation=sum(violations),
        infeasible_constraints=len(violations),
        variables=tuple(variables),
        constraints=tuple(constraints),
        iis=iis,
    )

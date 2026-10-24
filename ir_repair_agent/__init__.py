"""Verified ModelIR repair framework.

The package is intentionally independent from the legacy evaluation scripts.
Existing code can adopt it through the compiler, validator, and patch-generator
callbacks defined in ``repair.py``.
"""

from .budget import BudgetExceeded, BudgetLedger, BudgetLimits, BudgetSnapshot
from .contracts import (
    CheckEvidence,
    ConstraintSpec,
    EditAudit,
    FailureDiagnosis,
    IRPatch,
    LinearExpression,
    LinearTerm,
    ModelIR,
    ObjectiveSpec,
    ParameterSpec,
    PatchOperation,
    QuadraticTerm,
    SetSpec,
    ValidationResult,
    VariableSpec,
    assess_edit,
    diagnosis_from_validation,
)
from .experiment import ExperimentConfig, ExperimentRecord, compare_runs
from .execution_evidence import (
    ConstraintEvidence,
    SolverEvidence,
    VariableEvidence,
    collect_gurobi_evidence,
    safe_float,
)
from .memory import RepairCase, SqliteRepairMemory
from .memory_features import (
    constraint_skeleton,
    memory_error_pattern,
    structural_skeletons,
    structural_tokens,
)
from .repair import RepairEngine, RepairOutcome, RepairPolicy, RepairRequest

__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "BudgetLimits",
    "BudgetSnapshot",
    "CheckEvidence",
    "ConstraintSpec",
    "ConstraintEvidence",
    "EditAudit",
    "ExperimentConfig",
    "ExperimentRecord",
    "FailureDiagnosis",
    "IRPatch",
    "LinearExpression",
    "LinearTerm",
    "ModelIR",
    "ObjectiveSpec",
    "ParameterSpec",
    "PatchOperation",
    "QuadraticTerm",
    "RepairCase",
    "RepairEngine",
    "RepairOutcome",
    "RepairPolicy",
    "RepairRequest",
    "SetSpec",
    "SolverEvidence",
    "SqliteRepairMemory",
    "ValidationResult",
    "VariableSpec",
    "VariableEvidence",
    "assess_edit",
    "compare_runs",
    "collect_gurobi_evidence",
    "constraint_skeleton",
    "diagnosis_from_validation",
    "memory_error_pattern",
    "safe_float",
    "structural_skeletons",
    "structural_tokens",
]


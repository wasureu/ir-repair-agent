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
from .memory_inmemory import RepairMemory, normalize_class
from .memory_admission import (
    AdmissionDecision,
    TRUSTED_PROGRAM_SOURCES,
    assess_memory_admission,
)
from .repair import RepairEngine, RepairOutcome, RepairPolicy, RepairRequest
from .repair_gate import (
    GateDecision,
    assess_candidate,
    objective_improved,
    objective_nochange,
    select_final_fallback,
)
from .repair_patcher import (
    RepairPromptContext,
    build_repair_prompt,
    repair_code,
)
from .repair_pipeline import (
    RepairPipeline,
    RepairPipelineDependencies,
    RepairPipelineRequest,
)

__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "BudgetLimits",
    "BudgetSnapshot",
    "AdmissionDecision",
    "CheckEvidence",
    "ConstraintSpec",
    "ConstraintEvidence",
    "EditAudit",
    "ExperimentConfig",
    "ExperimentRecord",
    "FailureDiagnosis",
    "GateDecision",
    "IRPatch",
    "LinearExpression",
    "LinearTerm",
    "ModelIR",
    "ObjectiveSpec",
    "ParameterSpec",
    "PatchOperation",
    "QuadraticTerm",
    "RepairCase",
    "RepairMemory",
    "RepairEngine",
    "RepairOutcome",
    "RepairPolicy",
    "RepairRequest",
    "RepairPromptContext",
    "RepairPipeline",
    "RepairPipelineDependencies",
    "RepairPipelineRequest",
    "SetSpec",
    "SolverEvidence",
    "SqliteRepairMemory",
    "TRUSTED_PROGRAM_SOURCES",
    "ValidationResult",
    "VariableSpec",
    "VariableEvidence",
    "assess_edit",
    "assess_candidate",
    "assess_memory_admission",
    "build_repair_prompt",
    "compare_runs",
    "collect_gurobi_evidence",
    "constraint_skeleton",
    "diagnosis_from_validation",
    "memory_error_pattern",
    "normalize_class",
    "objective_improved",
    "objective_nochange",
    "safe_float",
    "select_final_fallback",
    "structural_skeletons",
    "structural_tokens",
    "repair_code",
]


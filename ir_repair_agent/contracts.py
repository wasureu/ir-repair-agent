from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

Relation = Literal["le", "ge", "eq"]
VariableType = Literal["continuous", "integer", "binary"]
ObjectiveSense = Literal["min", "max"]


def _number(value: int | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite number is not allowed: {value!r}")
    return 0.0 if result == 0 else result


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SetSpec:
    id: str
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("set id must not be empty")

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "members": list(self.members)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SetSpec":
        return cls(id=str(payload["id"]), members=tuple(map(str, payload.get("members", []))))


@dataclass(frozen=True)
class ParameterSpec:
    id: str
    value: Any
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("parameter id must not be empty")
        _json(self.value)

    def as_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        payload = {"id": self.id, "value": deepcopy(self.value)}
        if include_source:
            payload["source_refs"] = list(self.source_refs)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParameterSpec":
        return cls(
            id=str(payload["id"]),
            value=deepcopy(payload.get("value")),
            source_refs=tuple(map(str, payload.get("source_refs", []))),
        )


@dataclass(frozen=True)
class VariableSpec:
    id: str
    vtype: VariableType = "continuous"
    lb: float | None = 0.0
    ub: float | None = None
    indices: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("variable id must not be empty")
        if self.vtype not in {"continuous", "integer", "binary"}:
            raise ValueError(f"unsupported variable type: {self.vtype!r}")
        if self.lb is not None and self.ub is not None and float(self.lb) > float(self.ub):
            raise ValueError(f"variable {self.id!r} has lb > ub")

    def as_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "vtype": self.vtype,
            "lb": self.lb,
            "ub": self.ub,
            "indices": list(self.indices),
        }
        if include_source:
            payload["source_refs"] = list(self.source_refs)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VariableSpec":
        return cls(
            id=str(payload["id"]),
            vtype=str(payload.get("vtype", "continuous")),  # type: ignore[arg-type]
            lb=payload.get("lb"),
            ub=payload.get("ub"),
            indices=tuple(map(str, payload.get("indices", []))),
            source_refs=tuple(map(str, payload.get("source_refs", []))),
        )


@dataclass(frozen=True)
class LinearTerm:
    variable: str
    coefficient: float

    def __post_init__(self) -> None:
        if not self.variable.strip():
            raise ValueError("linear term variable must not be empty")
        object.__setattr__(self, "coefficient", _number(self.coefficient))

    def as_dict(self) -> dict[str, Any]:
        return {"variable": self.variable, "coefficient": self.coefficient}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearTerm":
        return cls(
            variable=str(payload["variable"]),
            coefficient=float(payload["coefficient"]),
        )


@dataclass(frozen=True)
class LinearExpression:
    terms: tuple[LinearTerm, ...] = ()
    constant: float = 0.0

    def __post_init__(self) -> None:
        combined: dict[str, float] = {}
        for term in self.terms:
            combined[term.variable] = combined.get(term.variable, 0.0) + term.coefficient
        normalized = tuple(
            LinearTerm(variable, value)
            for variable, value in sorted(combined.items())
            if value != 0
        )
        object.__setattr__(self, "terms", normalized)
        object.__setattr__(self, "constant", _number(self.constant))

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(term.variable for term in self.terms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "terms": [term.as_dict() for term in self.terms],
            "constant": self.constant,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearExpression":
        return cls(
            terms=tuple(LinearTerm.from_dict(item) for item in payload.get("terms", [])),
            constant=float(payload.get("constant", 0.0)),
        )


@dataclass(frozen=True)
class QuadraticTerm:
    variable_a: str
    variable_b: str
    coefficient: float

    def __post_init__(self) -> None:
        if not self.variable_a.strip() or not self.variable_b.strip():
            raise ValueError("quadratic term variables must not be empty")
        first, second = sorted((self.variable_a, self.variable_b))
        object.__setattr__(self, "variable_a", first)
        object.__setattr__(self, "variable_b", second)
        object.__setattr__(self, "coefficient", _number(self.coefficient))

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable_a": self.variable_a,
            "variable_b": self.variable_b,
            "coefficient": self.coefficient,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuadraticTerm":
        return cls(
            variable_a=str(payload["variable_a"]),
            variable_b=str(payload["variable_b"]),
            coefficient=float(payload["coefficient"]),
        )


@dataclass(frozen=True)
class ObjectiveSpec:
    sense: ObjectiveSense
    expression: LinearExpression
    source_refs: tuple[str, ...] = ()
    quadratic_terms: tuple[QuadraticTerm, ...] = ()

    def __post_init__(self) -> None:
        if self.sense not in {"min", "max"}:
            raise ValueError(f"unsupported objective sense: {self.sense!r}")
        combined: dict[tuple[str, str], float] = {}
        for term in self.quadratic_terms:
            key = (term.variable_a, term.variable_b)
            combined[key] = combined.get(key, 0.0) + term.coefficient
        object.__setattr__(
            self,
            "quadratic_terms",
            tuple(
                QuadraticTerm(first, second, coefficient)
                for (first, second), coefficient in sorted(combined.items())
                if coefficient != 0
            ),
        )

    def as_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sense": self.sense,
            "expression": self.expression.as_dict(),
            "quadratic_terms": [term.as_dict() for term in self.quadratic_terms],
        }
        if include_source:
            payload["source_refs"] = list(self.source_refs)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectiveSpec":
        return cls(
            sense=str(payload["sense"]),  # type: ignore[arg-type]
            expression=LinearExpression.from_dict(payload["expression"]),
            source_refs=tuple(map(str, payload.get("source_refs", []))),
            quadratic_terms=tuple(
                QuadraticTerm.from_dict(item)
                for item in payload.get("quadratic_terms", [])
            ),
        )


@dataclass(frozen=True)
class ConstraintSpec:
    id: str
    lhs: LinearExpression
    relation: Relation
    rhs: float
    family: str = "general"
    scope: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("constraint id must not be empty")
        if self.relation not in {"le", "ge", "eq"}:
            raise ValueError(f"unsupported relation: {self.relation!r}")
        object.__setattr__(self, "rhs", _number(self.rhs))

    @property
    def feature_tokens(self) -> tuple[str, ...]:
        variable_families = {
            variable.split("[", 1)[0].split("_", 1)[0]
            for variable in self.lhs.variables
        }
        return tuple(
            sorted(
                {
                    f"family:{self.family}",
                    f"relation:{self.relation}",
                    *(f"var_family:{item}" for item in variable_families if item),
                }
            )
        )

    def as_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "lhs": self.lhs.as_dict(),
            "relation": self.relation,
            "rhs": self.rhs,
            "family": self.family,
            "scope": list(self.scope),
        }
        if include_source:
            payload["source_refs"] = list(self.source_refs)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConstraintSpec":
        return cls(
            id=str(payload["id"]),
            lhs=LinearExpression.from_dict(payload["lhs"]),
            relation=str(payload["relation"]),  # type: ignore[arg-type]
            rhs=float(payload["rhs"]),
            family=str(payload.get("family", "general")),
            scope=tuple(map(str, payload.get("scope", []))),
            source_refs=tuple(map(str, payload.get("source_refs", []))),
        )


@dataclass(frozen=True)
class ModelIR:
    problem_id: str
    problem_class: str
    variables: tuple[VariableSpec, ...]
    objective: ObjectiveSpec
    constraints: tuple[ConstraintSpec, ...]
    sets: tuple[SetSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variables",
            tuple(sorted(self.variables, key=lambda item: item.id)),
        )
        object.__setattr__(
            self,
            "constraints",
            tuple(sorted(self.constraints, key=lambda item: item.id)),
        )
        object.__setattr__(self, "sets", tuple(sorted(self.sets, key=lambda item: item.id)))
        object.__setattr__(
            self,
            "parameters",
            tuple(sorted(self.parameters, key=lambda item: item.id)),
        )

    def validate(self) -> None:
        variable_ids = [item.id for item in self.variables]
        constraint_ids = [item.id for item in self.constraints]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("duplicate variable ids")
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("duplicate constraint ids")
        known_variables = set(variable_ids)
        referenced = set(self.objective.expression.variables)
        for term in self.objective.quadratic_terms:
            referenced.add(term.variable_a)
            referenced.add(term.variable_b)
        for constraint in self.constraints:
            referenced.update(constraint.lhs.variables)
        unknown = sorted(referenced - known_variables)
        if unknown:
            raise ValueError(f"model references unknown variables: {unknown}")

    def constraint(self, constraint_id: str) -> ConstraintSpec:
        for constraint in self.constraints:
            if constraint.id == constraint_id:
                return constraint
        raise KeyError(constraint_id)

    def variable(self, variable_id: str) -> VariableSpec:
        for variable in self.variables:
            if variable.id == variable_id:
                return variable
        raise KeyError(variable_id)

    def semantic_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "problem_class": self.problem_class,
            "sets": [item.as_dict() for item in self.sets],
            "parameters": [item.as_dict(include_source=False) for item in self.parameters],
            "variables": [
                item.as_dict(include_source=False) for item in self.variables
            ],
            "objective": self.objective.as_dict(include_source=False),
            "constraints": [
                item.as_dict(include_source=False) for item in self.constraints
            ],
        }

    def fingerprint(self) -> str:
        return _hash(self.semantic_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "problem_id": self.problem_id,
            **self.semantic_payload(),
            "metadata": deepcopy(self.metadata),
            "variables": [
                item.as_dict(include_source=True) for item in self.variables
            ],
            "parameters": [
                item.as_dict(include_source=True) for item in self.parameters
            ],
            "objective": self.objective.as_dict(include_source=True),
            "constraints": [
                item.as_dict(include_source=True) for item in self.constraints
            ],
        }
        return payload

    def canonical_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelIR":
        return cls(
            problem_id=str(payload["problem_id"]),
            problem_class=str(payload["problem_class"]),
            variables=tuple(
                VariableSpec.from_dict(item) for item in payload.get("variables", [])
            ),
            objective=ObjectiveSpec.from_dict(payload["objective"]),
            constraints=tuple(
                ConstraintSpec.from_dict(item) for item in payload.get("constraints", [])
            ),
            sets=tuple(SetSpec.from_dict(item) for item in payload.get("sets", [])),
            parameters=tuple(
                ParameterSpec.from_dict(item) for item in payload.get("parameters", [])
            ),
            metadata=deepcopy(payload.get("metadata", {})),
        )

    def changed_elements(self, other: "ModelIR") -> tuple[str, ...]:
        left = self.semantic_payload()
        right = other.semantic_payload()
        changed: list[str] = []
        for section, prefix in (
            ("sets", "set"),
            ("parameters", "parameter"),
            ("variables", "variable"),
        ):
            left_items = {item["id"]: item for item in left[section]}
            right_items = {item["id"]: item for item in right[section]}
            for item_id in sorted(set(left_items) | set(right_items)):
                if left_items.get(item_id) != right_items.get(item_id):
                    changed.append(f"{prefix}:{item_id}")
        if left["objective"] != right["objective"]:
            changed.append("objective")
        left_constraints = {item["id"]: item for item in left["constraints"]}
        right_constraints = {item["id"]: item for item in right["constraints"]}
        for item_id in sorted(set(left_constraints) | set(right_constraints)):
            if left_constraints.get(item_id) != right_constraints.get(item_id):
                changed.append(f"constraint:{item_id}")
        return tuple(changed)


@dataclass(frozen=True)
class PatchOperation:
    op: str
    target: str
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "target": self.target,
            "payload": deepcopy(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatchOperation":
        return cls(
            op=str(payload["op"]),
            target=str(payload.get("target", "")),
            payload=deepcopy(payload.get("payload", {})),
        )


@dataclass(frozen=True)
class IRPatch:
    operations: tuple[PatchOperation, ...]
    description: str = ""
    generator: str = "llm"
    confidence: float = 0.0
    memory_case_id: str = ""

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("a patch must contain at least one operation")

    def as_dict(self) -> dict[str, Any]:
        return {
            "operations": [operation.as_dict() for operation in self.operations],
            "description": self.description,
            "generator": self.generator,
            "confidence": float(self.confidence),
            "memory_case_id": self.memory_case_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IRPatch":
        return cls(
            operations=tuple(
                PatchOperation.from_dict(item) for item in payload["operations"]
            ),
            description=str(payload.get("description", "")),
            generator=str(payload.get("generator", "llm")),
            confidence=float(payload.get("confidence", 0.0)),
            memory_case_id=str(payload.get("memory_case_id", "")),
        )

    def bind_affected(self, affected_elements: tuple[str, ...]) -> "IRPatch":
        def bind(value: str) -> str:
            result = value
            for index, element_id in enumerate(affected_elements):
                result = result.replace(f"{{{{affected:{index}}}}}", element_id)
            return result

        return IRPatch(
            operations=tuple(
                PatchOperation(
                    op=operation.op,
                    target=bind(operation.target),
                    payload=deepcopy(operation.payload),
                )
                for operation in self.operations
            ),
            description=self.description,
            generator=self.generator,
            confidence=self.confidence,
            memory_case_id=self.memory_case_id,
        )

    def has_unbound_targets(self) -> bool:
        return any("{{affected:" in operation.target for operation in self.operations)

    def apply(self, model_ir: ModelIR) -> ModelIR:
        if self.has_unbound_targets():
            raise ValueError("patch still contains unbound affected-element placeholders")
        state = deepcopy(model_ir.to_dict())
        constraint_by_id = {item["id"]: item for item in state["constraints"]}
        variable_by_id = {item["id"]: item for item in state["variables"]}

        def element_id(target: str, kind: str) -> str:
            prefix = f"{kind}:"
            return target[len(prefix):] if target.startswith(prefix) else target

        for operation in self.operations:
            op = operation.op
            target = operation.target
            payload = operation.payload
            if op == "replace_relation":
                target_id = element_id(target, "constraint")
                if target_id not in constraint_by_id:
                    raise KeyError(f"constraint not found: {target_id}")
                constraint_by_id[target_id]["relation"] = str(payload["relation"])
            elif op == "replace_rhs":
                target_id = element_id(target, "constraint")
                if target_id not in constraint_by_id:
                    raise KeyError(f"constraint not found: {target_id}")
                constraint_by_id[target_id]["rhs"] = _number(payload["rhs"])
            elif op == "replace_constraint":
                replacement = ConstraintSpec.from_dict(payload["constraint"])
                target_id = element_id(target, "constraint")
                if target_id not in constraint_by_id:
                    raise KeyError(f"constraint not found: {target_id}")
                if replacement.id != target_id:
                    raise ValueError("replace_constraint cannot change the target id")
                constraint_by_id[target_id] = replacement.as_dict(include_source=True)
            elif op == "add_constraint":
                addition = ConstraintSpec.from_dict(payload["constraint"])
                if addition.id in constraint_by_id:
                    raise ValueError(f"constraint already exists: {addition.id}")
                constraint_by_id[addition.id] = addition.as_dict(include_source=True)
            elif op == "delete_constraint":
                target_id = element_id(target, "constraint")
                if target_id not in constraint_by_id:
                    raise KeyError(f"constraint not found: {target_id}")
                del constraint_by_id[target_id]
            elif op == "replace_objective":
                state["objective"] = ObjectiveSpec.from_dict(
                    payload["objective"]
                ).as_dict(include_source=True)
            elif op == "replace_objective_sense":
                state["objective"]["sense"] = str(payload["sense"])
            elif op == "replace_variable_domain":
                target_id = element_id(target, "variable")
                if target_id not in variable_by_id:
                    raise KeyError(f"variable not found: {target_id}")
                current = variable_by_id[target_id]
                for key in ("vtype", "lb", "ub"):
                    if key in payload:
                        current[key] = payload[key]
            else:
                raise ValueError(f"unsupported patch operation: {op}")

        state["constraints"] = sorted(
            constraint_by_id.values(),
            key=lambda item: item["id"],
        )
        state["variables"] = sorted(
            variable_by_id.values(),
            key=lambda item: item["id"],
        )
        result = ModelIR.from_dict(state)
        result.validate()
        return result


@dataclass(frozen=True)
class FailureDiagnosis:
    failure_code: str
    affected_elements: tuple[str, ...]
    features: tuple[str, ...] = ()
    summary: str = ""
    provenance: str = "validator"
    confidence: float = 1.0
    allow_new_elements: bool = False

    def __post_init__(self) -> None:
        if not self.failure_code.strip():
            raise ValueError("diagnosis failure_code must not be empty")

    def signature(self) -> str:
        payload = {
            "failure_code": self.failure_code,
            "features": sorted(set(self.features)),
            "affected_types": sorted(
                element.split(":", 1)[0]
                for element in self.affected_elements
                if ":" in element
            ),
        }
        return _hash(payload)


@dataclass(frozen=True)
class CheckEvidence:
    name: str
    status: Literal["PASS", "FAIL", "SKIP"]
    source: str = "program"
    hard: bool = True
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("check name must not be empty")
        if self.status not in {"PASS", "FAIL", "SKIP"}:
            raise ValueError(f"unsupported check status: {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "source": self.source,
            "hard": self.hard,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ValidationResult:
    status: Literal["PASS", "FAIL", "REVIEW"]
    checks: tuple[CheckEvidence, ...] = ()
    failure_code: str = ""
    affected_elements: tuple[str, ...] = ()
    solver_status: str = ""
    objective: float | None = None
    max_constraint_violation: float | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL", "REVIEW"}:
            raise ValueError(f"unsupported validation status: {self.status!r}")

    def hard_pass(
        self,
        required_checks: tuple[str, ...],
        *,
        require_program_evidence: bool = True,
    ) -> bool:
        if self.status != "PASS":
            return False
        if any(check.hard and check.status == "FAIL" for check in self.checks):
            return False
        by_name = {check.name: check for check in self.checks}
        trusted_sources = {"program", "solver", "metamorphic"}
        for name in required_checks:
            check = by_name.get(name)
            if check is None or check.status != "PASS":
                return False
            if require_program_evidence and check.source not in trusted_sources:
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": [check.as_dict() for check in self.checks],
            "failure_code": self.failure_code,
            "affected_elements": list(self.affected_elements),
            "solver_status": self.solver_status,
            "objective": self.objective,
            "max_constraint_violation": self.max_constraint_violation,
            "artifacts": deepcopy(self.artifacts),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ValidationResult":
        return cls(
            status=str(payload["status"]),  # type: ignore[arg-type]
            checks=tuple(
                CheckEvidence(
                    name=str(item["name"]),
                    status=str(item["status"]),  # type: ignore[arg-type]
                    source=str(item.get("source", "program")),
                    hard=bool(item.get("hard", True)),
                    detail=str(item.get("detail", "")),
                )
                for item in payload.get("checks", [])
            ),
            failure_code=str(payload.get("failure_code", "")),
            affected_elements=tuple(map(str, payload.get("affected_elements", []))),
            solver_status=str(payload.get("solver_status", "")),
            objective=payload.get("objective"),
            max_constraint_violation=payload.get("max_constraint_violation"),
            artifacts=deepcopy(payload.get("artifacts", {})),
        )


def diagnosis_from_validation(result: ValidationResult) -> FailureDiagnosis:
    failure_code = result.failure_code or (
        "VALIDATION_FAILED" if result.status != "PASS" else "NO_FAILURE"
    )
    features = tuple(
        f"{check.name}:{check.status}"
        for check in result.checks
        if check.status != "PASS"
    )
    return FailureDiagnosis(
        failure_code=failure_code,
        affected_elements=result.affected_elements,
        features=features,
        summary="; ".join(check.detail for check in result.checks if check.detail),
        provenance="validator",
        confidence=0.8,
    )


@dataclass(frozen=True)
class EditAudit:
    before_fingerprint: str
    after_fingerprint: str
    changed_elements: tuple[str, ...]
    allowed_elements: tuple[str, ...]
    extra_changes: tuple[str, ...]
    missed_changes: tuple[str, ...]
    regressions: tuple[str, ...]

    @property
    def wrong_edit_rate(self) -> float:
        if not self.changed_elements:
            return 0.0
        return len(self.extra_changes) / len(self.changed_elements)

    @property
    def has_scope_violation(self) -> bool:
        return bool(self.extra_changes)

    @property
    def has_regression(self) -> bool:
        return bool(self.regressions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "changed_elements": list(self.changed_elements),
            "allowed_elements": list(self.allowed_elements),
            "extra_changes": list(self.extra_changes),
            "missed_changes": list(self.missed_changes),
            "regressions": list(self.regressions),
            "wrong_edit_rate": self.wrong_edit_rate,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditAudit":
        return cls(
            before_fingerprint=str(payload["before_fingerprint"]),
            after_fingerprint=str(payload["after_fingerprint"]),
            changed_elements=tuple(map(str, payload.get("changed_elements", []))),
            allowed_elements=tuple(map(str, payload.get("allowed_elements", []))),
            extra_changes=tuple(map(str, payload.get("extra_changes", []))),
            missed_changes=tuple(map(str, payload.get("missed_changes", []))),
            regressions=tuple(map(str, payload.get("regressions", []))),
        )


def assess_edit(
    before: ModelIR,
    after: ModelIR,
    *,
    allowed_elements: tuple[str, ...],
    expected_elements: tuple[str, ...] | None = None,
    regressions: tuple[str, ...] = (),
) -> EditAudit:
    changed = before.changed_elements(after)
    target = expected_elements if expected_elements is not None else allowed_elements
    allowed = set(target)
    extra = tuple(item for item in changed if item not in allowed)
    missed = tuple(item for item in target if item not in changed)
    return EditAudit(
        before_fingerprint=before.fingerprint(),
        after_fingerprint=after.fingerprint(),
        changed_elements=changed,
        allowed_elements=tuple(sorted(allowed)),
        extra_changes=extra,
        missed_changes=missed,
        regressions=tuple(sorted(set(regressions))),
    )

# Repository Migration Status

## Phase 1: repository and package shell

Completed:

- `pyproject.toml`, README, Git ignore rules, Makefile, pre-commit config;
- installable flat package `ir_repair_agent/`;
- canonical `ExecutionEvidence` module;
- legacy evaluator subprocess adapter;
- unified CLI entry point;
- example TOML experiment configurations;
- unittest coverage for execution evidence and existing IR contracts.

Compatibility rule:

- `code/` remains the operational legacy implementation.
- New code should import from `ir_repair_agent/`.
- The CLI launches the legacy evaluator as a subprocess so environment flags
  such as `ORACLE_GATE` and `GATE_LOG` are fixed before module import.

## Phase 2: split the legacy evaluator

Completed modules:

```text
ir_repair_agent/llm/parsing.py
ir_repair_agent/data/dataset.py
ir_repair_agent/data/frozen_codes.py
ir_repair_agent/evaluation/scoring.py
ir_repair_agent/evaluation/metrics.py
```

The legacy evaluator now delegates code parsing, numeric-answer parsing,
problem/frozen-code loading, final scoring tolerance, token subtraction, and
token aggregation to these modules.

Still planned:

```text
llm/client.py
evaluation/runner.py
```

Remaining golden-output tests must cover:

- frozen initial-code loading;
- no-oracle behavior;
- token and elapsed-time accounting;
- incremental JSONL writing.

## Phase 3: split the auditor and repair pipeline

Completed modules:

```text
ir_repair_agent/audit/direction.py
ir_repair_agent/audit/equivalence.py
ir_repair_agent/audit/self_diagnosis.py
ir_repair_agent/memory_features.py
ir_repair_agent/memory_inmemory.py
ir_repair_agent/memory_admission.py
ir_repair_agent/repair_gate.py
ir_repair_agent/repair_patcher.py
ir_repair_agent/verification.py
```

The legacy auditor currently delegates multi-source objective-direction
evidence, equivalence verification, strict equivalence-result validation, and
solver-regression checks to these package modules. The three-run consensus
self-diagnosis and its independent verification are also delegated.
Legacy repair-memory error compression and structural/skeleton features now
delegate to `memory_features.py`.
The legacy `RepairMemory` is a compatibility wrapper over the package
implementation, including bucketing, similarity, confidence filtering, A/B
tier ordering, B-tier limits, and case export/load.
SQLite repair-memory admission now delegates to an auditable policy object
that reports missing, failed, and untrusted verification checks.
The legacy healing pipeline now delegates candidate accept/reject/provisional
decisions and final safe-fallback selection to `repair_gate.py`.
Repair prompt construction and LLM patch extraction now live in
`repair_patcher.py`; the prompt includes compact-formulation and
goal-programming constraints.

Still planned:

```text
 audit/hybrid_auditor.py
repair/pipeline.py
```

The gate must remain independent of the standard answer. Oracle-visible logic
stays behind an explicit diagnostic switch.

## Phase 4: research-grade experiment runner

Target:

- development/test split;
- sequential memory construction;
- frozen SQLite memory;
- multiple seeds;
- complete run manifest;
- paired statistics and confidence intervals.

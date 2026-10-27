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
ir_repair_agent/llm/client.py
ir_repair_agent/execution/gurobi_runner.py
ir_repair_agent/evaluation/runner.py
```

The legacy evaluator now delegates code parsing, numeric-answer parsing,
problem/frozen-code loading, final scoring tolerance, token subtraction, and
token aggregation to these modules.
OpenAI-compatible requests, empty-response retries, reasoning fallback,
timeout configuration, and usage callbacks now delegate to `llm/client.py`.
Gurobi environment creation, execution, result caching, solver evidence, and
IIS collection now delegate to `execution/gurobi_runner.py`.
The evaluator's prompt-to-code path, frozen-code reuse, healing request
adapter, fallback execution, thread-local token accounting, incremental JSONL
writing, initial-code dumping, and metrics now delegate to
`evaluation/runner.py`.

Still planned:

```text
legacy evaluator method cleanup after WSL parity acceptance
```

The migration tests now cover:

- frozen initial-code loading;
- no-oracle behavior;
- token and elapsed-time accounting;
- incremental JSONL writing and metrics output;
- evaluator-to-runner dependency wiring.

## Phase 3: split the auditor and repair pipeline

Completed modules:

```text
ir_repair_agent/audit/direction.py
ir_repair_agent/audit/equivalence.py
ir_repair_agent/audit/hybrid_auditor.py
ir_repair_agent/audit/objective_witness.py
ir_repair_agent/audit/self_diagnosis.py
ir_repair_agent/memory_features.py
ir_repair_agent/memory_inmemory.py
ir_repair_agent/memory_admission.py
ir_repair_agent/repair_gate.py
ir_repair_agent/repair_patcher.py
ir_repair_agent/repair_pipeline.py
ir_repair_agent/verification.py
```

The legacy auditor currently delegates three-stage audit orchestration,
schema/coverage hard checks, multi-source objective-direction
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
It also performs a post-loop strict audit when the final accepted candidate
was generated in the last repair round and therefore had not yet been audited.
Repair prompt construction and LLM patch extraction now live in
`repair_patcher.py`; the prompt includes compact-formulation and
goal-programming constraints.
The full audit-repair-gate loop now runs in `RepairPipeline`; the legacy
`run_healing_pipeline` remains only as a compatibility adapter.
Equivalent unchanged code now also passes an objective-variable witness:
Python extracts symbols used by `setObjective`, while the semantic layer
supplies required/forbidden variable families for a hard consistency check.
The same witness is applied to unchanged code that receives a strict PASS.
Post-loop `PASS_EQUIVALENT` is not treated as terminal and cannot bypass the
safe fallback.
Automatic consensus self-diagnosis remains enabled by default as the fallback
hint path: it runs only when hints are enabled and neither a direct
`AUDIT_HINTS` entry nor a migrated hint exists. `SELF_DIAGNOSE=0` is reserved
for the explicit no-auto-diagnosis ablation.

Still planned:

```text
remove the temporary `_legacy_*` evaluator methods after end-to-end parity
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

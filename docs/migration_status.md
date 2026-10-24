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
ir_repair_agent/verification.py
```

The legacy auditor currently delegates multi-source objective-direction
evidence and solver-regression checks to these package modules.

Still planned:

```text
 audit/hybrid_auditor.py
 audit/equivalence.py
audit/self_diagnosis.py
repair/pipeline.py
repair/patcher.py
repair/gate.py
memory/features.py
memory/admission.py
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

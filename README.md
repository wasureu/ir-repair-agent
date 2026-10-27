# IR Repair Agent

Verification-gated self-repair and experience memory for LLM-generated
optimization models.

## Status

This repository contains two layers:

- `ir_repair_agent/`: structured ModelIR contracts, evidence, budget,
  SQLite memory, repair engine, evaluator runner, and the new package CLI.
- `code/`: the operational legacy evaluator and auditor used by the current
  experiments. It is now a compatibility layer around the package for
  evaluation, LLM, Gurobi execution, audit, and repair orchestration.

The normal repair path never exposes the reference answer. `ORACLE_GATE=1` is
reserved for diagnostic upper-bound experiments.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[runtime]"
```

## Run the existing evaluator through the new CLI

```bash
ir-repair-run run --config configs/experiments/industryor_pilot.toml
```

The CLI is a compatibility adapter around `code/evaluation_tir_gurobi_robust.py`.
It records the configuration before launching and can be used while the
legacy evaluator is split into smaller modules.

## Test

```bash
python -m pytest
```

## Main research invariants

- Reference answers are used only for final scoring unless `ORACLE_GATE=1`.
- A changed candidate cannot be accepted from `PASS_EQUIVALENT` alone.
- Objective direction requires multi-source evidence.
- Solver-level regressions can reject a candidate.
- An unverified final candidate may fall back to the last verified or
  executable baseline.
- Memory admission must ultimately be backed by program, solver, or
  metamorphic evidence.

Architecture notes are in `docs/`.

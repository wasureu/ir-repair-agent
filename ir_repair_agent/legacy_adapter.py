"""Subprocess adapter for the operational legacy evaluator."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LegacyRunConfig:
    input_path: str
    output_path: str
    initial_codes_file: str | None = None
    model: str = "deepseek-v4-flash"
    max_workers: int = 4
    seed: int | None = None
    no_healing: bool = False
    stdout_path: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LegacyRunConfig":
        run = data.get("run", data)
        env = {str(k): str(v) for k, v in data.get("env", {}).items()}
        return cls(
            input_path=str(run["input_path"]),
            output_path=str(run["output_path"]),
            initial_codes_file=(
                str(run["initial_codes_file"])
                if run.get("initial_codes_file")
                else None
            ),
            model=str(run.get("model", "deepseek-v4-flash")),
            max_workers=int(run.get("max_workers", 4)),
            seed=int(run["seed"]) if run.get("seed") is not None else None,
            no_healing=bool(run.get("no_healing", False)),
            stdout_path=(
                str(run["stdout_path"]) if run.get("stdout_path") else None
            ),
            env=env,
        )


def build_legacy_command(config: LegacyRunConfig) -> list[str]:
    script = REPO_ROOT / "code" / "evaluation_tir_gurobi_robust.py"
    command = [
        sys.executable,
        "-u",
        str(script),
        "--input_path", config.input_path,
        "--output_path", config.output_path,
        "--model", config.model,
        "--max_workers", str(config.max_workers),
    ]
    if config.initial_codes_file:
        command.extend(["--initial_codes_file", config.initial_codes_file])
    if config.seed is not None:
        command.extend(["--seed", str(config.seed)])
    if config.no_healing:
        command.append("--no-healing")
    return command


def run_legacy(config: LegacyRunConfig, dry_run: bool = False) -> int:
    command = build_legacy_command(config)
    env = os.environ.copy()
    env.update(config.env)
    if config.seed is not None:
        env["SEED"] = str(config.seed)

    print("Running:", " ".join(command))
    print("Environment overrides:", config.env)
    if dry_run:
        return 0

    stdout_path = Path(config.stdout_path) if config.stdout_path else None
    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    else:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
    return completed.returncode

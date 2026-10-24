"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from .legacy_adapter import LegacyRunConfig, run_legacy


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as handle:
            return tomllib.load(handle)
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _record_config(path: Path, data: dict[str, Any]) -> None:
    snapshot = path.parent / f"{path.stem}.resolved.json"
    snapshot.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ir-repair-run")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a configured experiment")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        data = _load_config(args.config)
        _record_config(args.config, data)
        config = LegacyRunConfig.from_mapping(data)
        return run_legacy(config, dry_run=args.dry_run)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

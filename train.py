#!/usr/bin/env python3
"""Launch a task's training module using its YAML parameters."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


def build_command(config, overrides):
    arguments = dict(config["arguments"])
    for override in overrides:
        key, value = override.split("=", 1)
        if key not in arguments:
            raise ValueError(f"Unknown configuration parameter: {key}")
        arguments[key] = yaml.safe_load(value)
    command = [sys.executable, "-m", config["module"]]
    for key, value in arguments.items():
        if value is None:
            command.append(f"--{key}")
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, list):
            value = ",".join(map(str, value))
        command.append(f"--{key}={value}")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="PARAMETER=VALUE", help="Override a YAML parameter; repeat as needed.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without training.")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    elif extra:
        parser.error("Place native trainer flags after --")
    root = Path(__file__).resolve().parent
    config = yaml.safe_load(args.config.read_text())
    if config["backend"] not in ("custom", "velocity"):
        parser.error("backend must be custom or velocity")
    command = build_command(config, args.overrides) + extra
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / config["backend"])
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return subprocess.run(command, cwd=root, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())

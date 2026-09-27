#!/usr/bin/env python
"""Eval-accelerated wrapper for Quad2D stabilize-avoid training.

For this cheap 2D environment, per-step JAX env stepping is slower than the
plain Python/NumPy collector because it forces host/device synchronization.
Keep online collection on the Python backend and use the scan/JIT path for
evaluation and postprocess rollouts, where it actually pays off.
"""

from __future__ import annotations

import sys

from absl import app

from ssm.tasks.quad2d_stabilize import train as base


def _warn_forced_flag(name: str, value: str) -> None:
    prefix = f"--{name}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix) and arg.split("=", 1)[1].lower() != value:
            print(f"WARNING: train_quad2d_stab_p123.py forces {name}={value}; ignoring CLI override.")
            break


def main(argv):
    _warn_forced_flag("env_backend", "python")
    _warn_forced_flag("eval_backend", "p123")
    base.FLAGS["env_backend"].value = "python"
    base.FLAGS["eval_backend"].value = "p123"
    if not base.FLAGS.run_name:
        base.FLAGS["run_name"].value = base.auto_run_name()
    print("P123 wrapper active: using Python env backend and eval backend=scan_single_episode")
    return base.main(argv)


if __name__ == "__main__":
    app.run(main)

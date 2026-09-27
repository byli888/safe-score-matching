#!/usr/bin/env python3
"""Evaluate one checkpoint with an explicit task protocol."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Evaluation protocol YAML, not a training configuration.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='One locally trained or trusted checkpoint file.')
    parser.add_argument('--out', type=Path, required=True, help='Output JSON file.')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without evaluation.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    protocol = args.config.expanduser().resolve()
    config = yaml.safe_load(protocol.read_text())
    if config['backend'] not in ('custom', 'velocity'):
        parser.error('backend must be custom or velocity')
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error('--checkpoint must be a single existing file')
    command = [sys.executable, '-m', config['module'], '--protocol', str(protocol),
               '--checkpoint', str(checkpoint), '--out', str(args.out.expanduser().resolve())]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / config['backend'])
    env['JAX_PLATFORMS'] = 'cpu'
    return subprocess.run(command, cwd=root, env=env).returncode


if __name__ == '__main__':
    sys.exit(main())

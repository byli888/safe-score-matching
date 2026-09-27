#!/usr/bin/env python3
"""Evaluate one trusted local F16 Stage-A or Stage-B checkpoint.

The named protocol fixes the population, gate and stabilization definitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.protocol = a.protocol.resolve(strict=True)
    a.checkpoint = a.checkpoint.resolve(strict=True)
    a.out = a.out.resolve()
    if a.out.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation: {a.out}')
    import yaml
    specification = yaml.safe_load(a.protocol.read_text())
    if (specification['backend'], specification['module'], specification['task']) != ('custom', 'evaluate_f16', 'f16'):
        raise ValueError('Expected an F16 custom evaluation protocol')
    protocol = specification['protocol']
    grid = protocol['grid']
    for key in ('episodes', 'horizon', 'last_n', 'gate_chunk_size'):
        if isinstance(protocol[key], bool) or not isinstance(protocol[key], int) or protocol[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if protocol['last_n'] > protocol['horizon']:
        raise ValueError('last_n must not exceed horizon')
    for key in ('n_theta', 'n_height'):
        if isinstance(grid[key], bool) or not isinstance(grid[key], int) or grid[key] < 2:
            raise ValueError(f'grid.{key} must be an integer >= 2')
    for lo, hi in (('theta_min', 'theta_max'), ('height_min', 'height_max')):
        if not all(math.isfinite(float(grid[k])) for k in (lo, hi)) or grid[lo] >= grid[hi]:
            raise ValueError(f'Invalid grid bounds: {lo}/{hi}')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import jax
    import jax.numpy as jnp
    import numpy as np
    import ssm
    import jaxrl5
    from ssm.tasks.f16.evaluation import load_checkpoint, diagnostic_grid, score_grid, checkpoint_rollout
    source_root = Path(__file__).resolve().parent
    for package in (ssm, jaxrl5):
        if not Path(package.__file__).resolve().is_relative_to(source_root):
            raise RuntimeError(f'{package.__name__} loaded outside the release source')
    strict = bool(protocol.get("strict_invalid_dynamics", True))
    safety_geometry = protocol.get("safety_geometry", "checkpoint_effective")
    if safety_geometry not in ("checkpoint_effective", "region_bounds"):
        raise ValueError("safety_geometry must be checkpoint_effective or region_bounds")
    use_training_bounds = safety_geometry == "checkpoint_effective"
    agent, config, policy_name, env = load_checkpoint(str(a.checkpoint), int(protocol["policy_seed"]),
        strict_invalid_dynamics=strict, apply_training_safety_overrides=use_training_bounds)
    if agent.stage_mode != protocol.get("stage_mode", "stage_a"):
        raise ValueError("Checkpoint stage does not match protocol.stage_mode")
    states = diagnostic_grid(env, grid)
    physical_h = np.asarray([env._task_safety_margin(x) for x in states], dtype=np.float64)
    scores = score_grid(agent, env, states, int(protocol['gate_seed']), protocol['gate_chunk_size'])
    accepted = np.flatnonzero(scores < 0.0)
    sample_size = min(protocol['episodes'], len(accepted))
    offsets = np.random.default_rng(protocol['init_seed']).choice(len(accepted), size=sample_size, replace=False)
    indices = accepted[offsets]
    print(f'F16 {agent.stage_mode} grid: {len(accepted)}/{len(states)} predicted safe; {sample_size} rollouts', flush=True)
    rollout = checkpoint_rollout(env)
    base_key = jax.random.PRNGKey(protocol['policy_seed'])
    episodes = []
    for i, grid_index in enumerate(indices):
        key = jax.random.fold_in(base_key, i)
        outcome = rollout(agent.replace(rng=key), jnp.asarray(states[grid_index], jnp.float32),
                          horizon=protocol['horizon'], last_n=protocol['last_n'])
        outcome = jax.device_get({k: v for k, v in outcome.items() if k != 'agent'})
        record = {k: np.asarray(v).item() for k, v in outcome.items()}
        # The output label follows the configured window length.
        record['strict_stabilized_last_window'] = record.pop('strict_stabilized_last50')
        if not all(math.isfinite(v) for k, v in record.items() if k != 'time_to_band'):
            raise FloatingPointError(f'Non-finite rollout result in episode {i}')
        record.update(episode_index=i, grid_index=int(grid_index), selector_score=float(scores[grid_index]),
                      physical_h_margin0=float(physical_h[grid_index]), policy_key_uint32=np.asarray(key).tolist())
        episodes.append(record)
        if len(episodes) % 100 == 0 or len(episodes) == sample_size:
            print(f'F16 rollouts: {len(episodes)}/{sample_size}', flush=True)
    safe = sum(int(e['strict_safe'] > 0.5) for e in episodes)
    stable = sum(int(e['strict_stabilized_last_window'] > 0.5) for e in episodes)
    joint = sum(int(e['strict_safe'] > 0.5 and e['strict_stabilized_last_window'] > 0.5) for e in episodes)
    crashed = sum(int(e['crashed'] > 0.5) for e in episodes)
    dwell = sum(int(e['reached_goal_dwell'] > 0.5) for e in episodes)
    stable_any = sum(int(e['strict_stabilized_any_window'] > 0.5) for e in episodes)
    rate = lambda n: n / sample_size if sample_size else None
    source_hashes = {}
    for module in tuple(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).suffix == '.py' and Path(file).resolve().is_relative_to(source_root):
            file = Path(file).resolve()
            source_hashes[str(file.relative_to(source_root))] = sha256(file)
    step_match = re.fullmatch(r'step_(\d+)\.pkl', a.checkpoint.name)
    resolved = dict(protocol, action='one raw DDPM policy action; checkpoint temperature',
        classifier=('online Qh(s, pi(s)) < 0; one policy sample' if agent.stage_mode == 'stage_a' else
                    'online min-candidate Qh < 0; checkpoint candidate source, K and sigma'),
        stage_mode=agent.stage_mode, physical_h_prefilter=False,
        qh_candidate_source=agent.qh_candidate_source, qh_min_k_samples=agent.qh_min_k_samples,
        qh_min_k_sigma=agent.qh_min_k_sigma,
        gate_rng='PRNGKey(gate_seed) carried between sequential chunks; chunk size is part of the protocol',
        sample_rule='default_rng(init_seed).choice(predicted_safe_count, min(requested,count), replace=False)',
        policy_rng='fold_in(PRNGKey(policy_seed), episode_index), independent of gate stream',
        safe_label='initial state and every transition physically safe, with no crash',
        stabilization_label='final last_n transitions in goal altitude band, with no crash; independent of whole-trajectory safety',
        consecutive_stabilization_label='at least last_n consecutive in-band transitions anywhere in the episode, with no crash during the entire episode',
        reached_goal_dwell_label='at least last_n consecutive in-band transitions before termination; may subsequently crash',
        strict_invalid_dynamics=strict, safety_geometry=safety_geometry,
        resolved_safety_bounds={k: float(getattr(env, k)) for k in
            ("safe_h_min", "safe_h_max", "safe_alpha_lo", "safe_alpha_hi", "safe_beta", "safe_theta", "h_theta_denom")},
        checkpoint_selection='none; explicit single checkpoint',
        reference_scope=protocol.get('scope', 'conditional theta/height-grid evaluation'))
    out = dict(schema='f16-conditional-v2', status='evaluated', task='f16',
        checkpoint=dict(path=str(a.checkpoint), sha256=sha256(a.checkpoint), policy=policy_name, config=config,
                        train_seed=config.get('seed'), step=int(step_match.group(1)) if step_match else None,
                        step_source='filename' if step_match else 'unknown'),
        protocol=resolved, protocol_file_sha256=sha256(a.protocol),
        source=dict(evaluator_sha256=sha256(__file__), loaded_python_sha256=source_hashes),
        numerics=dict(backend=jax.default_backend(), jax_version=jax.__version__, numpy_version=np.__version__,
                      jax_enable_x64=bool(jax.config.jax_enable_x64), matmul_precision=str(jax.config.jax_default_matmul_precision)),
        full_grid=dict(population_size=len(states), predicted_safe_count=len(accepted), coverage=len(accepted)/len(states),
                       definition='classifier acceptance fraction over full grid, not recall',
                       flatten_order='meshgrid(theta,height), C order; nominal_state_v5 other coordinates',
                       grid_states_dtype=str(states.dtype), grid_states_shape=list(states.shape),
                       grid_states_sha256=hashlib.sha256(states.tobytes(order='C')).hexdigest(),
                       qh_scores=scores.tolist(), physical_h=physical_h.tolist(), accepted_grid_indices=accepted.tolist()),
        accepted_rollout_sample=dict(episodes_requested=protocol['episodes'], population_size=sample_size,
            conditional_rates_status='defined' if sample_size else 'undefined: no predicted-safe states',
            safe_count=safe, unsafe_count=sample_size-safe, safety_precision=rate(safe), false_safe=rate(sample_size-safe),
            stabilized_count=stable, stabilization_rate=rate(stable),
            consecutive_stabilized_count=stable_any, consecutive_stabilization_rate=rate(stable_any),
            reached_goal_dwell_count=dwell, reached_goal_dwell_rate=rate(dwell), safe_and_stabilized_count=joint,
            safe_and_stabilized_rate=rate(joint), crash_count=crashed, crash_rate=rate(crashed),
            definition='conditional sample estimates, not full-grid outcome counts'),
        sampled_accepted_offsets=offsets.tolist(), sampled_grid_indices=indices.tolist(),
        sampled_init_states=states[indices].tolist(), episodes=episodes,
        confusion_matrix=None, confusion_matrix_note='Rejected grid states were not rolled out; full-population TP/FP/FN/TN are not inferred.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(json_safe(out), f, indent=2, allow_nan=False); f.write('\n')
    print(f'Wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()

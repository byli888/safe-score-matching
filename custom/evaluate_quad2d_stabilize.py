"""Evaluate one trusted stabilize-avoid checkpoint on the three explicit start panels."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ssm.tasks.quad2d_stabilize.accelerated import build_rollout_fn_from_config
from ssm.tasks.quad2d_stabilize.env import FIXED_PAPER_START
from ssm.tasks.quad2d_stabilize.evaluation import (
    _convert_p123_episode, aggregate_rollouts_stab, load_agent_from_checkpoint_stab,
    sample_lowz_hrej_initial_states_stab,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_states(kind, spec, count, config):
    if kind == 'lower_region':
        return sample_lowz_hrej_initial_states_stab(
            config, episodes=count, init_seed=spec['init_seed'], z_high=float(spec['z_high']))
    base = np.array([FIXED_PAPER_START[k] for k in
                     ('init_x', 'init_vx', 'init_z', 'init_vz', 'init_theta', 'init_omega')], np.float32)
    states = np.repeat(base[None], count, axis=0)
    if kind == 'vel_small':
        rng = np.random.default_rng(spec['init_seed'])
        states[:, 1] += rng.uniform(*spec['vx'], count)
        states[:, 3] += rng.uniform(*spec['vz'], count)
    elif kind != 'exact':
        raise ValueError(f'Unknown panel: {kind}')
    return states


def evaluate_panel(loaded, kind, spec, count, horizon):
    states = initial_states(kind, spec, count, loaded.config)
    agent = loaded.agent.replace(rng=jax.random.PRNGKey(spec['base_policy_seed']))
    rollout = build_rollout_fn_from_config(loaded.config)
    traces, converted, episodes = [], [], []
    for i, state in enumerate(states):
        seeded = agent.replace(rng=jax.random.fold_in(agent.rng, spec['per_episode_fold_in_base'] + i))
        raw = jax.device_get(rollout(seeded, jnp.asarray(state), jnp.asarray(loaded.obs_mean),
                                   jnp.asarray(loaded.obs_var), horizon=horizon))
        traces.append({key: np.asarray(value) for key, value in raw.items() if key != 'agent'})
        r = _convert_p123_episode(raw, loaded.config)
        converted.append(r)
        q = r['episode_info']
        episodes.append(dict(episode=i, initial_state=state.tolist(), route=r['route'], reward=q['reward'],
                             cost=q['cost'], length=int(q['length']), success=bool(q['success'] > .5),
                             dwell_success=bool(q['dwell_success'] > .5), collision=bool(r['last_info']['collision']),
                             terminal_state_error_l2=q['terminal_state_error_norm']))
    arrays = {key: np.stack([trace[key] for trace in traces]) for key in traces[0]}
    arrays['initial_states'] = states
    costs = np.array([e['cost'] for e in episodes])
    success = np.array([e['success'] for e in episodes])
    collision = np.array([e['collision'] for e in episodes])
    routes = collections.Counter(e['route'] for e in episodes)
    summary = dict(n=count, mean_cost=float(costs.mean()), any_margin_violation_count=int((costs > 0).sum()),
                   collision_count=int(collision.sum()), success_count=int(success.sum()),
                   success_and_zero_margin_cost_count=int((success & (costs == 0)).sum()),
                   success_and_no_collision_count=int((success & ~collision).sum()),
                   route_counts_all_episodes=dict(routes), route_denominator=count,
                   left_count_all_episodes=sum(v for k, v in routes.items() if k.startswith('left')),
                   right_count_all_episodes=sum(v for k, v in routes.items() if k.startswith('right')))
    return dict(summary=summary, original_metrics=aggregate_rollouts_stab(converted, prefix=kind), episodes=episodes), arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    npz = a.out.with_suffix('.npz')
    if a.out == npz:
        p.error('--out must differ from its .npz companion; use a .json path.')
    for output in (a.out, npz):
        if output.exists():
            p.error(f'Refusing to overwrite existing output: {output}')
    doc = yaml.safe_load(a.protocol.read_text())
    if doc['task'] != 'quad2d_stabilize':
        p.error('Expected the quad2d_stabilize evaluation protocol.')
    spec = doc['protocol']
    loaded = load_agent_from_checkpoint_stab(str(a.checkpoint), eval_seed=0)
    if int(loaded.config['max_episode_steps']) != int(spec['horizon']):
        p.error('Protocol horizon must match the checkpoint environment horizon.')
    results, arrays = {}, {}
    for kind, panel in spec['panels'].items():
        results[kind], raw = evaluate_panel(loaded, kind, panel, int(spec['episodes']), int(spec['horizon']))
        arrays.update({f'{kind}/{key}': value for key, value in raw.items()})
        print(kind, json.dumps(results[kind]['summary'], sort_keys=True), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with npz.open('xb') as f:
        np.savez_compressed(f, **arrays)
    task_dir = Path(__file__).parent / 'ssm' / 'tasks' / 'quad2d_stabilize'
    sources = [Path(__file__), task_dir / 'env.py', task_dir / 'accelerated.py', task_dir / 'evaluation.py',
               Path(__file__).parent / 'ssm' / 'agents' / 'ssm_agent.py']
    output = dict(protocol=spec, task=doc['task'], checkpoint_config=loaded.config, panels=results,
                  definitions=dict(lower_region_scope='Historical lower-region panel used in checkpoint selection; not a held-out test.',
                                   cost='Sum of h_phys > 0 over scored transitions; 0.10 inflated obstacle/workspace margin.',
                                   collision='Nominal geometry collision, distinct from the margin cost.',
                                   success=f"Checkpoint success_mode={loaded.config.get('success_mode', 'full_state_terminal')}; terminal L2 radius={loaded.config.get('success_radius', 0.5)}; success also requires no nominal collision.",
                                   route='Label side_suffix: side at first z >= 0.62 crossing is center if |x| < 0.08, otherwise left for x < 0 or right; suffix uses max |x| among states with z < 1.12 (crossing |x| if none): outer if > 2.06, inner if > 1.00, else center. No crossing is none; denominator is all episodes.'),
                  identity=dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha256(a.checkpoint),
                                protocol_sha256=sha256(a.protocol), evaluator_sha256=sha256(__file__),
                                sources_sha256={str(f.relative_to(Path(__file__).parent)): sha256(f) for f in sources},
                                arrays_sha256=sha256(npz), backend=jax.default_backend(), jax_version=jax.__version__,
                                jax_threefry_partitionable=jax.config.jax_threefry_partitionable))
    serialized = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with a.out.open('x') as f:
        f.write(serialized)


if __name__ == '__main__':
    main()

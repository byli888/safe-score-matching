#!/usr/bin/env python3
"""Evaluate one trusted SSM Quad checkpoint on a fixed behavioral protocol.

Reports trajectory and initial-state safety-classification metrics. Checkpoints
use Python pickle and must come from a trusted source. Only inference state is restored.
"""
import argparse
import hashlib
import importlib
import json
import os
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SHAPES = {'quad2d': (100, 6, 360), 'quad3d': (500, 9, 500)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hidden(value):
    return tuple(int(v) for v in value.split(',')) if isinstance(value, str) else tuple(value)


def restore(checkpoint, task, core):
    import gym
    import jax
    import numpy as np
    with checkpoint.open('rb') as f:
        payload = pickle.load(f)
    config = dict(payload['config'])
    mean = np.asarray(payload['obs_mean'], dtype=np.float32)
    var = np.asarray(payload['obs_var'], dtype=np.float32)
    count = int(payload['obs_count'])
    obs_dim, act_dim = (12, 2) if task == 'quad2d' else (13, 4)
    if mean.shape != (obs_dim,) or var.shape != (obs_dim,):
        raise ValueError(f'{task}: wrong observation statistics shapes {mean.shape}/{var.shape}')
    if task == 'quad2d' and (not config.get('ref_velocity', True) or config.get('sdf_mode', 'baseline') != 'baseline'):
        raise ValueError('This matched Quad2D protocol requires ref_velocity and baseline SDF')
    if task == 'quad3d' and (config.get('obs_feature_mode') != 'h_components_closing' or config.get('sdf_mode', 'raw') != 'raw'):
        raise ValueError('This matched Quad3D protocol requires h_components_closing and raw SDF')
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
    kwargs = dict(seed=int(config['seed']), observation_space=obs_space, action_space=act_space,
                  actor_hidden_dims=hidden(config['actor_hidden']), critic_hidden_dims=hidden(config['critic_hidden']),
                  safety_estimator='expectile' if task == 'quad2d' else 'sampled_qh')
    for key in ('M_q', 'alpha_r', 'beta_safety', 'delta', 'discount', 'discount_h', 'tau', 'T',
                'beta_schedule', 'cost_critic_hyperparam', 'pure_qsm', 'ddpm_temperature',
                'fuse_clip_min', 'fuse_clip_max', 'huber_delta'):
        if key in config:
            kwargs[key] = config[key]
    if task == 'quad3d':
        kwargs.update(stage_mode=config['qh_stage_mode'], qh_candidate_source=config['qh_candidate_source'],
                      qh_min_k_samples=int(config['qh_min_k_samples']), qh_min_k_sigma=float(config['qh_min_k_sigma']))
    agent = core.SSMOnlineAgent.create(**kwargs)
    expected = {name for name in agent.__dataclass_fields__ if hasattr(getattr(agent, name), 'params')}
    saved = payload['params']
    if set(saved) != expected:
        raise ValueError(f'Network set mismatch: saved={sorted(saved)}, template={sorted(expected)}')
    replacements, network_shapes = {}, {}
    for name in sorted(expected):
        old_leaves, old_tree = jax.tree_util.tree_flatten(getattr(agent, name).params)
        leaves, tree = jax.tree_util.tree_flatten(saved[name])
        if tree != old_tree or [x.shape for x in leaves] != [x.shape for x in old_leaves]:
            raise ValueError(f'Network parameter shape/tree mismatch: {name}')
        replacements[name] = getattr(agent, name).replace(params=saved[name])
        network_shapes[name] = [list(x.shape) for x in leaves]
    # The checkpoint contains parameters/RNG/statistics, not optimizer state. Rollouts
    # use a separate fixed evaluation key, never the stored training key.
    agent = agent.replace(**replacements, rng=payload['rng'])
    loaded = SimpleNamespace(agent=agent, config=config, obs_mean=mean, obs_var=var,
                             obs_count=count, ckpt_path=str(checkpoint))
    return loaded, network_shapes




def confusion_rates(predicted_safe, actual_safe):
    """Classification rates; an empty predicted-safe set has undefined precision."""
    pred, truth = np.asarray(predicted_safe, bool), np.asarray(actual_safe, bool)
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    tn, fn = int((~pred & ~truth).sum()), int((~pred & truth).sum())
    accepted = tp + fp
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, population=int(pred.size),
                predicted_safe=accepted, coverage=accepted / pred.size,
                precision=tp / accepted if accepted else None,
                false_safe_rate=fp / accepted if accepted else None)


def initial_classifier_observations(loaded, task, states, helper, horizon):
    """Frozen-statistic observations for the supplied task's fixed start panel."""
    c = loaded.config
    if task == 'quad2d':
        waypoints = helper.make_waypoints(max_episode_steps=horizon)
        distances = states[:, None, [0, 2]] - waypoints[None, :, [0, 2]]
        nearest = np.argmin(np.sum(distances * distances, axis=-1), axis=1)
        raw = np.concatenate([states, waypoints[nearest]], axis=-1).astype(np.float32)
        if loaded.obs_count <= 0:
            return raw
        obs = (raw.astype(np.float64) - loaded.obs_mean) / np.sqrt(loaded.obs_var + 1e-8)
    else:
        # The supplied Quad3D protocol uses h_components_closing observations.
        norm = np.linalg.norm(states, axis=1)
        components = np.stack([states[:, 2], norm - 3.0], axis=-1).astype(np.float32)
        margin = np.max(components, axis=-1).astype(np.float32)
        derivative = np.zeros_like(states, dtype=np.float32)
        derivative[:, :3] = states[:, 3:6]
        radial_dot = np.sum(states * derivative, axis=1) / np.maximum(norm, 1e-8)
        hdot = np.where(components[:, 0] >= components[:, 1], states[:, 5], radial_dot)
        features = np.concatenate([
            np.tanh(margin[:, None] / float(c.get('obs_h_scale', 1.0))),
            np.tanh(components / float(c.get('obs_h_scale', 1.0))),
            np.tanh(hdot[:, None].astype(np.float32) / float(c.get('obs_hdot_scale', 1.0))),
        ], axis=-1).astype(np.float32)
        raw = np.concatenate([states.astype(np.float32), features], axis=-1)
        obs = (raw - loaded.obs_mean[None, :]) / np.sqrt(loaded.obs_var[None, :] + 1e-8)
    return np.clip(obs, -5.0, 5.0).astype(np.float32)


def classify_initial_states(loaded, task, states, helper, horizon, rollouts, spec):
    """Predict initial safety and compare it with the same scored trajectories."""
    import jax
    import jax.numpy as jnp
    from ssm.agents.ssm_agent import SSMOnlineAgent
    obs = initial_classifier_observations(loaded, task, states, helper, horizon)
    if task == 'quad2d':
        @jax.jit
        def predict(agent, observations):
            return agent.safe_value.apply_fn({'params': agent.safe_value.params}, observations)
        scores = np.asarray(predict(loaded.agent, jnp.asarray(obs))).reshape(-1)
        estimator = dict(name='online expectile value', score_seed_used=False)
    else:
        @jax.jit
        def predict(agent, observations, key):
            return SSMOnlineAgent._estimate_qh_from_candidates(
                agent, observations, key, params=agent.safe_critic.params, use_target=False)
        values, count, _ = predict(loaded.agent, jnp.asarray(obs), jax.random.PRNGKey(spec['score_seed']))
        scores = np.asarray(values, np.float32).reshape(-1)
        estimator = dict(name='online sampled Qh', stage=loaded.agent.stage_mode,
                         candidate_source=loaded.agent.qh_candidate_source,
                         realized_candidate_count=np.asarray(count).tolist(),
                         k=int(loaded.agent.qh_min_k_samples), sigma=float(loaded.agent.qh_min_k_sigma),
                         score_seed_used=True)
    if scores.shape != (len(states),) or not np.isfinite(scores).all():
        raise FloatingPointError('Expected one finite classifier score per initial state')
    pred = scores < float(spec['threshold'])
    valid = np.asarray(rollouts['valid'], bool)
    safe = ~(valid & (np.asarray(rollouts['h']) > 0)).any(axis=0)
    no_crash = ~(valid & np.asarray(rollouts['terminated'], bool)).any(axis=0)
    return dict(
        protocol=dict(spec, estimator=estimator, score_batch_size=len(states),
                      prediction='score < threshold; equality is predicted unsafe',
                      truth='No h > 0 over valid post-action transitions; t=0 is not included.',
                      interpretation='One stochastic rollout per state, not a reachability oracle.',
                      precision='TP / (TP + FP)', false_safe_rate='FP / (TP + FP)',
                      coverage='(TP + FP) / all panel states; not recall or state-space volume'),
        safety=confusion_rates(pred, safe), no_crash=confusion_rates(pred, no_crash),
        episodes=dict(score=scores.tolist(), predicted_safe=pred.tolist(),
                      actual_safe=safe.tolist(), actual_no_crash=no_crash.tolist()))


def evaluate(protocol_file, checkpoint):
    """Return the behavior result and raw rollout arrays for one checkpoint."""
    import jax
    import jax.numpy as jnp
    import ssm.agents.ssm_agent as core
    import jaxrl5.networks
    protocol_file, checkpoint = Path(protocol_file).resolve(), Path(checkpoint).resolve()
    recipe = yaml.safe_load(protocol_file.read_text())
    if recipe.get('backend') != 'custom' or recipe.get('module') != 'evaluate_quad':
        raise ValueError('Expected a custom/evaluate_quad protocol')
    task = recipe['task']; p = recipe['protocol']
    expected_n, state_dim, expected_horizon = TASK_SHAPES[task]
    n, horizon, policy_seed = int(p['episodes']), int(p['horizon']), int(p['policy_seed'])
    if (n, horizon, policy_seed) != (expected_n, expected_horizon, 20260504):
        raise ValueError('This evaluator uses the supplied fixed panel, horizon, and policy seed')
    initial_file = (protocol_file.parent / p['initial_states']).resolve()
    if sha(initial_file) != p['initial_states_sha256']:
        raise ValueError('Initial-state file does not match the protocol SHA-256')
    with np.load(initial_file, allow_pickle=False) as data:
        init_states = np.asarray(data['init_states'], np.float32)
    if init_states.shape != (n, state_dim) or not np.isfinite(init_states).all():
        raise ValueError('Invalid fixed initial-state array')
    match = re.fullmatch(r'step_(\d+)\.pkl', checkpoint.name)
    if not match:
        raise ValueError('Expected a checkpoint named step_<environment_steps>.pkl')
    loaded, networks = restore(checkpoint, task, core)
    cfg = loaded.config
    helper_name = 'ssm.tasks.' + ('quad2d_tracking' if task == 'quad2d' else 'quad3d') + '.evaluation'
    helper = importlib.import_module(helper_name)
    if task == 'quad2d':
        waypoints = helper.make_waypoints(max_episode_steps=horizon)
        ys = helper.rollout_batch_jax(
            loaded.agent, jnp.asarray(init_states), jnp.asarray(waypoints), jnp.asarray(loaded.obs_mean),
            jnp.asarray(loaded.obs_var), jax.random.PRNGKey(policy_seed), float(cfg['q_vel']),
            float(cfg['action_penalty']), 0, horizon, str(cfg.get('obs_feature_mode', 'state')),
            float(cfg.get('obs_h_scale', 1.0)), float(cfg.get('obs_hdot_scale', 1.0)),
            float(cfg.get('obs_ref_closing_scale', 0.75)))[1]
        rollouts = {k: np.asarray(v) for k, v in ys.items()}
        summary = helper.aggregate_collab_metrics(rollouts, horizon)
    else:
        rollouts = helper.run_rollouts_v2(loaded, init_states, policy_seed=policy_seed, horizon=horizon)
        summary = helper.rollout_l1_metrics(rollouts, float(cfg['target_pz']))
    valid = rollouts['valid'].astype(bool)
    episode = {'reward': rollouts['reward'].sum(axis=0), 'cost': rollouts['violation'].sum(axis=0),
               'length': valid.sum(axis=0), 'crash': rollouts['terminated'].any(axis=0),
               'any_violation': rollouts['violation'].any(axis=0)}
    if task == 'quad2d':
        episode['tracking_error'] = rollouts['unweighted_error'].sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
    else:
        target = np.zeros(state_dim); target[2] = float(cfg['target_pz'])
        delta = rollouts['states'][-1].astype(np.float64) - target
        episode['terminal_l1'] = np.abs(delta).sum(axis=1)
        episode['terminal_l2'] = np.linalg.norm(delta, axis=1)
        # Crashed rollouts retain their terminal state through the fixed horizon.
        final_window = rollouts['states'][-50:].astype(np.float64) - target
        episode['final50_l1'] = np.abs(final_window).sum(axis=2).mean(axis=0)
        summary['final50_l1'] = float(episode['final50_l1'].mean())
        summary['final50_l1_std'] = float(episode['final50_l1'].std())
    if not all(np.isfinite(v).all() for v in episode.values()):
        raise FloatingPointError('Non-finite per-episode behavior result')
    classification = (classify_initial_states(loaded, task, init_states, helper, horizon, rollouts, p['classification'])
                      if 'classification' in p else None)
    # Record the code actually imported, not only the requested module names.
    imports = {}
    for name, mod in sorted(sys.modules.items()):
        if name == 'ssm' or name.startswith('ssm.') or name == 'jaxrl5' or name.startswith('jaxrl5.'):
            path = getattr(mod, '__file__', None)
            if path:
                path = Path(path).resolve()
                if not path.is_relative_to(REPO_ROOT / 'custom'):
                    raise RuntimeError(f'Unexpected external repository import: {name} from {path}')
                imports[name] = {'file': str(path), 'sha256': sha(path)}
    evaluator_files = [Path(__file__).resolve(), Path(helper.__file__).resolve()]
    result = dict(
        schema='ssm_quad_behavior_v1', status='evaluated',
        protocol=dict(p, task=task, action='one raw DDPM sample at the checkpoint temperature; no selector',
                      checkpoint_selection='none; caller supplies one checkpoint',
                      learned_initial_state_filter=False, include_crashed_episodes=True,
                      uncertainty='summary standard deviations are across evaluation episodes, ddof=0',
                      scope='trajectory and classification metrics on the supplied fixed panel',
                      final50_l1_definition=('Mean full-state L1 over the last 50 post-action states of the fixed horizon; '
                                              'terminal state repeated after a crash, all episodes included.') if task == 'quad3d' else None),
        summary=summary, episodes={k: v.tolist() for k, v in episode.items()},
        restoration=dict(networks=networks, obs_mean=loaded.obs_mean.tolist(), obs_var=loaded.obs_var.tolist(),
                         obs_count=loaded.obs_count, optimizer_state='not stored in inference checkpoints'),
        numerics=dict(backend=jax.default_backend(), jax_version=jax.__version__, python=sys.executable,
                      checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                      checkpoint_step_from_filename=int(match.group(1)), train_seed=int(cfg['seed']),
                      protocol_file=str(protocol_file), protocol_sha256=sha(protocol_file),
                      initial_states_file=str(initial_file), initial_states_sha256=sha(initial_file),
                      evaluator_sha256={str(p.relative_to(REPO_ROOT)): sha(p) for p in evaluator_files},
                      actual_imports=imports, jax_enable_x64=bool(jax.config.jax_enable_x64),
                      matmul_precision=str(jax.config.jax_default_matmul_precision)),
        policy_config={k: cfg[k] for k in ('seed','actor_hidden','critic_hidden','T','ddpm_temperature','beta_schedule') if k in cfg})
    if classification is not None:
        result['classification'] = classification
    return result, rollouts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(f'Refusing to overwrite {a.out}')
    result, _ = evaluate(a.protocol, a.checkpoint)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    print(json.dumps({'output': str(a.out.resolve()), 'summary': result['summary']}), flush=True)


if __name__ == '__main__':
    main()

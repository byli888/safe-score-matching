"""Fixed-panel Quad3D behavior rollout and terminal L1/L2 metrics."""
from __future__ import annotations
from functools import partial
from typing import Dict, Optional
import jax
import jax.numpy as jnp
import numpy as np
from jaxrl5.networks import ddpm_sampler

from types import SimpleNamespace as LoadedQuad3D
from ssm.tasks.quad3d.env import GRAV, OBS_FEATURE_MODES
STATE_DIM = 9
DT = 0.01

def _axis_split_kwargs_from_config(config: Dict[str, object]) -> Dict[str, float]:
    keys = ("q_pos_xy", "q_pos_z", "q_vel_xy", "q_vel_z")
    values = {}
    for key in keys:
        value = config.get(key, None)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value >= 0.0:
            values[key] = value
    return values if len(values) == 4 else {}


def _reward_norm_code(value: str) -> int:
    return 1 if str(value).lower() == "l1" else 0


def _raw_h_components_jax(states):
    pz_margin = states[:, 2]
    radius_margin = jnp.linalg.norm(states, axis=-1) - 3.0
    components = jnp.stack([pz_margin, radius_margin], axis=-1)
    h_margin = jnp.max(components, axis=-1)
    return h_margin, components


def _obs_features_jax(
    states,
    obs_feature_mode: str,
    obs_h_scale,
    obs_hdot_scale,
    obs_reach_scale,
    obs_goal_pos_radius,
    obs_goal_vel_radius,
    obs_goal_angle_radius,
    target_pz,
):
    if obs_feature_mode == "state":
        return jnp.zeros((states.shape[0], 0), dtype=states.dtype)

    h_margin, components = _raw_h_components_jax(states)
    h_scaled = jnp.tanh(h_margin[:, None] / obs_h_scale)
    if obs_feature_mode == "h":
        return h_scaled

    comp_scaled = jnp.tanh(components / obs_h_scale)
    features = jnp.concatenate([h_scaled, comp_scaled], axis=-1)
    if obs_feature_mode in ("h_components_closing", "h_components_closing_reach"):
        pz_margin = components[:, 0]
        radius_margin = components[:, 1]
        norm = jnp.maximum(jnp.linalg.norm(states, axis=-1), 1e-8)
        approx_dot = jnp.zeros_like(states).at[:, 0:3].set(states[:, 3:6])
        radial_dot = jnp.sum(states * approx_dot, axis=-1) / norm
        hdot = jnp.where(pz_margin >= radius_margin, states[:, 5], radial_dot)
        hdot_scaled = jnp.tanh(hdot[:, None] / obs_hdot_scale)
        features = jnp.concatenate([features, hdot_scaled], axis=-1)

    if obs_feature_mode == "h_components_closing_reach":
        pos_ref = jnp.zeros_like(states[:, 0:3]).at[:, 2].set(target_pz)
        pos_err = states[:, 0:3] - pos_ref
        pos_norm = jnp.linalg.norm(pos_err, axis=-1)
        vel_norm = jnp.linalg.norm(states[:, 3:6], axis=-1)
        att_norm = jnp.linalg.norm(states[:, 6:9], axis=-1)
        v_toward_origin = -jnp.sum(pos_err * states[:, 3:6], axis=-1) / jnp.maximum(pos_norm, 1e-6)
        reach = jnp.stack(
            [
                pos_norm / (1.5 * obs_reach_scale),
                (obs_goal_pos_radius - pos_norm) / (0.3 * obs_reach_scale),
                (obs_goal_vel_radius - vel_norm) / (0.3 * obs_reach_scale),
                (obs_goal_angle_radius - att_norm) / (0.1 * obs_reach_scale),
                v_toward_origin / (0.5 * obs_reach_scale),
            ],
            axis=-1,
        )
        reach_scaled = jnp.tanh(reach)
        features = jnp.concatenate([features, reach_scaled], axis=-1)
    elif obs_feature_mode not in ("h_components", "h_components_closing"):
        raise ValueError(f"Unknown obs_feature_mode={obs_feature_mode!r}; expected one of {OBS_FEATURE_MODES}")
    return features


def _build_obs_raw_jax(
    states,
    obs_feature_mode: str,
    obs_h_scale,
    obs_hdot_scale,
    obs_reach_scale,
    obs_goal_pos_radius,
    obs_goal_vel_radius,
    obs_goal_angle_radius,
    target_pz,
):
    features = _obs_features_jax(
        states,
        obs_feature_mode,
        obs_h_scale,
        obs_hdot_scale,
        obs_reach_scale,
        obs_goal_pos_radius,
        obs_goal_vel_radius,
        obs_goal_angle_radius,
        target_pz,
    )
    return jnp.concatenate([states, features], axis=-1)


@partial(jax.jit, static_argnames=("horizon", "obs_feature_mode", "actor_kind"))
def rollout_batch_jax(
    agent,
    init_states,
    obs_mean,
    obs_var,
    rng,
    q_diag,
    x_ref,
    action_penalty,
    reward_norm_code,
    success_radius,
    obs_feature_mode: str,
    obs_h_scale,
    obs_hdot_scale,
    obs_reach_scale,
    obs_goal_pos_radius,
    obs_goal_vel_radius,
    obs_goal_angle_radius,
    actor_kind: str,
    horizon: int,
):
    active0 = jnp.ones((init_states.shape[0],), dtype=bool)

    def step_fn(carry, _):
        state, active, rng = carry
        obs_raw = _build_obs_raw_jax(
            state,
            obs_feature_mode,
            obs_h_scale,
            obs_hdot_scale,
            obs_reach_scale,
            obs_goal_pos_radius,
            obs_goal_vel_radius,
            obs_goal_angle_radius,
            x_ref[2],
        )
        obs = (obs_raw - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        obs = jnp.clip(obs, -5.0, 5.0)
        if actor_kind == "gaussian":
            dist = agent.score_model.apply_fn({"params": agent.score_model.params}, obs)
            actions = dist.mode()
        else:
            actions, rng = ddpm_sampler(
                agent.score_model.apply_fn,
                agent.score_model.params,
                agent.T,
                rng,
                agent.act_dim,
                obs,
                agent.alphas,
                agent.alpha_hats,
                agent.betas,
                agent.ddpm_temperature,
                0,
                agent.clip_sampler,
            )
        a = jnp.clip(actions, -1.0, 1.0)
        thrust = GRAV * (1.0 + a[:, 0])
        phi_dot = 5.0 * a[:, 1]
        theta_dot = 5.0 * a[:, 2]
        psi_dot = 5.0 * a[:, 3]

        px, py, pz, vx, vy, vz, phi, theta, psi = [state[:, i] for i in range(STATE_DIM)]
        s_theta, c_theta = jnp.sin(theta), jnp.cos(theta)
        s_phi, c_phi = jnp.sin(phi), jnp.cos(phi)
        xdot = jnp.stack(
            [
                vx,
                vy,
                vz,
                -thrust * s_theta,
                thrust * c_theta * s_phi,
                GRAV - thrust * c_theta * c_phi,
                phi_dot,
                theta_dot,
                psi_dot,
            ],
            axis=-1,
        )
        next_state_candidate = state + DT * xdot
        next_state = jnp.where(active[:, None], next_state_candidate, state)

        err = next_state - x_ref
        l2_reward = -(jnp.sum(jnp.square(err) * q_diag[None, :], axis=-1)
                      + action_penalty * jnp.sum(jnp.square(a), axis=-1))
        l1_reward = -(jnp.sum(jnp.abs(err) * q_diag[None, :], axis=-1)
                      + action_penalty * jnp.sum(jnp.abs(a), axis=-1))
        reward = jnp.where(reward_norm_code == 1, l1_reward, l2_reward)

        state_norm = jnp.linalg.norm(next_state, axis=-1)
        h = jnp.maximum(next_state[:, 2], state_norm - 3.0)
        violation = h > 0.0
        terminated = (next_state[:, 2] >= 0.3) | (state_norm >= 3.5)
        terminal_error = jnp.linalg.norm(next_state - x_ref, axis=-1)
        success = (terminal_error < success_radius) & (~terminated)
        valid = active
        next_active = active & (~terminated)

        out = {
            "states": next_state,
            "actions": a,
            "reward": jnp.where(valid, reward, 0.0),
            "h": h,
            "violation": valid & violation,
            "valid": valid,
            "terminated": valid & terminated,
            "success": valid & success,
        }
        return (next_state, next_active, rng), out

    _, ys = jax.lax.scan(
        step_fn,
        (jnp.asarray(init_states), active0, rng),
        xs=None,
        length=horizon,
    )
    return ys


def _q_diag_from_config(config: Dict[str, object]) -> np.ndarray:
    split = _axis_split_kwargs_from_config(config)
    q_ang = float(config.get("q_ang", 0.2))
    q_psi = float(config.get("q_psi_effective", config.get("q_psi", q_ang)))
    if q_psi < 0.0:
        q_psi = q_ang
    if split:
        return np.array(
            [
                split["q_pos_xy"], split["q_pos_xy"], split["q_pos_z"],
                split["q_vel_xy"], split["q_vel_xy"], split["q_vel_z"],
                q_ang, q_ang, q_psi,
            ],
            dtype=np.float32,
        )
    q_pos = float(config.get("q_pos", 10.0))
    q_vel = float(config.get("q_vel", 1.0))
    return np.array([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_ang, q_ang, q_psi], dtype=np.float32)


def run_rollouts_v2(
    loaded: LoadedQuad3D,
    init_states: np.ndarray,
    policy_seed: int = 0,
    horizon: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    config = loaded.config
    horizon = int(horizon or config.get("max_episode_steps", 500) or 500)
    x_ref = np.zeros(STATE_DIM, dtype=np.float32)
    x_ref[2] = float(config.get("target_pz", 0.0))
    ys = rollout_batch_jax(
        loaded.agent,
        jnp.asarray(init_states, dtype=jnp.float32),
        jnp.asarray(loaded.obs_mean, dtype=jnp.float32),
        jnp.asarray(loaded.obs_var, dtype=jnp.float32),
        jax.random.PRNGKey(policy_seed),
        jnp.asarray(_q_diag_from_config(config), dtype=jnp.float32),
        jnp.asarray(x_ref, dtype=jnp.float32),
        float(config.get("action_penalty", 1e-3)),
        _reward_norm_code(str(config.get("reward_norm", "l2"))),
        float(config.get("success_radius", 0.5)),
        str(config.get("obs_feature_mode", "state")),
        float(config.get("obs_h_scale", 1.0)),
        float(config.get("obs_hdot_scale", 1.0)),
        float(config.get("obs_reach_scale", 1.0)),
        float(config.get("obs_goal_pos_radius", 0.30)),
        float(config.get("obs_goal_vel_radius", 0.30)),
        float(config.get("obs_goal_angle_radius", 0.20)),
        "diffusion",
        horizon,
    )
    return {key: np.asarray(value) for key, value in ys.items()}


def rollout_l1_metrics(rollouts: Dict[str, np.ndarray], target_pz: float) -> Dict[str, float]:
    final_states = np.asarray(rollouts["states"][-1], dtype=np.float64)
    x_ref = np.zeros(final_states.shape[1], dtype=np.float64)
    x_ref[2] = target_pz
    delta = final_states - x_ref[None, :]
    terminal_l1 = np.sum(np.abs(delta), axis=1)
    terminal_l2 = np.linalg.norm(delta, axis=1)
    terminated = np.asarray(rollouts["terminated"], dtype=bool).any(axis=0)
    violation = np.asarray(rollouts["violation"], dtype=bool)
    valid = np.asarray(rollouts["valid"], dtype=bool)
    valid_steps = max(float(valid.sum()), 1.0)
    costs = violation.sum(axis=0).astype(np.float64)

    return {
        "terminal_l1": float(np.mean(terminal_l1)),
        "terminal_l1_std": float(np.std(terminal_l1)),
        "terminal_l2": float(np.mean(terminal_l2)),
        "terminal_l2_std": float(np.std(terminal_l2)),
        "crash_frac": float(np.mean(terminated)),
        "episode_violation_frac": float(np.mean(violation.any(axis=0))),
        "violation_rate": float(violation.sum() / valid_steps),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
    }

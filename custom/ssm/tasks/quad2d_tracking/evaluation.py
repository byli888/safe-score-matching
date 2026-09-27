"""Fixed-panel Quad2D behavior rollout and metrics. No checkpoint selection."""
from __future__ import annotations
from functools import partial
from typing import Dict
import jax
import jax.numpy as jnp
import numpy as np
from jaxrl5.networks import ddpm_sampler

STATE_DIM = 6
OBS_FEATURE_MODES = ("state", "h", "h_components", "h_components_closing", "h_components_refsafe", "h_components_closing_refsafe")

@jax.jit
def _nearest_waypoint_indices_jax(states, waypoints):
    state_xz = jnp.take(states, jnp.array([0, 2]), axis=1)
    waypoint_xz = jnp.take(waypoints, jnp.array([0, 2]), axis=1)
    deltas = state_xz[:, None, :] - waypoint_xz[None, :, :]
    return jnp.argmin(jnp.sum(jnp.square(deltas), axis=-1), axis=1)


def _sdf_jax(x, z, mode_code):
    h_z_low = 0.5 - z
    h_z_high = z - 1.5
    h_altitude = jnp.maximum(h_z_low, h_z_high)
    h_lateral = jnp.abs(x) - 2.0
    h_vertical_oob = jnp.abs(z) - 3.0
    h_raw = jnp.maximum(jnp.maximum(h_altitude, h_lateral), h_vertical_oob)

    def baseline(h):
        return h

    def hard1(h):
        return jnp.where(h > 0.0, 1.0, h)

    def hard5(h):
        return jnp.where(h > 0.0, 5.0, h)

    def hard10(h):
        return jnp.where(h > 0.0, 10.0, h)

    def hard5_scale10(h):
        return jnp.where(h > 0.0, 5.0, h) * 10.0

    return jax.lax.switch(mode_code, (baseline, hard1, hard5, hard10, hard5_scale10), h_raw)


def _raw_h_components_jax(states):
    x = states[:, 0]
    z = states[:, 2]
    h_z_low = 0.5 - z
    h_z_high = z - 1.5
    h_lateral = jnp.abs(x) - 2.0
    h_vertical_oob = jnp.abs(z) - 3.0
    components = jnp.stack([h_z_low, h_z_high, h_lateral, h_vertical_oob], axis=-1)
    h_margin = jnp.max(components, axis=-1)
    return h_margin, components


def _obs_features_jax(states, refs, obs_feature_mode: str, obs_h_scale, obs_hdot_scale, obs_ref_closing_scale):
    if obs_feature_mode == "state":
        return jnp.zeros((states.shape[0], 0), dtype=states.dtype)
    h_margin, components = _raw_h_components_jax(states)
    h_scaled = jnp.tanh(h_margin[:, None] / obs_h_scale)
    if obs_feature_mode == "h":
        return h_scaled
    comp_scaled = jnp.tanh(components / obs_h_scale)
    features = jnp.concatenate([h_scaled, comp_scaled], axis=-1)
    if obs_feature_mode in ("h_components_closing", "h_components_closing_refsafe"):
        dt = 1.0 / 60.0
        next_states = states.at[:, 0].set(states[:, 0] + states[:, 1] * dt)
        next_states = next_states.at[:, 2].set(states[:, 2] + states[:, 3] * dt)
        h_next, _ = _raw_h_components_jax(next_states)
        hdot = (h_next - h_margin) / dt
        hdot_scaled = jnp.tanh(hdot[:, None] / obs_hdot_scale)
        features = jnp.concatenate([features, hdot_scaled], axis=-1)
    if obs_feature_mode in ("h_components_refsafe", "h_components_closing_refsafe"):
        ref_h_margin, ref_components = _raw_h_components_jax(refs)
        ref_safety = jnp.stack(
            [ref_components[:, 0], ref_components[:, 1], ref_h_margin],
            axis=-1,
        )
        ref_safety_scaled = jnp.tanh(ref_safety / obs_h_scale)
        dx = states[:, 0] - refs[:, 0]
        dz = states[:, 2] - refs[:, 2]
        dvx = states[:, 1] - refs[:, 1]
        dvz = states[:, 3] - refs[:, 3]
        dist = jnp.maximum(jnp.sqrt(dx * dx + dz * dz), 1e-6)
        v_toward_ref = -(dx * dvx + dz * dvz) / dist
        v_toward_scaled = jnp.tanh(v_toward_ref[:, None] / obs_ref_closing_scale)
        features = jnp.concatenate([features, ref_safety_scaled, v_toward_scaled], axis=-1)
    elif obs_feature_mode not in ("h_components", "h_components_closing"):
        raise ValueError(f"Unknown obs_feature_mode={obs_feature_mode!r}; expected one of {OBS_FEATURE_MODES}")
    return features


def _build_obs_raw_jax(states, refs, obs_feature_mode: str, obs_h_scale, obs_hdot_scale, obs_ref_closing_scale):
    base = jnp.concatenate([states, refs], axis=-1)
    features = _obs_features_jax(states, refs, obs_feature_mode, obs_h_scale, obs_hdot_scale, obs_ref_closing_scale)
    return jnp.concatenate([base, features], axis=-1)


@partial(jax.jit, static_argnames=("horizon", "obs_feature_mode"))
def rollout_batch_jax(
    agent,
    init_states,
    waypoints,
    obs_mean,
    obs_var,
    rng,
    q_vel,
    action_penalty,
    sdf_code,
    horizon: int,
    obs_feature_mode: str = "state",
    obs_h_scale=1.0,
    obs_hdot_scale=1.0,
    obs_ref_closing_scale=0.75,
):
    wp0 = _nearest_waypoint_indices_jax(init_states, waypoints)
    active0 = jnp.ones((init_states.shape[0],), dtype=bool)

    q_diag = jnp.asarray([10.0, q_vel, 10.0, q_vel, 0.2, 0.2], dtype=jnp.float32)
    r_diag = jnp.asarray([action_penalty, action_penalty], dtype=jnp.float32)
    a_ref = jnp.asarray([0.5, 0.5], dtype=jnp.float32)

    def step_fn(carry, _):
        state, wp_idx, active, rng = carry
        ref = waypoints[wp_idx]
        obs_raw = _build_obs_raw_jax(state, ref, obs_feature_mode, obs_h_scale, obs_hdot_scale, obs_ref_closing_scale)
        obs = (obs_raw - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        obs = jnp.clip(obs, -5.0, 5.0)

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
        action_raw = (jnp.clip(actions, -1.0, 1.0) + 1.0) / 2.0

        x, xdot, z, zdot, theta, thetadot = [state[:, i] for i in range(STATE_DIM)]
        f1 = 9.81 * action_raw[:, 0]
        f2 = 9.81 * action_raw[:, 1]
        u1 = f1 + f2
        tau = 0.1 * (action_raw[:, 1] - action_raw[:, 0])
        dt = 1.0 / 60.0
        xdot_new = xdot + (-(u1) * jnp.sin(theta)) * dt
        x_new = x + xdot_new * dt
        zdot_new = zdot + ((u1) * jnp.cos(theta) - 9.81) * dt
        z_new = z + zdot_new * dt
        thetadot_new = thetadot + (tau / 0.02) * dt
        theta_new = theta + thetadot_new * dt
        next_state_candidate = jnp.stack(
            [x_new, xdot_new, z_new, zdot_new, theta_new, thetadot_new], axis=-1
        )
        next_state = jnp.where(active[:, None], next_state_candidate, state)

        next_wp = (wp_idx + active.astype(jnp.int32)) % waypoints.shape[0]
        ref_next = waypoints[next_wp]
        state_err = next_state - ref_next
        action_err = action_raw - a_ref
        reward = -(
            jnp.sum(jnp.square(state_err) * q_diag[None, :], axis=-1)
            + jnp.sum(jnp.square(action_err) * r_diag[None, :], axis=-1)
        )
        unweighted_error = jnp.sum(jnp.square(state_err), axis=-1)
        h = _sdf_jax(next_state[:, 0], next_state[:, 2], sdf_code)
        violation = h > 0.0
        terminated = (jnp.abs(next_state[:, 0]) > 2.0) | (jnp.abs(next_state[:, 2]) > 3.0)
        valid = active
        next_active = active & (~terminated)

        out = {
            "states": next_state,
            "refs": ref_next,
            "actions": actions,
            "reward": jnp.where(valid, reward, 0.0),
            "h": h,
            "violation": valid & violation,
            "valid": valid,
            "unweighted_error": jnp.where(valid, unweighted_error, 0.0),
            "terminated": valid & terminated,
        }
        return (next_state, next_wp, next_active, rng), out

    carry_final, ys = jax.lax.scan(
        step_fn,
        (jnp.asarray(init_states), wp0, active0, rng),
        xs=None,
        length=horizon,
    )
    return carry_final, ys


def aggregate_collab_metrics(rollouts: Dict[str, np.ndarray], horizon: int) -> Dict[str, float]:
    valid = rollouts["valid"].astype(bool)
    valid_counts = np.maximum(valid.sum(axis=0), 1)
    rewards = rollouts["reward"].sum(axis=0)
    costs = rollouts["violation"].sum(axis=0).astype(np.float64)
    tracking = rollouts["unweighted_error"].sum(axis=0) / valid_counts
    total_valid = max(float(valid.sum()), 1.0)
    violation_rate_actual_t = float(rollouts["violation"].sum() / total_valid)
    violation_rate_fixed_t = float(rollouts["violation"].sum() / (horizon * valid.shape[1]))
    terminated = rollouts["terminated"].any(axis=0)
    episode_has_cost = rollouts["violation"].any(axis=0)
    return {
        "num_episodes": float(valid.shape[1]),
        "horizon": float(horizon),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "cost_sum_mean": float(np.mean(costs)),
        "cost_sum_std": float(np.std(costs)),
        "cost_rate_actual_T": violation_rate_actual_t,
        "cost_rate_fixed_T": violation_rate_fixed_t,
        "episode_cost_rate": float(np.mean(episode_has_cost)),
        "tracking_error_mean": float(np.mean(tracking)),
        "tracking_error_std": float(np.std(tracking)),
        "crash_frac": float(np.mean(terminated)),
        "episode_length_mean": float(np.mean(valid_counts)),
    }


def make_waypoints(dt=1/60, max_episode_steps=360, center=(0.0, 1.0), radius=1.0):
    traj_length = max_episode_steps * dt
    traj_freq = 2.0 * np.pi / traj_length  # 1 cycle per episode

    times = np.arange(0, traj_length + dt, dt)  # 361 points

    x   =  radius * np.cos(traj_freq * times) + center[0]
    z   =  radius * np.sin(traj_freq * times) + center[1]
    vx  = -radius * traj_freq * np.sin(traj_freq * times)
    vz  =  radius * traj_freq * np.cos(traj_freq * times)

    zeros = np.zeros_like(times)
    return np.stack([x, vx, z, vz, zeros, zeros], axis=1).astype(np.float32)

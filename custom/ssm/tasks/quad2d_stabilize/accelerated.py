#!/usr/bin/env python3
"""P2/P3 acceleration helpers for Quad2D stabilize-avoid.

P2 is a jitted single-environment transition core with the same public API as
``Quad2DStabEnv``.  P3 is a scan/jit rollout backend used by eval, checkpoint
scans, and trajectory plotting.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from ssm.tasks.quad2d_stabilize.env import SUCCESS_MODES, Quad2DStabEnv, get_layout


ENV_BACKEND_P2 = "jit_core"


def _layout_arrays(layout_name: str):
    layout = get_layout(layout_name)
    centers = []
    half_sizes = []
    for obs in layout.obstacles:
        centers.append(obs.center)
        half_sizes.append((0.5 * obs.size[0], 0.5 * obs.size[1]))
    return {
        "xlim": jnp.asarray(layout.xlim, dtype=jnp.float32),
        "zlim": jnp.asarray(layout.zlim, dtype=jnp.float32),
        "centers": jnp.asarray(centers, dtype=jnp.float32),
        "half_sizes": jnp.asarray(half_sizes, dtype=jnp.float32),
        "obstacle_margin": float(layout.obstacle_margin),
        "boundary_margin": float(layout.boundary_margin),
        "goal": jnp.asarray(layout.goal, dtype=jnp.float32),
    }


def _rect_h_values_jax(x, z, centers, half_sizes, margin: float):
    pos = jnp.asarray([x, z], dtype=jnp.float32)
    q = jnp.abs(pos[None, :] - centers) - (half_sizes + jnp.asarray(margin, dtype=jnp.float32))
    outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
    inside = jnp.minimum(jnp.maximum(q[:, 0], q[:, 1]), 0.0)
    signed_distance = outside + inside
    return -signed_distance


def _workspace_h_value_jax(x, z, xlim, zlim, margin: float):
    return jnp.maximum(
        jnp.maximum(xlim[0] + margin - x, x - (xlim[1] - margin)),
        jnp.maximum(zlim[0] + margin - z, z - (zlim[1] - margin)),
    )


def _raw_workspace_h_value_jax(x, z, xlim, zlim):
    return jnp.maximum(
        jnp.maximum(xlim[0] - x, x - xlim[1]),
        jnp.maximum(zlim[0] - z, z - zlim[1]),
    )


def _physical_h_value_jax(state, xlim, zlim, centers, half_sizes, obstacle_margin: float, boundary_margin: float):
    x = state[0]
    z = state[2]
    rect_h = _rect_h_values_jax(x, z, centers, half_sizes, obstacle_margin)
    boundary_h = _workspace_h_value_jax(x, z, xlim, zlim, boundary_margin)
    return jnp.maximum(jnp.max(rect_h), boundary_h)


def _physical_h_components_jax(state, xlim, zlim, centers, half_sizes, obstacle_margin: float, boundary_margin: float):
    x = state[0]
    z = state[2]
    rect_h = _rect_h_values_jax(x, z, centers, half_sizes, obstacle_margin)
    boundary_h = _workspace_h_value_jax(x, z, xlim, zlim, boundary_margin)
    return jnp.concatenate([rect_h, jnp.asarray([boundary_h], dtype=jnp.float32)], axis=0)


def _nominal_collision_h_value_jax(state, xlim, zlim, centers, half_sizes):
    x = state[0]
    z = state[2]
    rect_h = _rect_h_values_jax(x, z, centers, half_sizes, 0.0)
    boundary_h = _raw_workspace_h_value_jax(x, z, xlim, zlim)
    return jnp.maximum(jnp.max(rect_h), boundary_h)


def _shape_h_jax(h_phys, sdf_mode: str, unsafe_const: float, unsafe_slope: float, global_scale: float):
    if sdf_mode == "raw":
        return h_phys
    if sdf_mode == "hard":
        return jnp.where(h_phys > 0.0, jnp.asarray(unsafe_const, dtype=jnp.float32), h_phys)
    if sdf_mode == "hard_slope":
        return jnp.where(
            h_phys > 0.0,
            jnp.asarray(unsafe_const, dtype=jnp.float32) + jnp.asarray(unsafe_slope, dtype=jnp.float32) * h_phys,
            h_phys,
        )
    if sdf_mode == "hard_scale":
        base = jnp.where(h_phys > 0.0, jnp.asarray(unsafe_const, dtype=jnp.float32), h_phys)
        return base * jnp.asarray(global_scale, dtype=jnp.float32)
    raise ValueError(f"Unsupported Quad2D-stab p123 sdf_mode={sdf_mode!r}")


@partial(
    jax.jit,
    static_argnames=(
        "reward_norm",
        "sdf_mode",
        "terminate_on_success",
    ),
)
def _transition_core_jit(
    state,
    action,
    xlim,
    zlim,
    centers,
    half_sizes,
    obstacle_margin: float,
    boundary_margin: float,
    goal,
    q_diag,
    r_diag,
    reward_norm: str,
    goal_pos_radius: float,
    goal_vel_radius: float,
    goal_angle_radius: float,
    terminate_on_success: bool,
    sdf_mode: str,
    sdf_unsafe_const: float,
    sdf_unsafe_slope: float,
    sdf_global_scale: float,
    gate_lambda: float,
    gate_x_max: float,
    gate_z_center: float,
    gate_z_width: float,
    dwell_bonus: float,
    dt: float,
    m: float,
    inertia: float,
    g: float,
    thrust_scale: float,
    torque_scale: float,
):
    del terminate_on_success
    state = jnp.asarray(state, dtype=jnp.float32)
    action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
    action_raw = (action + 1.0) / 2.0

    x, xdot, z, zdot, theta, theta_dot = state
    f1 = thrust_scale * action_raw[0]
    f2 = thrust_scale * action_raw[1]
    total_thrust = f1 + f2
    torque = torque_scale * (action_raw[1] - action_raw[0])

    xddot = -(total_thrust / m) * jnp.sin(theta)
    zddot = (total_thrust / m) * jnp.cos(theta) - g
    theta_ddot = torque / inertia

    xdot = xdot + xddot * dt
    x = x + xdot * dt
    zdot = zdot + zddot * dt
    z = z + zdot * dt
    theta_dot = theta_dot + theta_ddot * dt
    theta = theta + theta_dot * dt
    next_state = jnp.asarray([x, xdot, z, zdot, theta, theta_dot], dtype=jnp.float32)

    h_phys = _physical_h_value_jax(
        next_state,
        xlim,
        zlim,
        centers,
        half_sizes,
        obstacle_margin,
        boundary_margin,
    )
    h_val = _shape_h_jax(h_phys, sdf_mode, sdf_unsafe_const, sdf_unsafe_slope, sdf_global_scale)
    h_nominal = _nominal_collision_h_value_jax(next_state, xlim, zlim, centers, half_sizes)
    binary_cost = jnp.asarray(h_phys > 0.0, dtype=jnp.float32)
    collision = h_nominal > 0.0

    x_ref = jnp.asarray([goal[0], 0.0, goal[1], 0.0, 0.0, 0.0], dtype=jnp.float32)
    a_ref = jnp.asarray([0.5, 0.5], dtype=jnp.float32)
    err = next_state - x_ref
    action_err = action_raw - a_ref
    if reward_norm == "l2":
        state_cost = jnp.sum(jnp.square(err) * q_diag)
        action_cost = jnp.sum(jnp.square(action_err) * r_diag)
    else:
        state_cost = jnp.sum(jnp.abs(err) * q_diag)
        action_cost = jnp.sum(jnp.abs(action_err) * r_diag)
    gate_profile = jnp.maximum(
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32) - jnp.abs(z - gate_z_center) / jnp.maximum(gate_z_width, 1e-6),
    )
    x_req = gate_x_max * gate_profile
    gate_cost = gate_lambda * jnp.square(jnp.maximum(jnp.asarray(0.0, dtype=jnp.float32), x_req - jnp.abs(x)))
    reward = -(state_cost + action_cost + gate_cost)

    pos_err = jnp.linalg.norm(next_state[jnp.asarray([0, 2])] - goal)
    vel_norm = jnp.linalg.norm(next_state[jnp.asarray([1, 3])])
    angle_ok = jnp.abs(next_state[4]) <= goal_angle_radius
    in_goal = (
        (pos_err <= goal_pos_radius)
        & (vel_norm <= goal_vel_radius)
        & angle_ok
        & (h_phys < 0.0)
    )
    reward = reward + jnp.asarray(dwell_bonus, dtype=jnp.float32) * jnp.asarray(in_goal, dtype=jnp.float32)

    return {
        "next_state": next_state,
        "action": action,
        "action_raw": action_raw,
        "reward": reward,
        "h": h_val,
        "h_phys": h_phys,
        "h_nominal": h_nominal,
        "binary_cost": binary_cost,
        "collision": collision,
        "in_goal": in_goal,
        "pos_error": pos_err,
        "vel_norm": vel_norm,
    }


@lru_cache(maxsize=None)
def _build_transition_core(
    layout_name: str,
    q_pos: float,
    q_x: float,
    q_z: float,
    q_vel: float,
    q_angle: float,
    action_penalty: float,
    reward_norm: str,
    goal_pos_radius: float,
    goal_vel_radius: float,
    goal_angle_radius: float,
    terminate_on_success: bool,
    sdf_mode: str,
    sdf_unsafe_const: float,
    sdf_unsafe_slope: float,
    sdf_global_scale: float,
    gate_lambda: float,
    gate_x_max: float,
    gate_z_center: float,
    gate_z_width: float,
    dwell_bonus: float,
    dt: float,
):
    layout = _layout_arrays(layout_name)
    del q_pos
    q_diag = jnp.asarray([q_x, q_vel, q_z, q_vel, q_angle, q_angle], dtype=jnp.float32)
    r_diag = jnp.asarray([action_penalty, action_penalty], dtype=jnp.float32)
    m = 1.0
    g = 9.81
    return partial(
        _transition_core_jit,
        xlim=layout["xlim"],
        zlim=layout["zlim"],
        centers=layout["centers"],
        half_sizes=layout["half_sizes"],
        obstacle_margin=layout["obstacle_margin"],
        boundary_margin=layout["boundary_margin"],
        goal=layout["goal"],
        q_diag=q_diag,
        r_diag=r_diag,
        reward_norm=reward_norm,
        goal_pos_radius=float(goal_pos_radius),
        goal_vel_radius=float(goal_vel_radius),
        goal_angle_radius=float(goal_angle_radius),
        terminate_on_success=bool(terminate_on_success),
        sdf_mode=sdf_mode,
        sdf_unsafe_const=float(sdf_unsafe_const),
        sdf_unsafe_slope=float(sdf_unsafe_slope),
        sdf_global_scale=float(sdf_global_scale),
        gate_lambda=float(gate_lambda),
        gate_x_max=float(gate_x_max),
        gate_z_center=float(gate_z_center),
        gate_z_width=float(gate_z_width),
        dwell_bonus=float(dwell_bonus),
        dt=float(dt),
        m=m,
        inertia=0.02,
        g=g,
        thrust_scale=m * g,
        torque_scale=0.1,
    )


def _transition_core_from_config(config: Dict[str, object]):
    q_pos = float(config.get("q_pos", 10.0))
    return _build_transition_core(
        str(config.get("layout_name", "corridor_v2")),
        q_pos,
        float(config.get("q_x", q_pos)),
        float(config.get("q_z", q_pos)),
        float(config.get("q_vel", 1.0)),
        float(config.get("q_angle", 0.2)),
        float(config.get("action_penalty", 1e-3)),
        str(config.get("reward_norm", "l2")),
        float(config.get("goal_pos_radius", 0.30)),
        float(config.get("goal_vel_radius", 0.45)),
        float(config.get("goal_angle_radius", 0.25)),
        bool(config.get("terminate_on_success", False)),
        str(config.get("sdf_mode", "raw")),
        float(config.get("sdf_unsafe_const", 0.0)),
        float(config.get("sdf_unsafe_slope", 0.0)),
        float(config.get("sdf_global_scale", 0.0)),
        float(config.get("gate_lambda", 0.0)),
        float(config.get("gate_x_max", 0.90)),
        float(config.get("gate_z_center", 0.30)),
        float(config.get("gate_z_width", 1.20)),
        float(config.get("dwell_bonus", 0.0)),
        float(config.get("dt", 1.0 / 60.0)),
    )


def _raw_obs_jax(
    state,
    xlim,
    zlim,
    centers,
    half_sizes,
    obstacle_margin: float,
    boundary_margin: float,
    goal,
    goal_pos_radius: float,
    goal_vel_radius: float,
    goal_angle_radius: float,
    obs_feature_mode: str,
    obs_h_scale: float,
    obs_hdot_scale: float,
    dt: float,
):
    state = jnp.asarray(state, dtype=jnp.float32)
    if obs_feature_mode == "state":
        return state

    components = _physical_h_components_jax(state, xlim, zlim, centers, half_sizes, obstacle_margin, boundary_margin)
    h_margin = jnp.max(components)
    h_scaled = jnp.tanh(jnp.reshape(h_margin, (1,)) / jnp.asarray(obs_h_scale, dtype=jnp.float32))
    if obs_feature_mode == "h":
        return jnp.concatenate([state, h_scaled], axis=0)

    comp_scaled = jnp.tanh(components / jnp.asarray(obs_h_scale, dtype=jnp.float32))
    features = jnp.concatenate([h_scaled, comp_scaled], axis=0)
    if obs_feature_mode in ("h_components_closing", "h_components_closing_reach"):
        future_state = state.at[0].set(state[0] + state[1] * dt)
        future_state = future_state.at[2].set(state[2] + state[3] * dt)
        h_next = _physical_h_value_jax(
            future_state, xlim, zlim, centers, half_sizes, obstacle_margin, boundary_margin
        )
        hdot = (h_next - h_margin) / jnp.maximum(jnp.asarray(dt, dtype=jnp.float32), jnp.asarray(1e-8, dtype=jnp.float32))
        hdot_scaled = jnp.tanh(jnp.reshape(hdot, (1,)) / jnp.asarray(obs_hdot_scale, dtype=jnp.float32))
        features = jnp.concatenate([features, hdot_scaled], axis=0)
    if obs_feature_mode == "h_components_closing_reach":
        dx = state[0] - goal[0]
        dz = state[2] - goal[1]
        dist = jnp.sqrt(jnp.square(dx) + jnp.square(dz))
        vel_norm = jnp.sqrt(jnp.square(state[1]) + jnp.square(state[3]))
        v_toward_goal = -(dx * state[1] + dz * state[3]) / jnp.maximum(dist, 1e-6)
        angle_margin = jnp.asarray(goal_angle_radius, dtype=jnp.float32) - jnp.abs(state[4])
        reach_features = jnp.tanh(
            jnp.stack(
                [
                    dist / 1.50,
                    (jnp.asarray(goal_pos_radius, dtype=jnp.float32) - dist) / 0.30,
                    (jnp.asarray(goal_vel_radius, dtype=jnp.float32) - vel_norm) / 0.30,
                    angle_margin / 0.10,
                    v_toward_goal / 0.75,
                ],
                axis=0,
            )
        )
        features = jnp.concatenate([features, reach_features], axis=0)
    return jnp.concatenate([state, features], axis=0)


def _normalize_raw_obs_jax(obs, obs_mean, obs_var):
    obs_mean = jnp.asarray(obs_mean, dtype=jnp.float32)
    obs_var = jnp.asarray(obs_var, dtype=jnp.float32)
    obs = (jnp.asarray(obs, dtype=jnp.float32) - obs_mean) / jnp.sqrt(obs_var + 1e-8)
    return jnp.clip(obs, -5.0, 5.0)


def _build_obs_normalizer(
    layout_name: str,
    obs_feature_mode: str,
    obs_h_scale: float,
    obs_hdot_scale: float,
    dt: float,
    goal_pos_radius: float,
    goal_vel_radius: float,
    goal_angle_radius: float,
):
    layout = _layout_arrays(layout_name)
    mode = str(obs_feature_mode)
    if mode not in ("state", "h", "h_components", "h_components_closing", "h_components_closing_reach"):
        raise ValueError(f"Unsupported obs_feature_mode={mode!r}")

    def normalize(state, obs_mean, obs_var):
        raw_obs = _raw_obs_jax(
            state,
            layout["xlim"],
            layout["zlim"],
            layout["centers"],
            layout["half_sizes"],
            layout["obstacle_margin"],
            layout["boundary_margin"],
            layout["goal"],
            float(goal_pos_radius),
            float(goal_vel_radius),
            float(goal_angle_radius),
            mode,
            float(obs_h_scale),
            float(obs_hdot_scale),
            float(dt),
        )
        return _normalize_raw_obs_jax(raw_obs, obs_mean, obs_var)

    return normalize


def _inactive_step_out(state, act_dim: int):
    zeros_action = jnp.zeros((act_dim,), dtype=jnp.float32)
    return {
        "state": state,
        "action": zeros_action,
        "next_state": state,
        "reward": jnp.asarray(0.0, dtype=jnp.float32),
        "h": jnp.asarray(0.0, dtype=jnp.float32),
        "h_phys": jnp.asarray(0.0, dtype=jnp.float32),
        "h_nominal": jnp.asarray(0.0, dtype=jnp.float32),
        "binary_cost": jnp.asarray(0.0, dtype=jnp.float32),
        "collision": jnp.asarray(False),
        "in_goal": jnp.asarray(False),
        "pos_error": jnp.asarray(0.0, dtype=jnp.float32),
        "vel_norm": jnp.asarray(0.0, dtype=jnp.float32),
    }


def _rollout_episode_scan_core(
    agent,
    init_state,
    obs_mean,
    obs_var,
    goal,
    horizon: int,
    goal_dwell_steps: int,
    success_mode: str,
    success_radius: float,
    terminate_on_success: bool,
    transition_core,
    normalize_obs_fn,
):
    if success_mode not in SUCCESS_MODES:
        raise ValueError(f"success_mode must be one of {SUCCESS_MODES}, got {success_mode!r}")
    state0 = jnp.asarray(init_state, dtype=jnp.float32)
    x_ref = jnp.asarray([goal[0], 0.0, goal[1], 0.0, 0.0, 0.0], dtype=jnp.float32)
    goal_buf0 = jnp.zeros((goal_dwell_steps,), dtype=jnp.bool_)
    carry0 = (
        agent,
        state0,
        jnp.asarray(0.0, dtype=jnp.float32),   # reward_sum
        jnp.asarray(0.0, dtype=jnp.float32),   # cost_sum
        jnp.asarray(0, dtype=jnp.int32),       # steps_taken
        jnp.asarray(0, dtype=jnp.int32),       # goal_streak
        jnp.asarray(0, dtype=jnp.int32),       # total_goal_steps
        jnp.asarray(0, dtype=jnp.int32),       # max_goal_streak
        goal_buf0,
        jnp.asarray(False),                    # success
        jnp.asarray(-1, dtype=jnp.int32),      # success_step
        jnp.asarray(False),                    # collision
        jnp.asarray(False),                    # done
    )

    def active_step(carry):
        (
            ep_agent,
            state,
            reward_sum,
            cost_sum,
            steps_taken,
            goal_streak,
            total_goal_steps,
            max_goal_streak,
            goal_buf,
            success,
            success_step,
            collision,
            _done,
        ) = carry
        obs = normalize_obs_fn(state, obs_mean, obs_var)
        action, ep_agent = ep_agent.eval_actions(obs)
        step_out = transition_core(state, action)

        in_goal = step_out["in_goal"]
        next_goal_streak = jnp.where(in_goal, goal_streak + 1, jnp.asarray(0, dtype=jnp.int32))
        next_total_goal_steps = total_goal_steps + jnp.asarray(in_goal, dtype=jnp.int32)
        next_max_goal_streak = jnp.maximum(max_goal_streak, next_goal_streak)
        next_goal_buf = jnp.concatenate([goal_buf[1:], jnp.asarray([in_goal])]) if goal_dwell_steps > 1 else jnp.asarray([in_goal])

        if success_mode == "dwell_box":
            success_now = (~success) & (next_goal_streak >= goal_dwell_steps)
        else:
            success_now = jnp.asarray(False)
        next_success = success | success_now
        next_success_step = jnp.where(success_now, steps_taken + 1, success_step)
        next_collision = collision | step_out["collision"]
        terminated = step_out["collision"] | (
            jnp.asarray(success_mode == "dwell_box")
            & jnp.asarray(terminate_on_success)
            & next_success
        )
        next_steps_taken = steps_taken + 1
        truncated = next_steps_taken >= horizon
        next_done = terminated | truncated

        next_carry = (
            ep_agent,
            step_out["next_state"],
            reward_sum + step_out["reward"],
            cost_sum + step_out["binary_cost"],
            next_steps_taken,
            next_goal_streak,
            next_total_goal_steps,
            next_max_goal_streak,
            next_goal_buf,
            next_success,
            next_success_step,
            next_collision,
            next_done,
        )
        trace_out = {
            "state": state,
            "action": step_out["action"],
            "next_state": step_out["next_state"],
            "reward": step_out["reward"],
            "h": step_out["h"],
            "h_phys": step_out["h_phys"],
            "h_nominal": step_out["h_nominal"],
            "binary_cost": step_out["binary_cost"],
            "collision": step_out["collision"],
            "in_goal": in_goal,
            "pos_error": step_out["pos_error"],
            "vel_norm": step_out["vel_norm"],
        }
        return next_carry, trace_out

    def scan_step(carry, _):
        return jax.lax.cond(
            carry[-1],
            lambda c: (c, _inactive_step_out(c[1], 2)),
            active_step,
            carry,
        )

    carry, trace = jax.lax.scan(scan_step, carry0, xs=None, length=horizon)
    (
        ep_agent,
        final_state,
        reward_sum,
        cost_sum,
        steps_taken,
        goal_streak,
        total_goal_steps,
        max_goal_streak,
        goal_buf,
        success,
        success_step,
        collision,
        done,
    ) = carry
    del goal_buf, done
    final_pos_error = jnp.linalg.norm(final_state[jnp.asarray([0, 2])] - goal)
    final_vel_norm = jnp.linalg.norm(final_state[jnp.asarray([1, 3])])
    final_state_norm_to_ref = jnp.linalg.norm(final_state - x_ref)
    last_action_idx = jnp.clip(steps_taken - 1, 0, horizon - 1)
    last_action_raw = (trace["action"][last_action_idx] + 1.0) / 2.0
    terminal_action_error_norm = jnp.linalg.norm(last_action_raw - jnp.asarray([0.5, 0.5], dtype=jnp.float32))
    terminal_success = final_state_norm_to_ref < jnp.asarray(success_radius, dtype=jnp.float32)
    dwell_success = max_goal_streak >= jnp.asarray(goal_dwell_steps, dtype=jnp.int32)
    if success_mode == "full_state_terminal":
        final_success = terminal_success
        final_success_step = jnp.where(terminal_success & (~collision), steps_taken, jnp.asarray(-1, dtype=jnp.int32))
    else:
        final_success = success
        final_success_step = success_step
    return {
        "agent": ep_agent,
        "init_state": state0,
        "states": jnp.concatenate([state0[None, :], trace["next_state"]], axis=0),
        "actions": trace["action"],
        "rewards": trace["reward"],
        "h": trace["h"],
        "h_phys": trace["h_phys"],
        "h_nominal": trace["h_nominal"],
        "binary_costs": trace["binary_cost"],
        "collisions": trace["collision"],
        "in_goals": trace["in_goal"],
        "pos_errors": trace["pos_error"],
        "vel_norms": trace["vel_norm"],
        "reward_sum": reward_sum,
        "cost_sum": cost_sum,
        "steps_taken": steps_taken,
        "goal_streak": goal_streak,
        "total_goal_steps": total_goal_steps,
        "max_goal_streak": max_goal_streak,
        "success": jnp.asarray(final_success & (~collision), dtype=jnp.float32),
        "dwell_success": jnp.asarray(dwell_success & (~collision), dtype=jnp.float32),
        "success_step": final_success_step,
        "collision": jnp.asarray(collision, dtype=jnp.float32),
        "terminal_pos_error": final_pos_error,
        "terminal_vel_norm": final_vel_norm,
        "terminal_state_norm_to_ref": final_state_norm_to_ref,
        "terminal_action_error_norm": terminal_action_error_norm,
    }

@lru_cache(maxsize=None)
def build_rollout_fn(
    layout_name: str,
    q_pos: float,
    q_x: float,
    q_z: float,
    q_vel: float,
    q_angle: float,
    action_penalty: float,
    reward_norm: str,
    goal_pos_radius: float,
    goal_vel_radius: float,
    goal_angle_radius: float,
    goal_dwell_steps: int,
    success_mode: str,
    success_radius: float,
    terminate_on_success: bool,
    sdf_mode: str,
    sdf_unsafe_const: float,
    sdf_unsafe_slope: float,
    sdf_global_scale: float,
    gate_lambda: float,
    gate_x_max: float,
    gate_z_center: float,
    gate_z_width: float,
    dwell_bonus: float,
    dt: float,
    obs_feature_mode: str,
    obs_h_scale: float,
    obs_hdot_scale: float,
):
    transition_core = _build_transition_core(
        layout_name,
        q_pos,
        q_x,
        q_z,
        q_vel,
        q_angle,
        action_penalty,
        reward_norm,
        goal_pos_radius,
        goal_vel_radius,
        goal_angle_radius,
        terminate_on_success,
        sdf_mode,
        sdf_unsafe_const,
        sdf_unsafe_slope,
        sdf_global_scale,
        gate_lambda,
        gate_x_max,
        gate_z_center,
        gate_z_width,
        dwell_bonus,
        dt,
    )
    goal = _layout_arrays(layout_name)["goal"]
    normalize_obs_fn = _build_obs_normalizer(
        layout_name,
        obs_feature_mode,
        obs_h_scale,
        obs_hdot_scale,
        dt,
        goal_pos_radius,
        goal_vel_radius,
        goal_angle_radius,
    )

    @partial(jax.jit, static_argnames=("horizon",))
    def rollout(agent, init_state, obs_mean, obs_var, horizon: int):
        return _rollout_episode_scan_core(
            agent,
            init_state,
            obs_mean,
            obs_var,
            goal,
            horizon,
            int(goal_dwell_steps),
            str(success_mode),
            float(success_radius),
            bool(terminate_on_success),
            transition_core,
            normalize_obs_fn,
        )

    return rollout


def build_rollout_fn_from_config(config: Dict[str, object]):
    q_pos = float(config.get("q_pos", 10.0))
    return build_rollout_fn(
        str(config.get("layout_name", "corridor_v2")),
        q_pos,
        float(config.get("q_x", q_pos)),
        float(config.get("q_z", q_pos)),
        float(config.get("q_vel", 1.0)),
        float(config.get("q_angle", 0.2)),
        float(config.get("action_penalty", 1e-3)),
        str(config.get("reward_norm", "l2")),
        float(config.get("goal_pos_radius", 0.30)),
        float(config.get("goal_vel_radius", 0.45)),
        float(config.get("goal_angle_radius", 0.25)),
        int(config.get("goal_dwell_steps", 45)),
        str(config.get("success_mode", "full_state_terminal")),
        float(config.get("success_radius", 0.50)),
        bool(config.get("terminate_on_success", False)),
        str(config.get("sdf_mode", "raw")),
        float(config.get("sdf_unsafe_const", 0.0)),
        float(config.get("sdf_unsafe_slope", 0.0)),
        float(config.get("sdf_global_scale", 0.0)),
        float(config.get("gate_lambda", 0.0)),
        float(config.get("gate_x_max", 0.90)),
        float(config.get("gate_z_center", 0.30)),
        float(config.get("gate_z_width", 1.20)),
        float(config.get("dwell_bonus", 0.0)),
        float(config.get("dt", 1.0 / 60.0)),
        str(config.get("obs_feature_mode", "state")),
        float(config.get("obs_h_scale", 1.0)),
        float(config.get("obs_hdot_scale", 1.0)),
    )


class Quad2DStabEnvP2(Quad2DStabEnv):
    """Quad2D-stab env with a jitted transition hot path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_backend = ENV_BACKEND_P2
        config = {
            "layout_name": self.layout_name,
            "q_pos": self.q_pos,
            "q_x": self.q_x,
            "q_z": self.q_z,
            "q_vel": self.q_vel,
            "q_angle": self.q_angle,
            "action_penalty": self.action_penalty,
            "reward_norm": self.reward_norm,
            "goal_pos_radius": self.goal_pos_radius,
            "goal_vel_radius": self.goal_vel_radius,
            "goal_angle_radius": self.goal_angle_radius,
            "success_mode": self.success_mode,
            "success_radius": self.success_radius,
            "terminate_on_success": self.terminate_on_success,
            "sdf_mode": self.sdf_mode,
            "sdf_unsafe_const": self.sdf_unsafe_const,
            "sdf_unsafe_slope": self.sdf_unsafe_slope,
            "sdf_global_scale": self.sdf_global_scale,
            "gate_lambda": self.gate_lambda,
            "gate_x_max": self.gate_x_max,
            "gate_z_center": self.gate_z_center,
            "gate_z_width": self.gate_z_width,
            "dwell_bonus": self.dwell_bonus,
            "dt": self.dt,
        }
        self._transition_core_p2 = _transition_core_from_config(config)

    def step(self, action: np.ndarray):
        out = jax.device_get(self._transition_core_p2(np.asarray(self.state), np.asarray(action, dtype=np.float32)))
        self.state = np.asarray(out["next_state"], dtype=np.float32)

        h_phys = float(out["h_phys"])
        h_val = float(out["h"])
        h_nominal = float(out["h_nominal"])
        binary_cost = float(out["binary_cost"])
        reward = float(out["reward"])
        in_goal = bool(out["in_goal"])

        if in_goal:
            self._episode_goal_steps += 1
            self._episode_total_goal_steps += 1
            self._episode_max_goal_streak = max(self._episode_max_goal_streak, self._episode_goal_steps)
        else:
            self._episode_goal_steps = 0

        success_now = False
        if (
            self.success_mode == "dwell_box"
            and (not self._episode_success)
            and self._episode_goal_steps >= self.goal_dwell_steps
        ):
            self._episode_success = True
            self._episode_success_step = self._t + 1
            success_now = True

        collision = bool(out["collision"])
        terminated = collision
        self._t += 1
        truncated = self._t >= self.max_episode_steps
        if self.success_mode == "dwell_box" and self.terminate_on_success and self._episode_success:
            terminated = True
        done = terminated or truncated

        self._episode_reward += reward
        self._episode_cost += binary_cost
        self._episode_length += 1

        pos_err = float(out["pos_error"])
        vel_norm = float(out["vel_norm"])
        terminal_state_norm = self._terminal_state_norm_to_ref(self.state)
        terminal_action_error_norm = float(np.linalg.norm(np.asarray(out["action_raw"], dtype=np.float32) - self.a_ref))
        self._episode_last_action_error_norm = terminal_action_error_norm
        dwell_success = self._episode_max_goal_streak >= self.goal_dwell_steps and h_nominal <= 0.0
        info = {
            "h": h_val,
            "h_phys": h_phys,
            "h_nominal": h_nominal,
            "binary_cost": binary_cost,
            "x": float(self.state[self.X]),
            "z": float(self.state[self.Z]),
            "pos_error": pos_err,
            "vel_norm": vel_norm,
            "in_goal": bool(in_goal),
            "success": bool(self._full_state_success(self.state) if self.success_mode == "full_state_terminal" else (self._episode_success and h_nominal <= 0.0)),
            "dwell_success": bool(dwell_success),
            "terminal_state_norm_to_ref": float(terminal_state_norm),
            "terminal_state_error_norm": float(terminal_state_norm),
            "terminal_action_error_norm": terminal_action_error_norm,
            "success_now": bool(success_now),
            "collision": bool(collision),
            "success_terminal": bool(self.success_mode == "dwell_box" and self.terminate_on_success and self._episode_success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_reward": self._episode_reward,
            "episode_cost": self._episode_cost,
            "episode_length": self._episode_length,
            "episode_goal_steps": self._episode_goal_steps,
            "episode_total_goal_steps": self._episode_total_goal_steps,
            "episode_max_goal_streak": self._episode_max_goal_streak,
            "success_step": -1 if self._episode_success_step is None else self._episode_success_step,
        }
        return self._get_obs(), reward, h_val, binary_cost, done, info

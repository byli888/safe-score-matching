#!/usr/bin/env python3
"""Evaluation helpers for Quad2D obstacle stabilize-avoid checkpoints."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import gym
import gym.spaces
import jax
import jax.numpy as jnp
import numpy as np

from ssm.tasks.quad2d_stabilize.env import ACT_DIM, FIXED_PAPER_START, GLOBAL_RESET_INSET, STATE_DIM, Quad2DStabEnv


_P123_SUPPORTED_SDF_MODES = {"raw", "hard", "hard_slope", "hard_scale"}
MIXED50_BASE_COUNTS = {
    "paper": 15,
    "low": 15,
    "gap": 10,
    "global": 5,
    "near": 5,
}
MIXED50_TOTAL = sum(MIXED50_BASE_COUNTS.values())
LOWZ_HREJ_SUITES = {
    "lowz050_hrej": 0.50,
    "lowz055_hrej": 0.55,
}


@dataclass
class LoadedQuad2DStab:
    agent: object
    config: Dict[str, object]
    obs_mean: np.ndarray
    obs_var: np.ndarray
    obs_count: int
    ckpt_path: str


def parse_hidden_dims(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(int(x) for x in value.split(",") if x)
    return tuple(int(x) for x in value)


def _agent_cls_for_version(version):
    if version not in ("dzfill_qhstage", "dzfill_qhstage_v_2", "dzfill_qhstage_v2"):
        raise ValueError("This example restores sampled-Qh SSM checkpoints.")
    from ssm.agents.ssm_agent import SSMOnlineAgent
    return SSMOnlineAgent


def load_agent_from_checkpoint_stab(ckpt_path: str, eval_seed: int = 0) -> LoadedQuad2DStab:
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)

    config = dict(data.get("config", {}))
    if float(config.get("h_hardgap", 0.0)) != 0.0 or bool(config.get("use_action_margin", False)):
        raise ValueError("Checkpoint is outside the historical zero-gap/no-action-margin example.")
    version = str(data.get("agent", config.get("agent", "dzfill_qhstage_v_2")))
    SSMOnlineAgent = _agent_cls_for_version(version)

    obs_mean_raw = data.get("obs_mean", None)
    obs_dim = int(config.get("obs_dim", len(obs_mean_raw) if obs_mean_raw is not None else STATE_DIM))
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-np.ones(ACT_DIM), high=np.ones(ACT_DIM), dtype=np.float32)
    create_kwargs = dict(
        seed=int(config.get("seed", 0)),
        observation_space=obs_space,
        action_space=act_space,
        actor_hidden_dims=parse_hidden_dims(config.get("actor_hidden"), (512, 512, 512)),
        critic_hidden_dims=parse_hidden_dims(config.get("critic_hidden"), (512, 512)),
        M_q=float(config.get("M_q", 10.0)),
        alpha_r=float(config.get("alpha_r", 1.0)),
        beta_safety=float(config.get("beta_safety", 5.0)),
        delta=float(config.get("delta", 0.0)),
        hard_margin=float(config.get("hard_margin", 0.0)),
        discount=float(config.get("discount", 0.99)),
        discount_h=float(config.get("discount_h", 0.999)),
        tau=float(config.get("tau", 0.005)),
        T=int(config.get("T", 5)),
        beta_schedule=str(config.get("beta_schedule", "vp")),
        cost_critic_hyperparam=float(config.get("cost_critic_hyperparam", 0.9)),
        safety_estimator="sampled_qh",
        pure_qsm=bool(config.get("pure_qsm", False)),
        ddpm_temperature=float(config.get("ddpm_temperature", 0.2)),
        fuse_clip_min=float(config.get("fuse_clip_min", -10000.0)),
        fuse_clip_max=float(config.get("fuse_clip_max", 100.0)),
        huber_delta=float(config.get("huber_delta", 10.0)),
    )
    if version == "dzfill_adaptive":
        create_kwargs.update(
            beta_max=float(config.get("beta_max", 50.0)),
            beta_init=float(config.get("beta_init", 5.0)),
            beta_reg=float(config.get("beta_reg", 1.0)),
            beta_lr=float(config.get("beta_lr", 3e-4)),
            adaptive_beta=True,
        )
    elif version == "unified":
        create_kwargs.update(
            beta_max=float(config.get("beta_max", 50.0)),
            beta_init=float(config.get("beta_init", 5.0)),
            beta_reg=float(config.get("beta_reg", 1.0)),
            beta_lr=float(config.get("beta_lr", 3e-4)),
            adaptive_beta=bool(config.get("adaptive_beta", False)),
        )
    elif version in ("dzfill_qhstage", "dzfill_qhstage_v_2", "dzfill_qhstage_v2"):
        create_kwargs.update(
            stage_mode=str(config.get("qh_stage_mode", "stage_b")),
            qh_candidate_source=str(config.get("qh_candidate_source", "policy_plus_gaussian")),
            qh_min_k_samples=int(config.get("qh_min_k_samples", 8)),
            qh_min_k_sigma=float(config.get("qh_min_k_sigma", 0.3)),
        )

    agent = SSMOnlineAgent.create(**create_kwargs)
    replacements = {
        name: getattr(agent, name).replace(params=value)
        for name, value in data["params"].items()
        if hasattr(agent, name) and hasattr(getattr(agent, name), "replace")
    }
    agent = agent.replace(**replacements, rng=jax.random.PRNGKey(eval_seed))

    return LoadedQuad2DStab(
        agent=agent,
        config=config,
        obs_mean=np.asarray(data.get("obs_mean", np.zeros(obs_dim)), dtype=np.float32),
        obs_var=np.asarray(data.get("obs_var", np.ones(obs_dim)), dtype=np.float32),
        obs_count=int(data.get("obs_count", 0)),
        ckpt_path=ckpt_path,
    )


def make_env_from_config(config: Dict[str, object], seed: int, reset_mode: Optional[str] = None) -> Quad2DStabEnv:
    q_pos = float(config.get("q_pos", 10.0))
    env = Quad2DStabEnv(
        seed=seed,
        layout_name=str(config.get("layout_name", "corridor_v2")),
        max_episode_steps=int(config.get("max_episode_steps", 480)),
        q_pos=q_pos,
        q_x=float(config.get("q_x", q_pos)),
        q_z=float(config.get("q_z", q_pos)),
        q_vel=float(config.get("q_vel", 1.0)),
        q_angle=float(config.get("q_angle", 0.2)),
        action_penalty=float(config.get("action_penalty", 1e-3)),
        reward_norm=str(config.get("reward_norm", "l2")),
        goal_pos_radius=float(config.get("goal_pos_radius", 0.30)),
        goal_vel_radius=float(config.get("goal_vel_radius", 0.45)),
        goal_angle_radius=float(config.get("goal_angle_radius", 0.25)),
        goal_dwell_steps=int(config.get("goal_dwell_steps", 45)),
        success_mode=str(config.get("success_mode", "full_state_terminal")),
        success_radius=float(config.get("success_radius", 0.50)),
        terminate_on_success=bool(config.get("terminate_on_success", False)),
        reset_mode=reset_mode or str(config.get("reset_mode", "start_band")),
        reset_mix_start_frac=float(config.get("reset_mix_start_frac", 0.70)),
        reset_mix_global_frac=float(config.get("reset_mix_global_frac", 0.20)),
        reset_mix_goal_frac=float(config.get("reset_mix_goal_frac", 0.10)),
        reset_route_global_frac=float(config.get("reset_route_global_frac", 0.35)),
        reset_route_near_frac=float(config.get("reset_route_near_frac", 0.30)),
        reset_route_gap_frac=float(config.get("reset_route_gap_frac", 0.35)),
        reset_under_global_frac=float(config.get("reset_under_global_frac", 0.35)),
        reset_under_frac=float(config.get("reset_under_frac", 0.65)),
        reset_low_frac=float(config.get("reset_low_frac", 0.80)),
        reset_low_mid_frac=float(config.get("reset_low_mid_frac", 0.15)),
        reset_low_near_frac=float(config.get("reset_low_near_frac", 0.05)),
        reset_low_paper_jitter_frac=float(config.get("reset_low_paper_jitter_frac", 0.30)),
        reset_low_paper_pinpoint_vel_frac=float(config.get("reset_low_paper_pinpoint_vel_frac", 0.0)),
        reset_low_under_bar_frac=float(config.get("reset_low_under_bar_frac", 0.20)),
        reset_low_under_blocks_frac=float(config.get("reset_low_under_blocks_frac", 0.20)),
        reset_low_inner_gaps_frac=float(config.get("reset_low_inner_gaps_frac", 0.20)),
        reset_low_global_frac=float(config.get("reset_low_global_frac", 0.10)),
        reset_simple_bar_frac=float(config.get("reset_simple_bar_frac", 0.30)),
        reset_simple_left_block_frac=float(config.get("reset_simple_left_block_frac", 0.10)),
        reset_simple_right_block_frac=float(config.get("reset_simple_right_block_frac", 0.10)),
        reset_simple_gap_frac=float(config.get("reset_simple_gap_frac", 0.10)),
        reset_simple_low_uniform_frac=float(config.get("reset_simple_low_uniform_frac", 0.25)),
        reset_simple_near_frac=float(config.get("reset_simple_near_frac", 0.15)),
        reset_simple_side_frac=float(config.get("reset_simple_side_frac", 0.0)),
        reset_simple_fork_frac=float(config.get("reset_simple_fork_frac", 0.0)),
        reset_safety_bar_frac=float(config.get("reset_safety_bar_frac", 0.20)),
        reset_safety_left_block_frac=float(config.get("reset_safety_left_block_frac", 0.15)),
        reset_safety_right_block_frac=float(config.get("reset_safety_right_block_frac", 0.15)),
        reset_safety_gap_frac=float(config.get("reset_safety_gap_frac", 0.15)),
        reset_safety_low_uniform_frac=float(config.get("reset_safety_low_uniform_frac", 0.25)),
        reset_safety_mid_uniform_frac=float(config.get("reset_safety_mid_uniform_frac", 0.05)),
        reset_safety_near_frac=float(config.get("reset_safety_near_frac", 0.05)),
        simple_under_vx_abs=float(config.get("simple_under_vx_abs", 0.50)),
        simple_under_vz_low=float(config.get("simple_under_vz_low", 0.00)),
        simple_under_vz_high=float(config.get("simple_under_vz_high", 0.70)),
        simple_bar_vz_low=float(config.get("simple_bar_vz_low", config.get("simple_under_vz_low", 0.00))),
        simple_bar_vz_high=float(config.get("simple_bar_vz_high", config.get("simple_under_vz_high", 0.70))),
        simple_block_vz_low=float(config.get("simple_block_vz_low", config.get("simple_under_vz_low", 0.00))),
        simple_block_vz_high=float(config.get("simple_block_vz_high", config.get("simple_under_vz_high", 0.70))),
        simple_under_xdot_zero_frac=float(config.get("simple_under_xdot_zero_frac", 0.0)),
        simple_bar_xdot_zero_frac=float(
            config.get("simple_bar_xdot_zero_frac", config.get("simple_under_xdot_zero_frac", 0.0))
        ),
        simple_block_xdot_zero_frac=float(
            config.get("simple_block_xdot_zero_frac", config.get("simple_under_xdot_zero_frac", 0.0))
        ),
        simple_gap_vx_abs=float(config.get("simple_gap_vx_abs", 0.40)),
        simple_gap_vz_low=float(config.get("simple_gap_vz_low", -0.20)),
        simple_gap_vz_high=float(config.get("simple_gap_vz_high", 0.50)),
        simple_low_uniform_vx_abs=float(config.get("simple_low_uniform_vx_abs", 0.35)),
        simple_low_uniform_vz_abs=float(config.get("simple_low_uniform_vz_abs", 0.35)),
        simple_fork_x_low=float(config.get("simple_fork_x_low", 0.05)),
        simple_fork_x_high=float(config.get("simple_fork_x_high", 0.35)),
        simple_fork_z_low=float(config.get("simple_fork_z_low", -1.00)),
        simple_fork_z_high=float(config.get("simple_fork_z_high", -0.30)),
        simple_fork_vx_low=float(config.get("simple_fork_vx_low", 0.10)),
        simple_fork_vx_high=float(config.get("simple_fork_vx_high", 0.55)),
        simple_fork_vz_low=float(config.get("simple_fork_vz_low", -0.05)),
        simple_fork_vz_high=float(config.get("simple_fork_vz_high", 0.45)),
        simple_fork_theta_abs=float(config.get("simple_fork_theta_abs", 0.12)),
        simple_fork_omega_abs=float(config.get("simple_fork_omega_abs", 0.15)),
        low_paper_jitter_vx_abs=float(config.get("low_paper_jitter_vx_abs", 0.25)),
        low_paper_jitter_vz_abs=float(config.get("low_paper_jitter_vz_abs", 0.25)),
        low_paper_pinpoint_vx_abs=float(config.get("low_paper_pinpoint_vx_abs", 1.00)),
        low_paper_pinpoint_vz_abs=float(config.get("low_paper_pinpoint_vz_abs", 0.30)),
        low_paper_pinpoint_theta_abs=float(config.get("low_paper_pinpoint_theta_abs", 0.10)),
        low_paper_pinpoint_omega_abs=float(config.get("low_paper_pinpoint_omega_abs", 0.05)),
        low_paper_pinpoint_x_abs=float(config.get("low_paper_pinpoint_x_abs", 0.05)),
        low_paper_pinpoint_z_center=float(config.get("low_paper_pinpoint_z_center", -1.08)),
        low_paper_pinpoint_z_halfwidth=float(config.get("low_paper_pinpoint_z_halfwidth", 0.02)),
        low_under_bar_vx_abs=float(config.get("low_under_bar_vx_abs", 0.25)),
        low_under_bar_vz_abs=float(config.get("low_under_bar_vz_abs", 0.25)),
        low_under_blocks_vx_abs=float(config.get("low_under_blocks_vx_abs", 0.25)),
        low_under_blocks_vz_abs=float(config.get("low_under_blocks_vz_abs", 0.25)),
        low_inner_gaps_vx_abs=float(config.get("low_inner_gaps_vx_abs", 0.25)),
        low_inner_gaps_vz_abs=float(config.get("low_inner_gaps_vz_abs", 0.25)),
        init_h_threshold=float(config.get("init_h_threshold", 0.0)),
        sdf_mode=str(config.get("sdf_mode", "raw")),
        sdf_unsafe_const=float(config.get("sdf_unsafe_const", 0.0)),
        sdf_unsafe_slope=float(config.get("sdf_unsafe_slope", 0.0)),
        sdf_global_scale=float(config.get("sdf_global_scale", 0.0)),
        gate_lambda=float(config.get("gate_lambda", 0.0)),
        gate_x_max=float(config.get("gate_x_max", 0.90)),
        gate_z_center=float(config.get("gate_z_center", 0.30)),
        gate_z_width=float(config.get("gate_z_width", 1.20)),
        obs_feature_mode=str(config.get("obs_feature_mode", "state")),
        obs_h_scale=float(config.get("obs_h_scale", 1.0)),
        obs_hdot_scale=float(config.get("obs_hdot_scale", 1.0)),
        dwell_bonus=float(config.get("dwell_bonus", 0.0)),
    )
    env.set_eval_mode()
    return env


def sync_obs_stats(env: Quad2DStabEnv, loaded: LoadedQuad2DStab) -> None:
    env.set_obs_stats(loaded.obs_mean, loaded.obs_var, loaded.obs_count)
    env.set_eval_mode()


def fixed_start_options(name: str = "paper_center") -> Dict[str, float]:
    if name == "paper_center":
        return dict(FIXED_PAPER_START)
    if name == "paper_left":
        return {**FIXED_PAPER_START, "init_x": -0.12}
    if name == "paper_right":
        return {**FIXED_PAPER_START, "init_x": 0.12}
    if name == "near_goal":
        return {
            "init_x": 0.0,
            "init_vx": 0.0,
            "init_z": 2.18,
            "init_vz": 0.0,
            "init_theta": 0.0,
            "init_omega": 0.0,
        }
    raise ValueError(f"Unknown fixed start {name!r}")


def mixed50_counts(total_episodes: int) -> Dict[str, int]:
    total = max(int(total_episodes), 0)
    if total == 0:
        return {k: 0 for k in MIXED50_BASE_COUNTS}
    raw = {k: total * (v / MIXED50_TOTAL) for k, v in MIXED50_BASE_COUNTS.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(MIXED50_BASE_COUNTS, key=lambda k: (raw[k] - counts[k], MIXED50_BASE_COUNTS[k]), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def classify_route(states: np.ndarray) -> str:
    """Classify route from an x-z trajectory.

    Side is based on the first crossing above the horizontal-bar region.
    Inner/outer is based on how far sideways the trajectory went before the
    goal-front bar.
    """
    if states.size == 0:
        return "none"
    xs = states[:, 0]
    zs = states[:, 2]
    crossing_idx = np.where(zs >= 0.62)[0]
    if len(crossing_idx) == 0:
        return "none"
    x_cross = float(xs[crossing_idx[0]])
    if abs(x_cross) < 0.08:
        side = "center"
    else:
        side = "left" if x_cross < 0 else "right"
    pre_goal_bar = states[zs < 1.12]
    max_abs_x = float(np.max(np.abs(pre_goal_bar[:, 0]))) if len(pre_goal_bar) else abs(x_cross)
    if max_abs_x > 2.06:
        route = "outer"
    elif max_abs_x > 1.00:
        route = "inner"
    else:
        route = "center"
    return f"{side}_{route}"


def side_entropy_norm(left_frac: float, right_frac: float) -> float:
    total = float(left_frac + right_frac)
    if total <= 0.0:
        return 0.0
    p = float(left_frac / total)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0))


def run_episode(
    loaded: LoadedQuad2DStab,
    env: Quad2DStabEnv,
    start_options: Optional[Dict[str, float]],
    policy_seed: int,
) -> Dict[str, object]:
    obs = env.reset(options=start_options) if start_options is not None else env.reset()
    ep_agent = loaded.agent.replace(rng=jax.random.fold_in(loaded.agent.rng, policy_seed))
    states = [env.state.copy()]
    actions = []
    infos = []
    done = False
    while not done:
        action, ep_agent = ep_agent.eval_actions(obs)
        action = np.asarray(action, dtype=np.float32)
        obs, reward, h_val, binary_cost, done, info = env.step(action)
        states.append(env.state.copy())
        actions.append(action)
        infos.append(dict(info, reward=reward, h_val=h_val, binary_cost=binary_cost))
    states_arr = np.asarray(states, dtype=np.float32)
    ep_info = env.episode_info
    last_info = infos[-1] if infos else {}
    return {
        "states": states_arr,
        "actions": np.asarray(actions, dtype=np.float32),
        "infos": infos,
        "episode_info": ep_info,
        "last_info": last_info,
        "route": classify_route(states_arr),
    }


def run_rollouts_stab(
    loaded: LoadedQuad2DStab,
    episodes: int,
    init_seed: int,
    policy_seed: int,
    start: str = "paper_center",
    random_starts: bool = False,
    reset_mode_override: Optional[str] = None,
    rollout_backend: str = "python",
) -> List[Dict[str, object]]:
    backend = resolve_rollout_backend_stab(loaded.config, rollout_backend)
    if backend == "p123":
        return run_rollouts_for_agent_stab(
            agent=loaded.agent,
            config=loaded.config,
            obs_mean=loaded.obs_mean,
            obs_var=loaded.obs_var,
            episodes=episodes,
            init_seed=init_seed,
            policy_seed=policy_seed,
            start=start,
            random_starts=random_starts,
            reset_mode_override=reset_mode_override,
            rollout_backend="p123",
        )

    reset = reset_mode_override or (str(loaded.config.get("eval_reset_mode", "start_band")) if random_starts else "start_band")
    env = make_env_from_config(loaded.config, seed=init_seed, reset_mode=reset)
    sync_obs_stats(env, loaded)
    start_options = None if random_starts else fixed_start_options(start)
    rollouts = []
    for ep in range(episodes):
        rollouts.append(run_episode(loaded, env, start_options, policy_seed + ep))
    return rollouts


def resolve_rollout_backend_stab(config: Dict[str, object], rollout_backend: str = "auto") -> str:
    backend = str(rollout_backend or "auto")
    if backend not in ("auto", "python", "p123"):
        raise ValueError(f"Unsupported rollout_backend={rollout_backend!r}")
    if backend == "python":
        return "python"
    sdf_mode = str(config.get("sdf_mode", "raw"))
    if sdf_mode not in _P123_SUPPORTED_SDF_MODES:
        if backend == "p123":
            raise ValueError(f"Quad2D-stab p123 does not support sdf_mode={sdf_mode!r}")
        return "python"
    return "p123"


def sample_initial_states_stab(
    config: Dict[str, object],
    episodes: int,
    init_seed: int,
    start: str = "paper_center",
    random_starts: bool = False,
    reset_mode_override: Optional[str] = None,
) -> np.ndarray:
    reset = reset_mode_override or (str(config.get("eval_reset_mode", "start_band")) if random_starts else "start_band")
    env = make_env_from_config(config, seed=init_seed, reset_mode=reset)
    states = []
    start_options = None if random_starts else fixed_start_options(start)
    for _ in range(episodes):
        if start_options is None:
            env.reset()
        else:
            env.reset(options=start_options)
        states.append(env.state.copy())
    return np.asarray(states, dtype=np.float32)


def _reset_options_from_state(state: np.ndarray) -> Dict[str, float]:
    state = np.asarray(state, dtype=np.float32)
    return {
        "init_x": float(state[0]),
        "init_vx": float(state[1]),
        "init_z": float(state[2]),
        "init_vz": float(state[3]),
        "init_theta": float(state[4]),
        "init_omega": float(state[5]),
    }


def sample_lowz_hrej_initial_states_stab(
    config: Dict[str, object],
    episodes: int,
    init_seed: int,
    z_high: float = 0.55,
) -> np.ndarray:
    """Sample broad low-z initial states with h_phys rejection.

    This is deliberately independent of the training reset mixture: it tests
    low-altitude safety over the whole corridor, rather than paper-center or a
    particular curriculum branch.
    """
    env = make_env_from_config(config, seed=init_seed, reset_mode="global_safe")
    rng = np.random.default_rng(init_seed)
    xmin, xmax = env.layout.xlim
    zmin, zmax = env.layout.zlim
    x_low = xmin + GLOBAL_RESET_INSET
    x_high = xmax - GLOBAL_RESET_INSET
    z_low = max(zmin + GLOBAL_RESET_INSET, -1.25)
    z_high_eff = min(float(z_high), zmax - GLOBAL_RESET_INSET)
    if z_high_eff <= z_low:
        raise ValueError(f"low-z rejection range is empty: z_low={z_low}, z_high={z_high_eff}")

    vx_abs = float(config.get("simple_low_uniform_vx_abs", 0.35))
    vz_abs = float(config.get("simple_low_uniform_vz_abs", 0.35))
    theta_abs = 0.14
    omega_abs = 0.10
    init_h_threshold = float(config.get("init_h_threshold", 0.0))

    states = []
    for ep in range(episodes):
        accepted = None
        for _attempt in range(20000):
            proposal = np.array(
                [
                    rng.uniform(x_low, x_high),
                    rng.uniform(-vx_abs, vx_abs),
                    rng.uniform(z_low, z_high_eff),
                    rng.uniform(-vz_abs, vz_abs),
                    rng.uniform(-theta_abs, theta_abs),
                    rng.uniform(-omega_abs, omega_abs),
                ],
                dtype=np.float32,
            )
            if env.h_phys(proposal) < init_h_threshold and not env._is_in_goal(proposal):
                accepted = proposal
                break
        if accepted is None:
            raise RuntimeError(f"Failed to sample lowz h-rejected state for episode {ep}.")
        states.append(accepted)
    return np.asarray(states, dtype=np.float32)


def _convert_p123_episode(ep: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    steps = int(np.asarray(ep["steps_taken"]).item())
    steps = max(0, steps)
    states = np.asarray(ep["states"], dtype=np.float32)[: steps + 1]
    actions = np.asarray(ep["actions"], dtype=np.float32)[:steps]
    h_vals = np.asarray(ep["h"], dtype=np.float32)
    h_phys = np.asarray(ep["h_phys"], dtype=np.float32)
    h_nominal = np.asarray(ep["h_nominal"], dtype=np.float32)
    binary_costs = np.asarray(ep["binary_costs"], dtype=np.float32)
    in_goals = np.asarray(ep["in_goals"], dtype=np.bool_)
    pos_errors = np.asarray(ep["pos_errors"], dtype=np.float32)
    vel_norms = np.asarray(ep["vel_norms"], dtype=np.float32)
    last_idx = max(min(steps - 1, len(h_vals) - 1), 0)

    success = float(np.asarray(ep["success"]).item())
    dwell_success = float(np.asarray(ep.get("dwell_success", 0.0)).item())
    collision = bool(float(np.asarray(ep["collision"]).item()) > 0.5)
    reward = float(np.asarray(ep["reward_sum"]).item())
    cost = float(np.asarray(ep["cost_sum"]).item())
    success_step = int(np.asarray(ep["success_step"]).item())
    terminal_state_norm_to_ref = float(np.asarray(ep.get("terminal_state_norm_to_ref", np.nan)).item())
    terminal_action_error_norm = float(np.asarray(ep.get("terminal_action_error_norm", np.nan)).item())
    if np.isnan(terminal_action_error_norm) and len(actions):
        last_action = np.clip(actions[-1], -1.0, 1.0)
        last_action_raw = (last_action + 1.0) / 2.0
        terminal_action_error_norm = float(np.linalg.norm(last_action_raw - np.asarray([0.5, 0.5], dtype=np.float32)))
    total_goal_steps = float(np.asarray(ep["total_goal_steps"]).item())
    max_goal_streak = float(np.asarray(ep["max_goal_streak"]).item())
    goal_streak = float(np.asarray(ep["goal_streak"]).item())
    length = float(steps)
    success_terminal = bool(
        success > 0.5
        and str(config.get("success_mode", "full_state_terminal")) == "dwell_box"
        and bool(config.get("terminate_on_success", False))
    )
    env_terminated = bool(collision or success_terminal)

    last_info = {
        "h": float(h_vals[last_idx]) if len(h_vals) else float("nan"),
        "h_phys": float(h_phys[last_idx]) if len(h_phys) else float("nan"),
        "h_nominal": float(h_nominal[last_idx]) if len(h_nominal) else float("nan"),
        "binary_cost": float(binary_costs[last_idx]) if len(binary_costs) else 0.0,
        "x": float(states[-1, 0]) if len(states) else float("nan"),
        "z": float(states[-1, 2]) if len(states) else float("nan"),
        "pos_error": float(pos_errors[last_idx]) if len(pos_errors) else float("nan"),
        "vel_norm": float(vel_norms[last_idx]) if len(vel_norms) else float("nan"),
        "terminal_state_norm_to_ref": terminal_state_norm_to_ref,
        "terminal_state_error_norm": terminal_state_norm_to_ref,
        "terminal_action_error_norm": terminal_action_error_norm,
        "in_goal": bool(in_goals[last_idx]) if len(in_goals) else False,
        "success": bool(success > 0.5),
        "dwell_success": bool(dwell_success > 0.5),
        "success_now": False,
        "collision": collision,
        "success_terminal": success_terminal,
        "terminated": env_terminated,
        "truncated": bool(steps >= int(config.get("max_episode_steps", 480)) and not env_terminated),
        "episode_reward": reward,
        "episode_cost": cost,
        "episode_length": length,
        "episode_goal_steps": goal_streak,
        "episode_total_goal_steps": total_goal_steps,
        "episode_max_goal_streak": max_goal_streak,
        "success_step": success_step,
    }
    episode_info = {
        "reward": reward,
        "cost": cost,
        "length": length,
        "success": success,
        "dwell_success": dwell_success,
        "goal_steps": goal_streak,
        "total_goal_steps": total_goal_steps,
        "max_goal_streak": max_goal_streak,
        "goal_dwell_frac": total_goal_steps / max(length, 1.0),
        "terminal_state_norm_to_ref": terminal_state_norm_to_ref,
        "terminal_state_error_norm": terminal_state_norm_to_ref,
        "terminal_action_error_norm": terminal_action_error_norm,
        "success_step": float(success_step),
    }
    return {
        "states": states,
        "actions": actions,
        "infos": [],
        "episode_info": episode_info,
        "last_info": last_info,
        "route": classify_route(states),
    }


def run_rollouts_for_agent_stab(
    agent,
    config: Dict[str, object],
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    episodes: int,
    init_seed: int,
    policy_seed: int,
    start: str = "paper_center",
    random_starts: bool = False,
    reset_mode_override: Optional[str] = None,
    rollout_backend: str = "auto",
    init_states: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    backend = resolve_rollout_backend_stab(config, rollout_backend)
    init_states_arr = None if init_states is None else np.asarray(init_states, dtype=np.float32)
    if init_states_arr is not None and len(init_states_arr) != episodes:
        raise ValueError(f"init_states length {len(init_states_arr)} does not match episodes={episodes}")
    if backend == "python":
        loaded = LoadedQuad2DStab(
            agent=agent,
            config=config,
            obs_mean=np.asarray(obs_mean, dtype=np.float32),
            obs_var=np.asarray(obs_var, dtype=np.float32),
            obs_count=0,
            ckpt_path="",
        )
        reset = reset_mode_override or (str(config.get("eval_reset_mode", "start_band")) if random_starts else "start_band")
        env = make_env_from_config(config, seed=init_seed, reset_mode=reset)
        env.set_obs_stats(obs_mean, obs_var, 0)
        env.set_eval_mode()
        if init_states_arr is not None:
            return [
                run_episode(loaded, env, _reset_options_from_state(init_states_arr[ep]), policy_seed + ep)
                for ep in range(episodes)
            ]
        start_options = None if random_starts else fixed_start_options(start)
        return [run_episode(loaded, env, start_options, policy_seed + ep) for ep in range(episodes)]

    from ssm.tasks.quad2d_stabilize.accelerated import build_rollout_fn_from_config

    if init_states_arr is None:
        init_states_arr = sample_initial_states_stab(
            config=config,
            episodes=episodes,
            init_seed=init_seed,
            start=start,
            random_starts=random_starts,
            reset_mode_override=reset_mode_override,
        )
    rollout_fn = build_rollout_fn_from_config(config)
    horizon = int(config.get("max_episode_steps", 480))
    base_rng = agent.rng
    obs_mean_j = jnp.asarray(obs_mean, dtype=jnp.float32)
    obs_var_j = jnp.asarray(obs_var, dtype=jnp.float32)
    rollouts = []
    for ep_idx, init_state in enumerate(init_states_arr):
        ep_agent = agent.replace(rng=jax.random.fold_in(base_rng, policy_seed + ep_idx))
        ep = jax.device_get(
            rollout_fn(
                ep_agent,
                jnp.asarray(init_state, dtype=jnp.float32),
                obs_mean_j,
                obs_var_j,
                horizon=horizon,
            )
        )
        rollouts.append(_convert_p123_episode(ep, config))
    return rollouts


def aggregate_rollouts_stab(rollouts: List[Dict[str, object]], prefix: str = "eval") -> Dict[str, float]:
    rewards = [r["episode_info"]["reward"] for r in rollouts]
    costs = [r["episode_info"]["cost"] for r in rollouts]
    lengths = [r["episode_info"]["length"] for r in rollouts]
    successes = [r["episode_info"]["success"] for r in rollouts]
    dwell_successes = [r["episode_info"].get("dwell_success", np.nan) for r in rollouts]
    crashes = [1.0 if r["last_info"].get("collision", False) else 0.0 for r in rollouts]
    pos_errors = [r["last_info"].get("pos_error", np.nan) for r in rollouts]
    vel_norms = [r["last_info"].get("vel_norm", np.nan) for r in rollouts]
    terminal_state_norms = [r["episode_info"].get("terminal_state_norm_to_ref", np.nan) for r in rollouts]
    terminal_action_error_norms = [r["episode_info"].get("terminal_action_error_norm", np.nan) for r in rollouts]
    dwell_fracs = [r["episode_info"].get("goal_dwell_frac", np.nan) for r in rollouts]
    routes = [str(r["route"]) for r in rollouts]
    successful_routes = [route for route, success in zip(routes, successes) if float(success) > 0.5]
    left = np.mean([route.startswith("left") for route in routes]) if routes else 0.0
    right = np.mean([route.startswith("right") for route in routes]) if routes else 0.0
    inner = np.mean([route.endswith("inner") for route in routes]) if routes else 0.0
    outer = np.mean([route.endswith("outer") for route in routes]) if routes else 0.0
    succ_left = np.mean([route.startswith("left") for route in successful_routes]) if successful_routes else 0.0
    succ_right = np.mean([route.startswith("right") for route in successful_routes]) if successful_routes else 0.0
    succ_inner = np.mean([route.endswith("inner") for route in successful_routes]) if successful_routes else 0.0
    succ_outer = np.mean([route.endswith("outer") for route in successful_routes]) if successful_routes else 0.0
    return {
        f"{prefix}/num_episodes": float(len(rollouts)),
        f"{prefix}/reward_mean": float(np.mean(rewards)),
        f"{prefix}/reward_std": float(np.std(rewards)),
        f"{prefix}/cost_mean": float(np.mean(costs)),
        f"{prefix}/length_mean": float(np.mean(lengths)),
        f"{prefix}/success_frac": float(np.mean(successes)),
        f"{prefix}/dwell_success_frac": float(np.nanmean(dwell_successes)),
        f"{prefix}/crash_frac": float(np.mean(crashes)),
        f"{prefix}/goal_dwell_frac": float(np.nanmean(dwell_fracs)),
        f"{prefix}/terminal_state_norm_to_ref": float(np.nanmean(terminal_state_norms)),
        f"{prefix}/terminal_state_error_norm": float(np.nanmean(terminal_state_norms)),
        f"{prefix}/terminal_action_error_norm": float(np.nanmean(terminal_action_error_norms)),
        f"{prefix}/terminal_pos_error": float(np.nanmean(pos_errors)),
        f"{prefix}/terminal_vel_norm": float(np.nanmean(vel_norms)),
        f"{prefix}/route_left_frac": float(left),
        f"{prefix}/route_right_frac": float(right),
        f"{prefix}/route_inner_frac": float(inner),
        f"{prefix}/route_outer_frac": float(outer),
        f"{prefix}/side_mode_coverage": float(2.0 * min(left, right)),
        f"{prefix}/side_route_split_entropy_norm": side_entropy_norm(left, right),
        f"{prefix}/success_route_left_frac": float(succ_left),
        f"{prefix}/success_route_right_frac": float(succ_right),
        f"{prefix}/success_route_inner_frac": float(succ_inner),
        f"{prefix}/success_route_outer_frac": float(succ_outer),
        f"{prefix}/side_mode_coverage_success": float(2.0 * min(succ_left, succ_right)),
        f"{prefix}/success_side_route_split_entropy_norm": side_entropy_norm(succ_left, succ_right),
    }


def _add_region_aliases(metrics: Dict[str, float], prefix: str, label: str, region_metrics: Dict[str, float]) -> None:
    region_prefix = f"{prefix}_{label}"
    for field in [
        "num_episodes",
        "success_frac",
        "dwell_success_frac",
        "crash_frac",
        "cost_mean",
        "reward_mean",
        "terminal_state_norm_to_ref",
        "terminal_state_error_norm",
        "terminal_action_error_norm",
        "terminal_pos_error",
        "terminal_vel_norm",
        "route_left_frac",
        "route_right_frac",
        "success_route_left_frac",
        "success_route_right_frac",
        "side_mode_coverage",
        "side_mode_coverage_success",
        "side_route_split_entropy_norm",
        "success_side_route_split_entropy_norm",
    ]:
        metrics[f"{prefix}/{label}_{field}"] = float(region_metrics.get(f"{region_prefix}/{field}", np.nan))


def _mixed50_composite(metrics: Dict[str, float], prefix: str) -> float:
    def g(key: str, default: float = 0.0) -> float:
        value = float(metrics.get(f"{prefix}/{key}", default))
        if np.isnan(value):
            return default
        return value

    return float(
        2.0 * g("paper_success_frac")
        + 1.5 * g("low_success_frac")
        + 1.0 * g("gap_success_frac")
        + 0.5 * g("global_success_frac")
        + 0.5 * g("near_success_frac")
        - 1.5 * g("paper_crash_frac")
        - 1.0 * g("low_crash_frac")
        - 0.5 * g("gap_crash_frac")
        - 0.3 * g("global_crash_frac")
    )


def _lowz_hrej_safety_score(metrics: Dict[str, float], prefix: str) -> float:
    def g(key: str, default: float = 0.0) -> float:
        value = float(metrics.get(f"{prefix}/{key}", default))
        if np.isnan(value):
            return default
        return value

    return float(
        1.0 * g("success_frac")
        + 0.5 * g("dwell_success_frac")
        - 1.5 * g("crash_frac")
        - 0.3 * g("cost_mean") / 400.0
        - 0.05 * g("terminal_state_norm_to_ref")
    )


def _add_lowz_hrej_aliases(metrics: Dict[str, float], prefix: str, suite: str, z_high: float) -> None:
    label = suite.replace("_hrej", "")
    metrics[f"{prefix}/lowz_hrej_z_high"] = float(z_high)
    metrics[f"{prefix}/safety_score"] = _lowz_hrej_safety_score(metrics, prefix)
    for field in [
        "num_episodes",
        "success_frac",
        "dwell_success_frac",
        "crash_frac",
        "cost_mean",
        "reward_mean",
        "terminal_state_norm_to_ref",
        "terminal_state_error_norm",
        "terminal_action_error_norm",
        "terminal_pos_error",
        "terminal_vel_norm",
        "goal_dwell_frac",
        "route_left_frac",
        "route_right_frac",
        "success_route_left_frac",
        "success_route_right_frac",
        "side_mode_coverage",
        "side_mode_coverage_success",
        "side_route_split_entropy_norm",
        "success_side_route_split_entropy_norm",
    ]:
        metrics[f"{prefix}/{label}_{field}"] = float(metrics.get(f"{prefix}/{field}", np.nan))


def run_lowz_hrej_rollouts_for_agent_stab(
    agent,
    config: Dict[str, object],
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    episodes: int,
    init_seed: int,
    policy_seed: int,
    suite: str = "lowz055_hrej",
    rollout_backend: str = "auto",
) -> List[Dict[str, object]]:
    if suite not in LOWZ_HREJ_SUITES:
        raise ValueError(f"Unsupported low-z h-rejection suite {suite!r}")
    init_states = sample_lowz_hrej_initial_states_stab(
        config=config,
        episodes=episodes,
        init_seed=init_seed,
        z_high=LOWZ_HREJ_SUITES[suite],
    )
    return run_rollouts_for_agent_stab(
        agent=agent,
        config=config,
        obs_mean=obs_mean,
        obs_var=obs_var,
        episodes=episodes,
        init_seed=init_seed,
        policy_seed=policy_seed,
        random_starts=True,
        rollout_backend=rollout_backend,
        init_states=init_states,
    )


def evaluate_agent_stab_suite(
    agent,
    config: Dict[str, object],
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    episodes: int,
    init_seed: int,
    policy_seed: int,
    suite: str = "mixed50",
    prefix: str = "eval",
    rollout_backend: str = "auto",
) -> Dict[str, float]:
    if suite in LOWZ_HREJ_SUITES:
        rollouts = run_lowz_hrej_rollouts_for_agent_stab(
            agent=agent,
            config=config,
            obs_mean=obs_mean,
            obs_var=obs_var,
            episodes=episodes,
            init_seed=init_seed,
            policy_seed=policy_seed,
            suite=suite,
            rollout_backend=rollout_backend,
        )
        metrics = aggregate_rollouts_stab(rollouts, prefix=prefix)
        _add_lowz_hrej_aliases(metrics, prefix, suite, LOWZ_HREJ_SUITES[suite])
        return metrics

    if suite != "mixed50":
        raise ValueError(f"Unsupported Quad2D-stab eval suite {suite!r}")

    counts = mixed50_counts(episodes)
    specs = [
        ("paper", counts["paper"], "paper_center", False, None),
        ("low", counts["low"], "paper_center", True, "low_mix"),
        ("gap", counts["gap"], "paper_center", True, "gap_mix"),
        ("global", counts["global"], "paper_center", True, "global_safe"),
        ("near", counts["near"], "near_goal", False, None),
    ]
    all_rollouts: List[Dict[str, object]] = []
    metrics: Dict[str, float] = {}
    seed_cursor = 0
    for label, count, start, random_starts, reset_override in specs:
        if count <= 0:
            continue
        rollouts = run_rollouts_for_agent_stab(
            agent=agent,
            config=config,
            obs_mean=obs_mean,
            obs_var=obs_var,
            episodes=count,
            init_seed=init_seed + 17 * seed_cursor,
            policy_seed=policy_seed + 1000 * seed_cursor,
            start=start,
            random_starts=random_starts,
            reset_mode_override=reset_override,
            rollout_backend=rollout_backend,
        )
        region_metrics = aggregate_rollouts_stab(rollouts, prefix=f"{prefix}_{label}")
        _add_region_aliases(metrics, prefix, label, region_metrics)
        all_rollouts.extend(rollouts)
        seed_cursor += 1

    metrics.update(aggregate_rollouts_stab(all_rollouts, prefix=prefix))
    metrics[f"{prefix}/composite_score"] = _mixed50_composite(metrics, prefix)
    metrics[f"{prefix}/suite_episodes"] = float(sum(counts.values()))
    return metrics


def evaluate_checkpoint_stab(
    ckpt_path: str,
    episodes: int,
    init_seed: int,
    policy_seed: int,
    start: str = "paper_center",
    random_starts: bool = False,
    reset_mode_override: Optional[str] = None,
    prefix: str = "eval_stab",
    rollout_backend: str = "auto",
):
    loaded = load_agent_from_checkpoint_stab(ckpt_path, eval_seed=policy_seed)
    rollouts = run_rollouts_stab(
        loaded,
        episodes=episodes,
        init_seed=init_seed,
        policy_seed=policy_seed,
        start=start,
        random_starts=random_starts,
        reset_mode_override=reset_mode_override,
        rollout_backend=rollout_backend,
    )
    metrics = aggregate_rollouts_stab(rollouts, prefix=prefix)
    return metrics, rollouts, loaded


def evaluate_checkpoint_stab_suite(
    ckpt_path: str,
    episodes: int,
    init_seed: int,
    policy_seed: int,
    suite: str = "mixed50",
    prefix: str = "eval_stab",
    rollout_backend: str = "auto",
):
    loaded = load_agent_from_checkpoint_stab(ckpt_path, eval_seed=policy_seed)
    if suite in LOWZ_HREJ_SUITES:
        rollouts = run_lowz_hrej_rollouts_for_agent_stab(
            agent=loaded.agent,
            config=loaded.config,
            obs_mean=loaded.obs_mean,
            obs_var=loaded.obs_var,
            episodes=episodes,
            init_seed=init_seed,
            policy_seed=policy_seed,
            suite=suite,
            rollout_backend=rollout_backend,
        )
        metrics = aggregate_rollouts_stab(rollouts, prefix=prefix)
        _add_lowz_hrej_aliases(metrics, prefix, suite, LOWZ_HREJ_SUITES[suite])
        return metrics, rollouts, loaded
    metrics = evaluate_agent_stab_suite(
        agent=loaded.agent,
        config=loaded.config,
        obs_mean=loaded.obs_mean,
        obs_var=loaded.obs_var,
        episodes=episodes,
        init_seed=init_seed,
        policy_seed=policy_seed,
        suite=suite,
        prefix=prefix,
        rollout_backend=rollout_backend,
    )
    return metrics, [], loaded


def evaluate_agent_stab(
    agent,
    config: Dict[str, object],
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    episodes: int,
    init_seed: int,
    policy_seed: int,
    start: str = "paper_center",
    random_starts: bool = False,
    reset_mode_override: Optional[str] = None,
    prefix: str = "eval",
    rollout_backend: str = "auto",
) -> Dict[str, float]:
    rollouts = run_rollouts_for_agent_stab(
        agent=agent,
        config=config,
        obs_mean=obs_mean,
        obs_var=obs_var,
        episodes=episodes,
        init_seed=init_seed,
        policy_seed=policy_seed,
        start=start,
        random_starts=random_starts,
        reset_mode_override=reset_mode_override,
        rollout_backend=rollout_backend,
    )
    return aggregate_rollouts_stab(rollouts, prefix=prefix)


def write_summary_json(path: str, payload: Dict[str, object]) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(convert(payload), f, indent=2, sort_keys=True)

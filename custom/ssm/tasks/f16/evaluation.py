"""F16 Stage-A/Stage-B checkpoint evaluation.

The scan includes initial-state safety and reports both final-window stability
and the longest consecutive goal-altitude run over the full episode.
"""
from __future__ import annotations

import pickle
from functools import lru_cache, partial
from typing import Dict
import jax
import jax.numpy as jnp
import numpy as np
from ssm.agents.ssm_agent import SSMOnlineAgent
import ssm.tasks.f16.env as env_v6
from ssm.tasks.f16.env import F16StabilizeEnvV6, TASK_SPEC_VERSION_V6, IDX_ALPHA, IDX_BETA, IDX_H, IDX_P, IDX_PE, IDX_THETA
from ssm.tasks.f16.accelerated import make_obs_core, make_transition_core, _task_goal_distance_jax, _task_safety_margin_jax

PAPER_EVAL_HORIZON_V5 = 512
PAPER_LAST_N_V5 = 50

_CAUSE_NONE = 0


_CAUSE_ALPHA = 1


_CAUSE_P = 2


_CAUSE_ALT = 3


_CAUSE_PE = 4


_CAUSE_BETA = 5


_CAUSE_THETA = 6


_CAUSE_OTHER = 7


_TERM_NONE = 0


_TERM_INVALID = 1


_TERM_HARD = 2


_TERM_INVALID_AND_HARD = 3


def _classify_crash_cause_code_jax(
    x_raw,
    terminate_h_min: float,
    terminate_h_max: float,
):
    return jnp.where(
        jnp.logical_or(x_raw[IDX_ALPHA] <= env_v6._TERMINATE_ALPHA_LO, x_raw[IDX_ALPHA] >= env_v6._TERMINATE_ALPHA_HI),
        jnp.int32(_CAUSE_ALPHA),
        jnp.where(
            jnp.abs(x_raw[IDX_P]) >= env_v6._TERMINATE_P,
            jnp.int32(_CAUSE_P),
            jnp.where(
                jnp.logical_or(
                    x_raw[IDX_H] <= terminate_h_min,
                    x_raw[IDX_H] >= terminate_h_max,
                ),
                jnp.int32(_CAUSE_ALT),
                jnp.where(
                    jnp.abs(x_raw[IDX_PE]) >= env_v6._TERMINATE_PE_LIMIT,
                    jnp.int32(_CAUSE_PE),
                    jnp.where(
                        jnp.abs(x_raw[IDX_BETA]) >= env_v6._TERMINATE_BETA,
                        jnp.int32(_CAUSE_BETA),
                        jnp.where(
                            jnp.abs(x_raw[IDX_THETA]) >= env_v6._TERMINATE_THETA,
                            jnp.int32(_CAUSE_THETA),
                            jnp.int32(_CAUSE_OTHER),
                        ),
                    ),
                ),
            ),
        ),
    )


def _classify_terminate_reason_code_jax(invalid_dynamics, hard_terminal):
    return jnp.where(
        jnp.logical_and(invalid_dynamics, hard_terminal),
        jnp.int32(_TERM_INVALID_AND_HARD),
        jnp.where(
            hard_terminal,
            jnp.int32(_TERM_HARD),
            jnp.where(invalid_dynamics, jnp.int32(_TERM_INVALID), jnp.int32(_TERM_NONE)),
        ),
    )


def _rollout_episode_scan_core(
    agent,
    init_state: jnp.ndarray,
    horizon: int,
    last_n: int,
    obs_core,
    transition_core,
    use_literal_gap: bool,
    h_theta_denom: float,
    safe_alpha_hi: float,
    safe_alpha_lo: float,
    safe_beta: float,
    safe_h_min: float,
    safe_h_max: float,
    safe_theta: float,
    safe_pe: float,
    safe_p: float,
    terminate_h_min: float,
    terminate_h_max: float,
    terminate_pe_limit: float,
    terminate_p: float,
):
    x0 = jnp.asarray(init_state, dtype=jnp.float32)
    peak_h0 = _task_safety_margin_jax(
        x0,
        use_literal_gap=use_literal_gap,
        h_theta_denom=h_theta_denom,
        safe_alpha_hi=safe_alpha_hi,
        safe_alpha_lo=safe_alpha_lo,
        safe_beta=safe_beta,
        safe_h_min=safe_h_min,
        safe_h_max=safe_h_max,
        safe_theta=safe_theta,
        safe_pe=safe_pe,
        safe_p=safe_p,
        terminate_h_min=terminate_h_min,
        terminate_h_max=terminate_h_max,
        terminate_pe_limit=terminate_pe_limit,
        terminate_p=terminate_p,
    )
    min_altitude0 = x0[IDX_H]
    max_abs_theta0 = jnp.abs(x0[IDX_THETA])
    max_abs_beta0 = jnp.abs(x0[IDX_BETA])
    traj_safe0 = peak_h0 <= 0.0
    goal_buf0 = jnp.zeros((last_n,), dtype=jnp.bool_) if last_n > 0 else jnp.zeros((1,), dtype=jnp.bool_)
    entered_band0 = _task_goal_distance_jax(x0) <= 0.0
    time_to_band0 = jnp.where(entered_band0, jnp.array(0.0, dtype=jnp.float32), jnp.array(jnp.nan, dtype=jnp.float32))

    carry0 = (
        agent,
        x0,
        jnp.array(0.0, dtype=jnp.float32),  # reward_sum
        jnp.array(0.0, dtype=jnp.float32),  # unsafe_step_count
        jnp.array(0.0, dtype=jnp.float32),  # task_cost_sum
        jnp.maximum(peak_h0, 0.0),          # h_pos_sum
        peak_h0,
        min_altitude0,
        max_abs_theta0,
        max_abs_beta0,
        traj_safe0,
        jnp.array(False),
        jnp.array(_CAUSE_NONE, dtype=jnp.int32),
        jnp.array(_TERM_NONE, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),  # steps_taken
        goal_buf0,
        entered_band0,
        time_to_band0,
        jnp.array(0, dtype=jnp.int32),  # current goal streak
        jnp.array(0, dtype=jnp.int32),  # longest goal streak
        jnp.array(False),  # done
    )

    def active_step(carry):
        (
            ep_agent,
            x,
            reward_sum,
            unsafe_step_count,
            task_cost_sum,
            h_pos_sum,
            peak_h,
            min_altitude,
            max_abs_theta,
            max_abs_beta,
            traj_safe,
            crashed,
            crash_cause_code,
            terminated_reason_code,
            steps_taken,
            goal_buf,
            entered_band,
            time_to_band,
            goal_streak,
            longest_goal_streak,
            _done,
        ) = carry

        obs = obs_core(x)
        action, ep_agent = ep_agent.eval_actions(obs)
        (
            x_next,
            _action_clipped,
            _control,
            _h_components,
            h_margin,
            task_cost,
            reward,
            goal_distance,
            in_goal,
            safe,
            _invalid_dynamics,
            _hard_terminal,
            terminated,
        ) = transition_core(x, action)

        invalid_dynamics = _invalid_dynamics
        hard_terminal = _hard_terminal

        if last_n > 0:
            goal_buf = goal_buf.at[steps_taken % last_n].set(in_goal)
        steps_taken = steps_taken + 1
        time_to_band = jnp.where(
            jnp.logical_and(jnp.isnan(time_to_band), in_goal),
            steps_taken.astype(jnp.float32),
            time_to_band,
        )
        entered_band = jnp.logical_or(entered_band, in_goal)
        goal_streak = jnp.where(in_goal, goal_streak + 1, 0)
        longest_goal_streak = jnp.maximum(longest_goal_streak, goal_streak)
        reward_sum = reward_sum + reward
        unsafe_step_count = unsafe_step_count + jnp.asarray(h_margin > 0.0, dtype=jnp.float32)
        task_cost_sum = task_cost_sum + task_cost
        h_pos_sum = h_pos_sum + jnp.maximum(h_margin, 0.0)
        peak_h = jnp.maximum(peak_h, h_margin)
        min_altitude = jnp.minimum(min_altitude, x_next[IDX_H])
        max_abs_theta = jnp.maximum(max_abs_theta, jnp.abs(x_next[IDX_THETA]))
        max_abs_beta = jnp.maximum(max_abs_beta, jnp.abs(x_next[IDX_BETA]))
        traj_safe = jnp.logical_and(traj_safe, safe)
        crashed = jnp.logical_or(crashed, terminated)
        crash_cause_code = jnp.where(
            jnp.logical_and(terminated, crash_cause_code == _CAUSE_NONE),
            _classify_crash_cause_code_jax(
                x_next,
                terminate_h_min=terminate_h_min,
                terminate_h_max=terminate_h_max,
            ),
            crash_cause_code,
        )
        terminated_reason_code = jnp.where(
            jnp.logical_and(terminated, terminated_reason_code == _TERM_NONE),
            _classify_terminate_reason_code_jax(invalid_dynamics, hard_terminal),
            terminated_reason_code,
        )

        return (
            ep_agent,
            x_next,
            reward_sum,
            unsafe_step_count,
            task_cost_sum,
            h_pos_sum,
            peak_h,
            min_altitude,
            max_abs_theta,
            max_abs_beta,
            traj_safe,
            crashed,
            crash_cause_code,
            terminated_reason_code,
            steps_taken,
            goal_buf,
            entered_band,
            time_to_band,
            goal_streak,
            longest_goal_streak,
            terminated,
        )

    def scan_step(carry, _):
        next_carry = jax.lax.cond(carry[-1], lambda c: c, active_step, carry)
        return next_carry, None

    carry, _ = jax.lax.scan(scan_step, carry0, xs=None, length=horizon)
    (
        ep_agent,
        x_final,
        reward_sum,
        unsafe_step_count,
        task_cost_sum,
        h_pos_sum,
        peak_h,
        min_altitude,
        max_abs_theta,
        max_abs_beta,
        traj_safe,
        crashed,
        crash_cause_code,
        terminated_reason_code,
        steps_taken,
        goal_buf,
        entered_band,
        time_to_band,
        goal_streak,
        longest_goal_streak,
        _done,
    ) = carry

    stabilized_last50 = (steps_taken >= last_n) & jnp.all(goal_buf) if last_n > 0 else jnp.array(False)
    strict_safe = jnp.asarray(jnp.logical_and(traj_safe, jnp.logical_not(crashed)), dtype=jnp.float32)
    strict_stabilized = jnp.asarray(
        jnp.logical_and(stabilized_last50, jnp.logical_not(crashed)),
        dtype=jnp.float32,
    )
    return {
        "agent": ep_agent,
        "reward_sum": reward_sum,
        "unsafe_step_count": unsafe_step_count,
        "task_cost_sum": task_cost_sum,
        "h_pos_sum": h_pos_sum,
        "strict_safe": strict_safe,
        "strict_stabilized_last50": strict_stabilized,
        "longest_goal_run": longest_goal_streak,
        "reached_goal_dwell": jnp.asarray(longest_goal_streak >= last_n, dtype=jnp.float32),
        "strict_stabilized_any_window": jnp.asarray(
            (longest_goal_streak >= last_n) & ~crashed, dtype=jnp.float32),
        "crashed": jnp.asarray(crashed, dtype=jnp.float32),
        "crash_cause_code": crash_cause_code,
        "terminated_reason_code": terminated_reason_code,
        "init_h": peak_h0,
        "peak_h": peak_h,
        "min_altitude": min_altitude,
        "max_abs_theta": max_abs_theta,
        "max_abs_beta": max_abs_beta,
        "final_goal_distance": _task_goal_distance_jax(x_final),
        "steps_taken": steps_taken.astype(jnp.float32),
        "one_step_death": jnp.asarray(jnp.logical_and(crashed, steps_taken <= 1), dtype=jnp.float32),
        "entered_band_before_fail": jnp.asarray(entered_band, dtype=jnp.float32),
        "time_to_band": time_to_band,
    }


@lru_cache(maxsize=None)
def _build_rollout_fn(
    safety_gap_mode: str,
    h_theta_denom: float,
    obs_task_feats_mode: str,
    l_form: str,
    l_q_center: float,
    l_q_out: float,
    l_kappa: float,
    l_stage_form: str,
    l_stage_reward: float,
    safe_alpha_hi: float,
    safe_alpha_lo: float,
    safe_beta: float,
    safe_h_min: float,
    safe_h_max: float,
    safe_theta: float,
    safe_pe: float,
    safe_p: float,
    terminate_h_min: float,
    terminate_h_max: float,
    terminate_pe_limit: float,
    terminate_p: float,
    alpha_counts_as_invalid_dynamics: bool,
    beta_counts_as_invalid_dynamics: bool,
):
    use_literal_gap = safety_gap_mode == "literal"
    obs_core = make_obs_core(
        use_literal_gap=use_literal_gap,
        h_theta_denom=h_theta_denom,
        safe_alpha_hi=safe_alpha_hi,
        safe_alpha_lo=safe_alpha_lo,
        safe_beta=safe_beta,
        safe_h_min=safe_h_min,
        safe_h_max=safe_h_max,
        safe_theta=safe_theta,
        safe_pe=safe_pe,
        safe_p=safe_p,
        terminate_h_min=terminate_h_min,
        terminate_h_max=terminate_h_max,
        terminate_pe_limit=terminate_pe_limit,
        terminate_p=terminate_p,
        obs_task_feats_mode=obs_task_feats_mode,
    )
    transition_core = make_transition_core(
        use_literal_gap=use_literal_gap,
        h_theta_denom=h_theta_denom,
        l_form=l_form,
        l_q_center=l_q_center,
        l_q_out=l_q_out,
        l_kappa=l_kappa,
        l_stage_form=l_stage_form,
        l_stage_reward=l_stage_reward,
        safe_alpha_hi=safe_alpha_hi,
        safe_alpha_lo=safe_alpha_lo,
        safe_beta=safe_beta,
        safe_h_min=safe_h_min,
        safe_h_max=safe_h_max,
        safe_theta=safe_theta,
        safe_pe=safe_pe,
        safe_p=safe_p,
        terminate_h_min=terminate_h_min,
        terminate_h_max=terminate_h_max,
        terminate_pe_limit=terminate_pe_limit,
        terminate_p=terminate_p,
        alpha_counts_as_invalid_dynamics=alpha_counts_as_invalid_dynamics,
        beta_counts_as_invalid_dynamics=beta_counts_as_invalid_dynamics,
    )

    @partial(jax.jit, static_argnames=("horizon", "last_n"))
    def rollout(agent, init_state, horizon: int = PAPER_EVAL_HORIZON_V5, last_n: int = PAPER_LAST_N_V5):
        return _rollout_episode_scan_core(
            agent,
            init_state,
            horizon,
            last_n,
            obs_core,
            transition_core,
            use_literal_gap,
            h_theta_denom,
            safe_alpha_hi,
            safe_alpha_lo,
            safe_beta,
            safe_h_min,
            safe_h_max,
            safe_theta,
            safe_pe,
            safe_p,
            terminate_h_min,
            terminate_h_max,
            terminate_pe_limit,
            terminate_p,
        )

    return rollout


def make_env_from_config_v6(config: Dict, seed: int, strict_invalid_dynamics: bool = True,
                            apply_training_safety_overrides: bool = True) -> F16StabilizeEnvV6:
    task_spec_version = config.get("task_spec_version")
    if task_spec_version != TASK_SPEC_VERSION_V6:
        raise ValueError(
            f"Unsupported task_spec_version={task_spec_version!r}; "
            f"this evaluator requires {TASK_SPEC_VERSION_V6!r}."
        )
    env = F16StabilizeEnvV6(
        seed=seed,
        max_episode_steps=int(config.get("max_episode_steps", 640)),
        goal_dwell_steps=int(config.get("goal_dwell_steps", 50)),
        terminate_on_crash=bool(config.get("terminate_on_crash", True)),
        init_curriculum=False,
        safety_gap_mode=str(config.get("safety_gap_mode", "nogap")),
        obs_task_feats_mode=str(config.get("obs_task_feats_mode", "per_axis")),
        h_form=str(config.get("h_form", "linear")),
        h_theta_denom=float(config.get("h_theta_denom")),
        l_form=str(config.get("l_form", "split_log")),
        l_q=float(config.get("l_q", 0.0)),
        l_q_center=float(config.get("l_q_center", 0.03)),
        l_q_out=float(config.get("l_q_out", 0.16)),
        l_kappa=float(config.get("l_kappa", 1.0)),
        l_stage_form=str(config.get("l_stage_form", "none")),
        l_stage_reward=float(config.get("l_stage_reward", 0.0)),
        reset_box_mode=str(config.get("reset_box_mode", "ours")),
        region_profile=str(config.get("region_profile", "v6")),
        reset_requires_safe=bool(config.get("reset_requires_safe", True)),
        safe_alpha_hi=float(config.get("safe_alpha_hi", 0.7853981633974483)),
        safe_beta=float(config.get("safe_beta", 0.5235987755982988)),
        train_safe_h_min=(float(config.get("train_safe_h_min", float("nan")))
                          if apply_training_safety_overrides else float("nan")),
        train_safe_h_max=(float(config.get("train_safe_h_max", float("nan")))
                          if apply_training_safety_overrides else float("nan")),
        train_safe_alpha_lo=(float(config.get("train_safe_alpha_lo", float("nan")))
                          if apply_training_safety_overrides else float("nan")),
        train_safe_alpha_hi=(float(config.get("train_safe_alpha_hi", float("nan")))
                          if apply_training_safety_overrides else float("nan")),
        train_safe_beta=(float(config.get("train_safe_beta", float("nan")))
                          if apply_training_safety_overrides else float("nan")),
        alpha_counts_as_invalid_dynamics=(
            True if strict_invalid_dynamics else bool(config.get("alpha_counts_as_invalid_dynamics", True))
        ),
        beta_counts_as_invalid_dynamics=(
            True if strict_invalid_dynamics else bool(config.get("beta_counts_as_invalid_dynamics", True))
        ),
        clip_relaxed_alpha_beta_to_terminate=(
            False if strict_invalid_dynamics else bool(config.get("clip_relaxed_alpha_beta_to_terminate", False))
        ),
    )
    env.set_eval_mode()
    return env


def _parse_hidden(value: str):
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def load_checkpoint(
    ckpt_path: str,
    eval_seed: int,
    *,
    strict_invalid_dynamics: bool = True,
    apply_training_safety_overrides: bool = True,
):
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    config = ckpt.get("config", {})
    env = make_env_from_config_v6(config, seed=eval_seed,
        strict_invalid_dynamics=strict_invalid_dynamics,
        apply_training_safety_overrides=apply_training_safety_overrides)
    policy_name = ckpt.get("policy", config.get("policy", ckpt.get("agent", config.get("agent", "ssm"))))
    agent_cls = SSMOnlineAgent
    ddpm_temperature = config.get("ddpm_temperature", 0.2)

    agent_kwargs = dict(
        seed=eval_seed,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor_hidden_dims=_parse_hidden(config.get("actor_hidden", "256,256,256")),
        critic_hidden_dims=_parse_hidden(config.get("critic_hidden", "256,256")),
        M_q=config.get("M_q", 10.0),
        alpha_r=config.get("alpha_r", 1.0),
        beta_safety=config.get("beta_safety", 5.0),
        delta=config.get("delta", 0.0),
        hard_margin=config.get("hard_margin", 0.0),
        discount=config.get("discount", 0.995),
        discount_h=config.get("discount_h", 0.999),
        tau=config.get("tau", 0.005),
        T=config.get("T", 5),
        beta_schedule=config.get("beta_schedule", "vp"),
        cost_critic_hyperparam=config.get("cost_critic_hyperparam", 0.9),
        safety_estimator=config.get("safety_estimator", "sampled_qh"),
        vc_mode=config.get("vc_mode", "buffer_expectile"),
        vc_policy_samples=config.get("vc_policy_samples", 0),
        vc_proposal_samples=config.get("vc_proposal_samples", 0),
        vc_proposal_std=config.get("vc_proposal_std", 0.6),
        pure_qsm=config.get("pure_qsm", False),
        ddpm_temperature=ddpm_temperature,
        fuse_clip_min=config.get("fuse_clip_min", -2000.0),
        fuse_clip_max=config.get("fuse_clip_max", 100.0),
        huber_delta=config.get("huber_delta", 10.0),
    )
    agent_kwargs.update(
        stage_mode=config.get("qh_stage_mode", "stage_a"),
        qh_candidate_source=config.get("qh_candidate_source", "policy_plus_gaussian"),
        qh_min_k_samples=config.get("qh_min_k_samples", 8),
        qh_min_k_sigma=config.get("qh_min_k_sigma", 0.3),
    )
    agent = agent_cls.create(**agent_kwargs)

    if agent.safety_estimator != "sampled_qh":
        raise ValueError("F16 evaluation requires a sampled-Qh checkpoint.")
    params = ckpt["params"]
    expected = {name for name in agent.__dataclass_fields__ if hasattr(getattr(agent, name), "params")}
    # Early sampled-Qh checkpoints omit the unused expectile-value heads.
    optional = {"safe_value", "safe_target_value"}
    if not (expected - optional <= set(params) <= expected):
        raise ValueError("Checkpoint parameter heads do not match the configured learner")
    replacements = {}
    for key in sorted(params):
        template_leaves, template_tree = jax.tree_util.tree_flatten(getattr(agent, key).params)
        leaves, tree = jax.tree_util.tree_flatten(params[key])
        if tree != template_tree or [x.shape for x in leaves] != [x.shape for x in template_leaves]:
            raise ValueError(f"Checkpoint parameter tree/shape mismatch: {key}")
        replacements[key] = getattr(agent, key).replace(params=params[key])
    agent = agent.replace(**replacements)
    agent = agent.replace(rng=jax.random.PRNGKey(eval_seed))
    return agent, config, policy_name, env


def diagnostic_grid(env, grid):
    """Nominal theta/height grid, meshgrid/C order and float64 states."""
    theta = np.linspace(grid["theta_min"], grid["theta_max"], num=grid["n_theta"])
    height = np.linspace(grid["height_min"], grid["height_max"], num=grid["n_height"])
    tt, hh = np.meshgrid(theta, height)
    states = np.tile(env.nominal_state_v5()[None, None, :], (len(height), len(theta), 1))
    states[:, :, IDX_THETA] = tt
    states[:, :, IDX_H] = hh
    return np.asarray(states.reshape(-1, states.shape[-1]), dtype=np.float64)


def score_grid(agent, env, states, gate_seed, chunk_size):
    """Checkpoint-defined Qh gate: policy draw for A, candidate minimum for B."""
    scores = np.empty(len(states), dtype=np.float64)
    key = jax.random.PRNGKey(gate_seed)
    for start in range(0, len(states), chunk_size):
        chunk = states[start:start + chunk_size].astype(np.float32)
        obs = np.stack([env._get_obs(x.astype(np.float64)) for x in chunk]).astype(np.float32)
        qh, _, key = SSMOnlineAgent._estimate_qh_from_candidates(
            agent, jnp.asarray(obs), key, params=agent.safe_critic.params, use_target=False)
        scores[start:start + len(chunk)] = np.asarray(jax.device_get(qh), dtype=np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Non-finite Qh scores; coverage is undefined")
    return scores


def checkpoint_rollout(env):
    """Bind resolved geometry and invalid-dynamics rules to the rollout scan."""
    if env.clip_relaxed_alpha_beta_to_terminate:
        raise ValueError("The scan does not support relaxed alpha/beta state clipping.")
    keys = ("safety_gap_mode", "h_theta_denom", "obs_task_feats_mode", "l_form", "l_q_center", "l_q_out", "l_kappa",
            "l_stage_form", "l_stage_reward", "safe_alpha_hi", "safe_alpha_lo", "safe_beta", "safe_h_min", "safe_h_max",
            "safe_theta", "safe_pe", "safe_p", "terminate_h_min", "terminate_h_max", "terminate_pe_limit", "terminate_p")
    return _build_rollout_fn(**{key: getattr(env, key) for key in keys},
                             alpha_counts_as_invalid_dynamics=env.alpha_counts_as_invalid_dynamics,
                             beta_counts_as_invalid_dynamics=env.beta_counts_as_invalid_dynamics)


def strict_rollout(env):
    """Bind strict invalid-dynamics termination, independently of training flags."""
    keys = ("safety_gap_mode", "h_theta_denom", "obs_task_feats_mode", "l_form", "l_q_center", "l_q_out", "l_kappa",
            "l_stage_form", "l_stage_reward", "safe_alpha_hi", "safe_alpha_lo", "safe_beta", "safe_h_min", "safe_h_max",
            "safe_theta", "safe_pe", "safe_p", "terminate_h_min", "terminate_h_max", "terminate_pe_limit", "terminate_p")
    return _build_rollout_fn(**{key: getattr(env, key) for key in keys},
                            alpha_counts_as_invalid_dynamics=True, beta_counts_as_invalid_dynamics=True)

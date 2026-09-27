"""Accelerated F16 stabilize-avoid environment backend."""

from __future__ import annotations

from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax_f16.f16 import F16

import ssm.tasks.f16.env as base


TASK_SPEC_VERSION_V6 = base.TASK_SPEC_VERSION_V6
ENV_BACKEND_P2 = "jit_core"

_F16_MODEL = F16()
_ANGLE_IDXS_JNP = jnp.asarray(base._ANGLE_IDXS)
_OTHER_IDXS_JNP = jnp.asarray(base._OTHER_IDXS)
_OBS_MEAN_JNP = jnp.asarray(base._OBS_MEAN, dtype=jnp.float32)
_OBS_STD_JNP = jnp.asarray(base._OBS_STD, dtype=jnp.float32)
_CONTROL_LOW_JNP = jnp.array([-10.0, -10.0, -10.0, 0.0], dtype=jnp.float32)
_CONTROL_HIGH_JNP = jnp.array([15.0, 10.0, 10.0, 1.0], dtype=jnp.float32)
_DT = 0.05


def _decode_action_jax(action):
    action_clipped = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
    control = 0.5 * (_CONTROL_HIGH_JNP - _CONTROL_LOW_JNP) * action_clipped
    control = control + 0.5 * (_CONTROL_HIGH_JNP + _CONTROL_LOW_JNP)
    return action_clipped, control


def _task_goal_distance_jax(x):
    return jnp.maximum(jnp.abs(x[base.IDX_H] - base._GOAL_H_MID) - base._GOAL_H_HALFWIDTH, 0.0) / 250.0


def _task_l_value_jax(
    x,
    l_form: str,
    l_q_center: float,
    l_q_out: float,
    l_kappa: float,
    l_stage_form: str,
    l_stage_reward: float,
):
    z = (x[base.IDX_H] - base._GOAL_H_MID) / base._GOAL_H_HALFWIDTH
    d_out = jnp.maximum(jnp.abs(z) - 1.0, 0.0)
    l_center = l_q_center * jnp.minimum(z**2, 1.0)
    if l_form == base.L_FORM_SPLIT_LINEAR:
        phi = d_out
    elif l_form == base.L_FORM_SPLIT_LOG:
        phi = jnp.log1p(l_kappa * d_out) / l_kappa
    elif l_form == base.L_FORM_SPLIT_ATAN:
        phi = jnp.arctan(l_kappa * d_out) / l_kappa
    else:
        raise ValueError(f"Unsupported v5 l_form={l_form!r}.")
    l_base = l_center + l_q_out * phi
    if l_stage_form == base.L_STAGE_FORM_BAND_BOWL:
        l_base = l_base - l_stage_reward * jnp.maximum(1.0 - z**2, 0.0)
    return l_base


def _task_h_components_raw_jax(
    x,
    h_theta_denom: float,
    safe_alpha_hi: float,
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
    safe_alpha_lo: float = base._SAFE_ALPHA_LO,
):
    h_altitude = jnp.maximum(
        (safe_h_min - x[base.IDX_H]) / (safe_h_min - terminate_h_min),
        (x[base.IDX_H] - safe_h_max) / (terminate_h_max - safe_h_max),
    )
    h_alpha = jnp.maximum(
        (safe_alpha_lo - x[base.IDX_ALPHA]) / (safe_alpha_lo - base._TERMINATE_ALPHA_LO),
        (x[base.IDX_ALPHA] - safe_alpha_hi) / (base._TERMINATE_ALPHA_HI - safe_alpha_hi),
    )
    h_beta = (jnp.abs(x[base.IDX_BETA]) - safe_beta) / (base._TERMINATE_BETA - safe_beta)
    h_theta = (jnp.abs(x[base.IDX_THETA]) - safe_theta) / h_theta_denom
    h_pe = (jnp.abs(x[base.IDX_PE]) - safe_pe) / (terminate_pe_limit - safe_pe)
    h_p = (jnp.abs(x[base.IDX_P]) - safe_p) / (terminate_p - safe_p)
    return jnp.array([h_altitude, h_alpha, h_beta, h_theta, h_pe, h_p], dtype=jnp.float32)


def _task_h_components_jax(
    x,
    use_literal_gap: bool,
    h_theta_denom: float,
    safe_alpha_hi: float,
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
    safe_alpha_lo: float = base._SAFE_ALPHA_LO,
):
    h_raw = _task_h_components_raw_jax(
        x,
        h_theta_denom=h_theta_denom,
        safe_alpha_hi=safe_alpha_hi,
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
        safe_alpha_lo=safe_alpha_lo,
    )
    if use_literal_gap:
        return jnp.where(h_raw >= 0.0, h_raw + 0.5, h_raw - 0.5)
    return h_raw


def _task_safety_margin_jax(
    x,
    use_literal_gap: bool,
    h_theta_denom: float,
    safe_alpha_hi: float,
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
    safe_alpha_lo: float = base._SAFE_ALPHA_LO,
):
    return jnp.max(
        _task_h_components_jax(
            x,
            use_literal_gap=use_literal_gap,
            h_theta_denom=h_theta_denom,
            safe_alpha_hi=safe_alpha_hi,
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
            safe_alpha_lo=safe_alpha_lo,
        )
    )


def _is_valid_v6_jax(x, alpha_counts_as_invalid_dynamics: bool, beta_counts_as_invalid_dynamics: bool):
    finite_ok = jnp.all(jnp.isfinite(x)) & jnp.isfinite(x[base.IDX_H]) & jnp.isfinite(x[base.IDX_PE])
    alpha_ok = (base._TERMINATE_ALPHA_LO <= x[base.IDX_ALPHA]) & (x[base.IDX_ALPHA] <= base._TERMINATE_ALPHA_HI)
    beta_ok = jnp.abs(x[base.IDX_BETA]) <= base._TERMINATE_BETA
    theta_ok = jnp.abs(x[base.IDX_THETA]) < base._TERMINATE_THETA
    p_ok = jnp.abs(x[base.IDX_P]) < base._TERMINATE_P
    alpha_valid = jnp.logical_or(alpha_ok, jnp.logical_not(jnp.asarray(alpha_counts_as_invalid_dynamics)))
    beta_valid = jnp.logical_or(beta_ok, jnp.logical_not(jnp.asarray(beta_counts_as_invalid_dynamics)))
    return finite_ok & alpha_valid & beta_valid & theta_ok & p_ok


def _hits_hard_terminal_v6_jax(x, terminate_h_min: float, terminate_h_max: float, terminate_pe_limit: float):
    return (
        (x[base.IDX_H] <= terminate_h_min)
        | (x[base.IDX_H] >= terminate_h_max)
        | (jnp.abs(x[base.IDX_PE]) >= terminate_pe_limit)
    )


def _simulate_transition_core_impl(
    state,
    action,
    use_literal_gap: bool,
    h_theta_denom: float,
    l_form: str,
    l_q_center: float,
    l_q_out: float,
    l_kappa: float,
    l_stage_form: str,
    l_stage_reward: float,
    safe_alpha_hi: float,
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
    safe_alpha_lo: float = base._SAFE_ALPHA_LO,
):
    action_clipped, control = _decode_action_jax(action)
    x = jnp.asarray(state)
    u = jnp.asarray(control)

    k1 = _F16_MODEL.xdot(x, u)
    k2 = _F16_MODEL.xdot(x + 0.5 * _DT * k1, u)
    k3 = _F16_MODEL.xdot(x + 0.5 * _DT * k2, u)
    k4 = _F16_MODEL.xdot(x + _DT * k3, u)
    x_raw = x + (_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    h_components = _task_h_components_jax(
        x_raw,
        use_literal_gap=use_literal_gap,
        h_theta_denom=h_theta_denom,
        safe_alpha_hi=safe_alpha_hi,
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
        safe_alpha_lo=safe_alpha_lo,
    )
    h_margin = jnp.max(h_components)
    task_cost = _task_l_value_jax(
        x_raw,
        l_form=l_form,
        l_q_center=l_q_center,
        l_q_out=l_q_out,
        l_kappa=l_kappa,
        l_stage_form=l_stage_form,
        l_stage_reward=l_stage_reward,
    )
    goal_distance = _task_goal_distance_jax(x_raw)
    in_goal = goal_distance <= 0.0
    safe = h_margin <= 0.0
    invalid_dynamics = jnp.logical_not(
        _is_valid_v6_jax(
            x_raw,
            alpha_counts_as_invalid_dynamics=alpha_counts_as_invalid_dynamics,
            beta_counts_as_invalid_dynamics=beta_counts_as_invalid_dynamics,
        )
    )
    hard_terminal = _hits_hard_terminal_v6_jax(
        x_raw,
        terminate_h_min=terminate_h_min,
        terminate_h_max=terminate_h_max,
        terminate_pe_limit=terminate_pe_limit,
    )
    terminated = invalid_dynamics | hard_terminal

    return (
        x_raw,
        action_clipped,
        control,
        h_components,
        h_margin,
        task_cost,
        -task_cost,
        goal_distance,
        in_goal,
        safe,
        invalid_dynamics,
        hard_terminal,
        terminated,
    )


@partial(jax.jit, static_argnames=("use_literal_gap", "l_form", "l_stage_form"))
def _simulate_transition_core(
    state,
    action,
    use_literal_gap: bool,
    h_theta_denom: float,
    l_form: str,
    l_q_center: float,
    l_q_out: float,
    l_kappa: float,
    l_stage_form: str,
    l_stage_reward: float,
    safe_alpha_hi: float,
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
    safe_alpha_lo: float = base._SAFE_ALPHA_LO,
):
    return _simulate_transition_core_impl(
        state,
        action,
        use_literal_gap,
        h_theta_denom,
        l_form,
        l_q_center,
        l_q_out,
        l_kappa,
        l_stage_form,
        l_stage_reward,
        safe_alpha_hi,
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
        alpha_counts_as_invalid_dynamics,
        beta_counts_as_invalid_dynamics,
        safe_alpha_lo,
    )


def make_transition_core(
    use_literal_gap: bool,
    h_theta_denom: float,
    l_form: str,
    l_q_center: float,
    l_q_out: float,
    l_kappa: float,
    l_stage_form: str,
    l_stage_reward: float,
    safe_alpha_hi: float,
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
    safe_alpha_lo: float = base._SAFE_ALPHA_LO,
):
    return partial(
        _simulate_transition_core,
        use_literal_gap=use_literal_gap,
        h_theta_denom=float(h_theta_denom),
        l_form=l_form,
        l_q_center=float(l_q_center),
        l_q_out=float(l_q_out),
        l_kappa=float(l_kappa),
        l_stage_form=l_stage_form,
        l_stage_reward=float(l_stage_reward),
        safe_alpha_hi=float(safe_alpha_hi),
        safe_beta=float(safe_beta),
        safe_h_min=float(safe_h_min),
        safe_h_max=float(safe_h_max),
        safe_theta=float(safe_theta),
        safe_pe=float(safe_pe),
        safe_p=float(safe_p),
        terminate_h_min=float(terminate_h_min),
        terminate_h_max=float(terminate_h_max),
        terminate_pe_limit=float(terminate_pe_limit),
        terminate_p=float(terminate_p),
        alpha_counts_as_invalid_dynamics=bool(alpha_counts_as_invalid_dynamics),
        beta_counts_as_invalid_dynamics=bool(beta_counts_as_invalid_dynamics),
        safe_alpha_lo=float(safe_alpha_lo),
    )


def _state_enc_jax(x):
    x_jnp = jnp.asarray(x)
    angles = x_jnp[_ANGLE_IDXS_JNP]
    other = x_jnp[_OTHER_IDXS_JNP]
    angles_enc = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)])
    state_enc = jnp.concatenate([other, angles_enc])
    vel_feats = base._compute_vel_angles(x_jnp)
    state_enc = jnp.concatenate([state_enc, vel_feats])
    state_enc = (state_enc - _OBS_MEAN_JNP) / _OBS_STD_JNP
    state_enc = jnp.clip(state_enc, -10.0, 10.0)
    return state_enc


@partial(jax.jit, static_argnames=("use_literal_gap",))
def _get_obs_core_aggregate(
    x,
    use_literal_gap: bool,
    h_theta_denom: float,
    safe_alpha_hi: float,
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
    safe_alpha_lo: float = base._SAFE_ALPHA_LO,
):
    x_jnp = jnp.asarray(x)
    state_enc = _state_enc_jax(x_jnp)

    task_feats = jnp.array(
        [
            jnp.tanh(
                _task_safety_margin_jax(
                    x_jnp,
                    use_literal_gap=use_literal_gap,
                    h_theta_denom=h_theta_denom,
                    safe_alpha_hi=safe_alpha_hi,
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
                    safe_alpha_lo=safe_alpha_lo,
                )
            ),
            jnp.tanh(_task_goal_distance_jax(x_jnp)),
        ],
        dtype=jnp.float32,
    )
    return jnp.concatenate([state_enc, task_feats]).astype(jnp.float32)


@partial(jax.jit, static_argnames=("use_literal_gap",))
def _get_obs_core_per_axis(
    x,
    use_literal_gap: bool,
    h_theta_denom: float,
    safe_alpha_hi: float,
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
    safe_alpha_lo: float = base._SAFE_ALPHA_LO,
):
    x_jnp = jnp.asarray(x)
    state_enc = _state_enc_jax(x_jnp)
    h_components = _task_h_components_jax(
        x_jnp,
        use_literal_gap=use_literal_gap,
        h_theta_denom=h_theta_denom,
        safe_alpha_hi=safe_alpha_hi,
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
        safe_alpha_lo=safe_alpha_lo,
    )
    h_margin = jnp.max(h_components)
    task_feats = jnp.array(
        [
            jnp.tanh(h_margin),
            jnp.tanh(_task_goal_distance_jax(x_jnp)),
            *jnp.tanh(h_components),
        ],
        dtype=jnp.float32,
    )
    return jnp.concatenate([state_enc, task_feats]).astype(jnp.float32)


def make_obs_core(
    use_literal_gap: bool,
    h_theta_denom: float,
    safe_alpha_hi: float,
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
    obs_task_feats_mode: str,
    safe_alpha_lo: float = base._SAFE_ALPHA_LO,
):
    common_kwargs = dict(
        use_literal_gap=use_literal_gap,
        h_theta_denom=float(h_theta_denom),
        safe_alpha_hi=float(safe_alpha_hi),
        safe_beta=float(safe_beta),
        safe_h_min=float(safe_h_min),
        safe_h_max=float(safe_h_max),
        safe_theta=float(safe_theta),
        safe_pe=float(safe_pe),
        safe_p=float(safe_p),
        terminate_h_min=float(terminate_h_min),
        terminate_h_max=float(terminate_h_max),
        terminate_pe_limit=float(terminate_pe_limit),
        terminate_p=float(terminate_p),
        safe_alpha_lo=float(safe_alpha_lo),
    )
    if obs_task_feats_mode == base.OBS_TASK_FEATS_MODE_AGGREGATE:
        return partial(_get_obs_core_aggregate, **common_kwargs)
    if obs_task_feats_mode == base.OBS_TASK_FEATS_MODE_PER_AXIS:
        return partial(_get_obs_core_per_axis, **common_kwargs)
    raise ValueError(f"Unsupported obs_task_feats_mode={obs_task_feats_mode!r}.")


class F16StabilizeEnvV6P2(base.F16StabilizeEnvV6):
    """P2 env with the same v6 contract and a jitted single-env hot path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_backend = ENV_BACKEND_P2
        self._literal_gap = self.safety_gap_mode == "literal"
        self._simulate_transition_core = make_transition_core(
            use_literal_gap=self._literal_gap,
            h_theta_denom=self.h_theta_denom,
            l_form=self.l_form,
            l_q_center=self.l_q_center,
            l_q_out=self.l_q_out,
            l_kappa=self.l_kappa,
            l_stage_form=self.l_stage_form,
            l_stage_reward=self.l_stage_reward,
            safe_alpha_hi=self.safe_alpha_hi,
            safe_alpha_lo=self.safe_alpha_lo,
            safe_beta=self.safe_beta,
            safe_h_min=self.safe_h_min,
            safe_h_max=self.safe_h_max,
            safe_theta=self.safe_theta,
            safe_pe=self.safe_pe,
            safe_p=self.safe_p,
            terminate_h_min=self.terminate_h_min,
            terminate_h_max=self.terminate_h_max,
            terminate_pe_limit=self.terminate_pe_limit,
            terminate_p=self.terminate_p,
            alpha_counts_as_invalid_dynamics=self.alpha_counts_as_invalid_dynamics,
            beta_counts_as_invalid_dynamics=self.beta_counts_as_invalid_dynamics,
        )
        self._obs_core = make_obs_core(
            use_literal_gap=self._literal_gap,
            h_theta_denom=self.h_theta_denom,
            safe_alpha_hi=self.safe_alpha_hi,
            safe_alpha_lo=self.safe_alpha_lo,
            safe_beta=self.safe_beta,
            safe_h_min=self.safe_h_min,
            safe_h_max=self.safe_h_max,
            safe_theta=self.safe_theta,
            safe_pe=self.safe_pe,
            safe_p=self.safe_p,
            terminate_h_min=self.terminate_h_min,
            terminate_h_max=self.terminate_h_max,
            terminate_pe_limit=self.terminate_pe_limit,
            terminate_p=self.terminate_p,
            obs_task_feats_mode=self.obs_task_feats_mode,
        )
        self._f16 = _F16_MODEL
        # Same scalar RK4 dynamics/observation functions, explicitly compiled on CPU.
        # The learner and action sampler retain the process default GPU backend.
        cpu = jax.devices("cpu")[0]
        for attr, static in (
            ("_simulate_transition_core", ("use_literal_gap", "l_form", "l_stage_form")),
            ("_obs_core", ("use_literal_gap",)),
        ):
            core = getattr(self, attr)
            compiled = jax.jit(core.func.__wrapped__, backend="cpu", static_argnames=static)
            setattr(self, attr, partial(compiled, **{
                name: value if name in static else jax.device_put(value, cpu)
                for name, value in core.keywords.items()
            }))

    def simulate_transition(self, state: np.ndarray, action: np.ndarray) -> Dict[str, object]:
        outputs = self._simulate_transition_core(np.asarray(state), np.asarray(action, dtype=np.float32))
        (
            x_raw,
            action_clipped,
            control,
            h_components,
            h_margin,
            task_cost,
            reward,
            goal_distance,
            in_goal,
            safe,
            invalid_dynamics,
            hard_terminal,
            terminated,
        ) = jax.device_get(outputs)

        x_raw_np = np.asarray(x_raw, dtype=np.float64)
        validity_flags = self._validity_flags_v5(x_raw_np)
        invalid_dynamics_b = bool(invalid_dynamics)
        hard_terminal_b = bool(hard_terminal)
        terminated_b = bool(terminated)
        terminated_reason = self._termination_reason_from_flags_v5(invalid_dynamics_b, hard_terminal_b)
        crash_cause = self._classify_crash_cause_v5(x_raw_np) if terminated_b else "none"
        return {
            "next_state": x_raw_np,
            "raw_next_state": x_raw_np,
            "action_clipped": np.asarray(action_clipped, dtype=np.float32),
            "control": np.asarray(control, dtype=np.float32),
            "h_components": np.asarray(h_components, dtype=np.float64),
            "h_margin": float(h_margin),
            "task_cost": float(task_cost),
            "reward": float(reward),
            "binary_cost": float(bool(h_margin > 0.0)),
            "goal_distance": float(goal_distance),
            "in_goal": bool(in_goal),
            "safe": bool(safe),
            "invalid_dynamics": invalid_dynamics_b,
            "would_invalid_alpha": bool(validity_flags["alpha_invalid"]),
            "would_invalid_beta": bool(validity_flags["beta_invalid"]),
            "would_invalid_theta": bool(validity_flags["theta_invalid"]),
            "would_invalid_p": bool(validity_flags["p_invalid"]),
            "hard_terminal": hard_terminal_b,
            "terminated": terminated_b,
            "terminated_reason": terminated_reason,
            "crash_cause": crash_cause,
        }

    def _get_obs(self, x: np.ndarray) -> np.ndarray:
        obs = self._obs_core(np.asarray(x))
        return np.asarray(jax.device_get(obs), dtype=np.float32)

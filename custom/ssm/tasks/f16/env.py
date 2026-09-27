"""F16 stabilize-avoid v6 environment for SSM training.

Reset distributions, safety margins, and termination bounds are configurable.
The region profile and optional safety-bound overrides determine the geometry;
reset rejection is controlled by ``reset_requires_safe``. A terminal event can
arise from invalid dynamics or crossing an altitude or lateral-position bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from jax_f16.f16 import F16

from ssm.common.gym_compat import spaces as gym_spaces


TASK_SPEC_VERSION_V6 = "ssm_f16stab_v6_three_region_linear_split_per_axis_obs"
H_FORM_LINEAR = "linear"
L_FORM_SPLIT_LINEAR = "split_linear"
L_FORM_SPLIT_LOG = "split_log"
L_FORM_SPLIT_ATAN = "split_atan"
L_STAGE_FORM_NONE = "none"
L_STAGE_FORM_BAND_BOWL = "band_bowl"
OBS_TASK_FEATS_MODE_AGGREGATE = "aggregate"
OBS_TASK_FEATS_MODE_PER_AXIS = "per_axis"
RESET_BOX_MODE_OURS = "ours"
RESET_BOX_MODE_EFPPO_TRAIN = "efppo_train"
INIT_CURRICULUM_MODE_BOX = "box"
INIT_CURRICULUM_MODE_BOX_MIXTURE = "box_mixture"
INIT_CURRICULUM_MODE_AB_BOUNDARY = "ab_boundary"
INIT_CURRICULUM_MODE_THETA_TAIL_PROXY = "theta_tail_proxy"
INIT_CURRICULUM_MODE_EASY_TO_HARD = "easy_to_hard"

IDX_VT, IDX_ALPHA, IDX_BETA = 0, 1, 2
IDX_PHI, IDX_THETA, IDX_PSI = 3, 4, 5
IDX_P, IDX_Q, IDX_R = 6, 7, 8
IDX_PN, IDX_PE, IDX_H = 9, 10, 11
IDX_POW = 12
IDX_NZINT = 13
IDX_PSINT = 14
IDX_NYRINT = 15
NX = 16
NU = 4

_ANGLE_IDXS = np.array([IDX_ALPHA, IDX_BETA, IDX_PHI, IDX_THETA, IDX_PSI])
_OTHER_IDXS = np.array([i for i in range(NX) if i not in _ANGLE_IDXS])

# Morelli aerodynamic validity bounds used by the official EFPPO F16 task.
_ALPHA_LO = -0.17453292519943295
_ALPHA_HI = 0.7853981633974483
_BETA_LO = -0.5235987755982988
_BETA_HI = 0.5235987755982988

# Observation normalization constants for the 24D encoded state prior to the
# appended task features.
# fmt: off
_OBS_MEAN = np.array([3.4e+02, -1.7e-01, 2.9e-01, 1.0e-01, 1.0e+03, 1.0e-01, 3.1e+02, 12.0e+00,
                       3.0e-02, 2.1e-02, 2.4e+00, 8.7e-01, 8.9e-01, 7.7e-01, 8.0e-01,
                       7.6e-01, 3.6e-01, -1.3e-02, -7.6e-03, -3.5e-01, -1.4e-03, 5.9e-01, -3.6e-03,
                       5.4e-01])
_OBS_STD = np.array([1.1e+02, 1.7e+00, 6.3e-01, 3.2e+00, 1.0e+03, 1.3e+02, 2.2e+02, 5.0e+00,
                      1.9e+00, 1.5e+00, 4.5e+00, 1.4e-01, 9.6e-02, 2.6e-01, 2.2e-01,
                      3.7e-01, 3.1e-01, 4.4e-01, 5.8e-01, 4.4e-01, 5.4e-01, 3.5e-01, 3.1e-01,
                      3.8e-01])
# fmt: on


def _rotz(psi):
    c, s = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _roty(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotx(phi):
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _compute_vel_angles(x):
    """Velocity-vector features used by the 24D encoded observation."""
    r_world = _rotz(x[IDX_PSI]) @ _roty(x[IDX_THETA]) @ _rotx(x[IDX_PHI])
    ca, sa = jnp.cos(x[IDX_ALPHA]), jnp.sin(x[IDX_ALPHA])
    cb, sb = jnp.cos(x[IDX_BETA]), jnp.sin(x[IDX_BETA])
    v_body = jnp.array([ca * cb, sb, sa * cb])
    v_world = r_world @ v_body
    return jnp.array([v_world[0], v_world[1], v_world[2]])

_GOAL_H_MIN = 50.0
_GOAL_H_MAX = 150.0
_GOAL_H_MID = 0.5 * (_GOAL_H_MIN + _GOAL_H_MAX)
_GOAL_H_HALFWIDTH = 0.5 * (_GOAL_H_MAX - _GOAL_H_MIN)

_SAFE_H_MIN = 50.0
_SAFE_H_MAX = 1000.0
_SAFE_ALPHA_LO = _ALPHA_LO
_SAFE_ALPHA_HI = _ALPHA_HI
_SAFE_BETA = _BETA_HI
_SAFE_THETA = 1.40
_SAFE_PE = 200.0
_SAFE_P = 8.0

_TERMINATE_ALPHA_MARGIN = 0.10
_TERMINATE_BETA_MARGIN = 0.10
_TERMINATE_ALPHA_LO = _SAFE_ALPHA_LO - _TERMINATE_ALPHA_MARGIN
_TERMINATE_ALPHA_HI = _SAFE_ALPHA_HI + _TERMINATE_ALPHA_MARGIN
_TERMINATE_BETA = _SAFE_BETA + _TERMINATE_BETA_MARGIN
_TERMINATE_THETA = np.pi / 2.0
_TERMINATE_P = 10.0
_TERMINATE_H_MIN = 0.0
_TERMINATE_H_MAX = 1100.0
_TERMINATE_PE_LIMIT = 250.0

_TASK_H_LABELS = ("alt", "alpha", "beta", "theta", "pe", "p")
_TASK_FEAT_LABELS_AGGREGATE = ("h_margin", "goal_distance")
_TASK_FEAT_LABELS_PER_AXIS = ("h_margin", "goal_distance", "h_alt", "h_alpha", "h_beta", "h_theta", "h_pe", "h_p")
_DEFAULT_H_THETA_DENOM = _TERMINATE_THETA - _SAFE_THETA
_VALID_H_FORMS = (H_FORM_LINEAR,)
_VALID_L_FORMS = (L_FORM_SPLIT_LINEAR, L_FORM_SPLIT_LOG, L_FORM_SPLIT_ATAN)
_VALID_L_STAGE_FORMS = (L_STAGE_FORM_NONE, L_STAGE_FORM_BAND_BOWL)
_VALID_OBS_TASK_FEATS_MODES = (OBS_TASK_FEATS_MODE_AGGREGATE, OBS_TASK_FEATS_MODE_PER_AXIS)
_VALID_RESET_BOX_MODES = (RESET_BOX_MODE_OURS, RESET_BOX_MODE_EFPPO_TRAIN)
REGION_PROFILE_V6 = "v6"
REGION_PROFILE_EFPPO_PLUS_P = "efppo_plus_p"
_VALID_REGION_PROFILES = (REGION_PROFILE_V6, REGION_PROFILE_EFPPO_PLUS_P)

_EFPPO_PLUS_P_SAFE_H_MIN = 0.0
_EFPPO_PLUS_P_SAFE_H_MAX = 1000.0
_EFPPO_PLUS_P_SAFE_THETA = 0.95 * np.pi / 2.0
_EFPPO_PLUS_P_TERMINATE_H_MIN = -100.0
_EFPPO_PLUS_P_TERMINATE_H_MAX = 1100.0
_EFPPO_PLUS_P_TERMINATE_THETA = np.pi / 2.0
_EFPPO_PLUS_P_TERMINATE_P = 10.0
_EFPPO_PLUS_P_TERMINATE_PE_LIMIT = 250.0


def resolve_region_geometry(region_profile: str) -> Dict[str, float]:
    if region_profile == REGION_PROFILE_V6:
        return {
            "safe_h_min": float(_SAFE_H_MIN),
            "safe_h_max": float(_SAFE_H_MAX),
            "safe_theta": float(_SAFE_THETA),
            "safe_pe": float(_SAFE_PE),
            "safe_p": float(_SAFE_P),
            "terminate_h_min": float(_TERMINATE_H_MIN),
            "terminate_h_max": float(_TERMINATE_H_MAX),
            "terminate_theta": float(_TERMINATE_THETA),
            "terminate_pe_limit": float(_TERMINATE_PE_LIMIT),
            "terminate_p": float(_TERMINATE_P),
        }
    if region_profile == REGION_PROFILE_EFPPO_PLUS_P:
        return {
            "safe_h_min": float(_EFPPO_PLUS_P_SAFE_H_MIN),
            "safe_h_max": float(_EFPPO_PLUS_P_SAFE_H_MAX),
            "safe_theta": float(_EFPPO_PLUS_P_SAFE_THETA),
            "safe_pe": float(_SAFE_PE),
            "safe_p": float(_SAFE_P),
            "terminate_h_min": float(_EFPPO_PLUS_P_TERMINATE_H_MIN),
            "terminate_h_max": float(_EFPPO_PLUS_P_TERMINATE_H_MAX),
            "terminate_theta": float(_EFPPO_PLUS_P_TERMINATE_THETA),
            "terminate_pe_limit": float(_EFPPO_PLUS_P_TERMINATE_PE_LIMIT),
            "terminate_p": float(_EFPPO_PLUS_P_TERMINATE_P),
        }
    raise ValueError(f"Unsupported region_profile={region_profile!r}.")


def resolve_h_theta_denom(region_profile: str, requested_h_theta_denom: float) -> float:
    geometry = resolve_region_geometry(region_profile)
    profile_default = float(geometry["terminate_theta"] - geometry["safe_theta"])
    if abs(float(requested_h_theta_denom) - float(_DEFAULT_H_THETA_DENOM)) <= 1e-12:
        return profile_default
    return float(requested_h_theta_denom)

_BASE_BOX_V5 = np.array(
    [
        (150.0, 550.0),                 # VT
        (-0.025, 0.35),                 # ALPHA
        (-0.15, 0.15),                  # BETA
        (-np.pi / 3.0, np.pi / 3.0),    # PHI
        (-0.60, 0.60),                  # THETA
        (-1e-4, 1e-4),                  # PSI
        (-1.0, 1.0),                    # P
        (-0.5, 0.5),                    # Q
        (-2.0 * np.pi, 2.0 * np.pi),    # R
        (-1000.0, 1000.0),              # PN
        (-100.0, 100.0),                # PE
        (150.0, 600.0),                 # H
        (0.0, 10.0),                    # POW
        (-2.0, 2.0),                    # NZINT
        (-2.0, 2.0),                    # PSINT
        (-2.0, 2.0),                    # NYRINT
    ],
    dtype=np.float64,
)

# Canonical EFPPO training proposal box from efppo/src/efppo/task/f16.py::train_bounds().
_BASE_BOX_EFPPO_TRAIN = np.array(
    [
        (150.0, 550.0),                 # VT
        (_ALPHA_LO, _ALPHA_HI),         # ALPHA
        (_BETA_LO, _BETA_HI),           # BETA
        (-np.pi / 3.0, np.pi / 3.0),    # PHI
        (-1.4, 0.4),                    # THETA
        (-1e-4, 1e-4),                  # PSI
        (-0.5, 0.5),                    # P
        (-0.5, 0.5),                    # Q
        (-2.0 * np.pi, 2.0 * np.pi),    # R
        (-1000.0, 1000.0),              # PN
        (-210.0, 210.0),                # PE
        (-10.0, 700.0),                 # H
        (0.0, 10.0),                    # POW
        (-2.0, 2.0),                    # NZINT
        (-2.0, 2.0),                    # PSINT
        (-2.0, 2.0),                    # NYRINT
    ],
    dtype=np.float64,
)

_RESET_OVERRIDE_KEYS = {
    "init_vt": IDX_VT,
    "init_alpha": IDX_ALPHA,
    "init_beta": IDX_BETA,
    "init_phi": IDX_PHI,
    "init_theta": IDX_THETA,
    "init_psi": IDX_PSI,
    "init_p": IDX_P,
    "init_q": IDX_Q,
    "init_r": IDX_R,
    "init_pn": IDX_PN,
    "init_pe": IDX_PE,
    "init_h": IDX_H,
    "init_pow": IDX_POW,
}

_CURRICULUM_INHERIT = float("nan")


def _curriculum_override_active(start_value: float, end_value: float) -> bool:
    return not (np.isnan(float(start_value)) and np.isnan(float(end_value)))


def _resolve_optional_curriculum_pair(
    start_value: float,
    end_value: float,
    base_value: float,
) -> Tuple[float, float] | None:
    start_spec = not np.isnan(float(start_value))
    end_spec = not np.isnan(float(end_value))
    if (not start_spec) and (not end_spec):
        return None
    start_eff = float(start_value) if start_spec else float(base_value)
    end_eff = float(end_value) if end_spec else start_eff
    return start_eff, end_eff


def _optional_float(value: float, default_value: float) -> float:
    return float(default_value) if np.isnan(float(value)) else float(value)


@dataclass(frozen=True)
class SamplerMetadata:
    tries: int
    rejects: int

    @property
    def reject_rate(self) -> float:
        return float(self.rejects / self.tries) if self.tries > 0 else 0.0


class F16StabilizeEnvV6:
    """F16 stabilize-avoid environment with configurable three-region geometry."""

    def __init__(
        self,
        seed: int = 0,
        max_episode_steps: int = 640,
        goal_dwell_steps: int = 50,
        terminate_on_crash: bool = True,
        init_curriculum: bool = False,
        init_curriculum_mode: str = INIT_CURRICULUM_MODE_BOX,
        init_curriculum_frac: float = 0.0,
        init_curriculum_frac_end: float = 0.0,
        init_curriculum_box2_frac: float = 0.0,
        init_curriculum_box2_frac_end: float = 0.0,
        init_curriculum_anneal_start: int = 120000,
        init_curriculum_anneal_end: int = 400000,
        init_curriculum_h_min: float = _CURRICULUM_INHERIT,
        init_curriculum_h_max: float = _CURRICULUM_INHERIT,
        init_curriculum_h_min_end: float = _CURRICULUM_INHERIT,
        init_curriculum_h_max_end: float = _CURRICULUM_INHERIT,
        init_curriculum_box2_h_min: float = _CURRICULUM_INHERIT,
        init_curriculum_box2_h_max: float = _CURRICULUM_INHERIT,
        init_curriculum_box2_h_min_end: float = _CURRICULUM_INHERIT,
        init_curriculum_box2_h_max_end: float = _CURRICULUM_INHERIT,
        init_curriculum_pe_abs_min: float = _CURRICULUM_INHERIT,
        init_curriculum_pe_abs_max: float = _CURRICULUM_INHERIT,
        init_curriculum_pe_abs_min_end: float = _CURRICULUM_INHERIT,
        init_curriculum_pe_abs_max_end: float = _CURRICULUM_INHERIT,
        init_curriculum_theta_lo: float = _CURRICULUM_INHERIT,
        init_curriculum_theta_hi: float = _CURRICULUM_INHERIT,
        init_curriculum_theta_lo_end: float = _CURRICULUM_INHERIT,
        init_curriculum_theta_hi_end: float = _CURRICULUM_INHERIT,
        init_curriculum_box2_theta_lo: float = _CURRICULUM_INHERIT,
        init_curriculum_box2_theta_hi: float = _CURRICULUM_INHERIT,
        init_curriculum_box2_theta_lo_end: float = _CURRICULUM_INHERIT,
        init_curriculum_box2_theta_hi_end: float = _CURRICULUM_INHERIT,
        init_curriculum_alpha_lo: float = _CURRICULUM_INHERIT,
        init_curriculum_alpha_hi: float = _CURRICULUM_INHERIT,
        init_curriculum_alpha_lo_end: float = _CURRICULUM_INHERIT,
        init_curriculum_alpha_hi_end: float = _CURRICULUM_INHERIT,
        init_curriculum_beta_abs_max: float = _CURRICULUM_INHERIT,
        init_curriculum_beta_abs_max_end: float = _CURRICULUM_INHERIT,
        init_curriculum_ab_kind: str = "mixed",
        init_curriculum_ab_joint_frac: float = 0.0,
        init_curriculum_ab_alpha_h_lo: float = -0.60,
        init_curriculum_ab_alpha_h_hi: float = -0.15,
        init_curriculum_ab_alpha_h_lo_end: float = -0.60,
        init_curriculum_ab_alpha_h_hi_end: float = -0.15,
        init_curriculum_ab_beta_h_lo: float = -0.60,
        init_curriculum_ab_beta_h_hi: float = -0.15,
        init_curriculum_ab_beta_h_lo_end: float = -0.60,
        init_curriculum_ab_beta_h_hi_end: float = -0.15,
        init_curriculum_p_abs_max: float = _CURRICULUM_INHERIT,
        init_curriculum_p_abs_max_end: float = _CURRICULUM_INHERIT,
        init_curriculum_q_abs_max: float = _CURRICULUM_INHERIT,
        init_curriculum_q_abs_max_end: float = _CURRICULUM_INHERIT,
        init_curriculum_r_abs_max: float = _CURRICULUM_INHERIT,
        init_curriculum_r_abs_max_end: float = _CURRICULUM_INHERIT,
        init_curriculum_allow_in_goal: bool = False,
        init_curriculum_require_train_safe: bool = True,
        safety_gap_mode: str = "nogap",
        obs_task_feats_mode: str = OBS_TASK_FEATS_MODE_PER_AXIS,
        h_form: str = H_FORM_LINEAR,
        h_theta_denom: float = _DEFAULT_H_THETA_DENOM,
        l_form: str = L_FORM_SPLIT_LOG,
        l_q: float = 0.0,
        l_q_center: float = 0.03,
        l_q_out: float = 0.16,
        l_kappa: float = 1.0,
        l_stage_form: str = L_STAGE_FORM_NONE,
        l_stage_reward: float = 0.0,
        reset_box_mode: str = RESET_BOX_MODE_OURS,
        region_profile: str = REGION_PROFILE_V6,
        reset_requires_safe: bool = True,
        safe_alpha_hi: float = _SAFE_ALPHA_HI,
        safe_beta: float = _SAFE_BETA,
        train_safe_h_min: float = np.nan,
        train_safe_h_max: float = np.nan,
        train_safe_alpha_lo: float = np.nan,
        train_safe_alpha_hi: float = np.nan,
        train_safe_beta: float = np.nan,
        alpha_counts_as_invalid_dynamics: bool = True,
        beta_counts_as_invalid_dynamics: bool = True,
        clip_relaxed_alpha_beta_to_terminate: bool = False,
    ):
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0.")
        if goal_dwell_steps <= 0:
            raise ValueError("goal_dwell_steps must be > 0.")
        if not (0.0 <= init_curriculum_frac <= 1.0):
            raise ValueError("init_curriculum_frac must be in [0, 1].")
        if not (0.0 <= init_curriculum_frac_end <= 1.0):
            raise ValueError("init_curriculum_frac_end must be in [0, 1].")
        if not (0.0 <= init_curriculum_box2_frac <= 1.0):
            raise ValueError("init_curriculum_box2_frac must be in [0, 1].")
        if not (0.0 <= init_curriculum_box2_frac_end <= 1.0):
            raise ValueError("init_curriculum_box2_frac_end must be in [0, 1].")
        if init_curriculum_mode not in (
            INIT_CURRICULUM_MODE_BOX,
            INIT_CURRICULUM_MODE_BOX_MIXTURE,
            INIT_CURRICULUM_MODE_AB_BOUNDARY,
            INIT_CURRICULUM_MODE_THETA_TAIL_PROXY,
            INIT_CURRICULUM_MODE_EASY_TO_HARD,
        ):
            raise ValueError(f"Unsupported init_curriculum_mode={init_curriculum_mode!r}.")
        if init_curriculum_mode == INIT_CURRICULUM_MODE_BOX_MIXTURE:
            if init_curriculum_frac + init_curriculum_box2_frac > 1.0 + 1e-12:
                raise ValueError("init_curriculum_frac + init_curriculum_box2_frac must be <= 1.")
            if init_curriculum_frac_end + init_curriculum_box2_frac_end > 1.0 + 1e-12:
                raise ValueError("init_curriculum_frac_end + init_curriculum_box2_frac_end must be <= 1.")
        if init_curriculum_ab_kind not in ("alpha", "beta", "joint", "mixed"):
            raise ValueError(f"Unsupported init_curriculum_ab_kind={init_curriculum_ab_kind!r}.")
        if not (0.0 <= init_curriculum_ab_joint_frac <= 1.0):
            raise ValueError("init_curriculum_ab_joint_frac must be in [0, 1].")
        if safety_gap_mode not in ("literal", "nogap"):
            raise ValueError(f"Unsupported safety_gap_mode={safety_gap_mode!r}.")
        if obs_task_feats_mode not in _VALID_OBS_TASK_FEATS_MODES:
            raise ValueError(f"Unsupported obs_task_feats_mode={obs_task_feats_mode!r}.")
        if h_form not in _VALID_H_FORMS:
            raise ValueError(f"Unsupported h_form={h_form!r}.")
        if l_form not in _VALID_L_FORMS:
            raise ValueError(f"Unsupported l_form={l_form!r}.")
        if l_stage_form not in _VALID_L_STAGE_FORMS:
            raise ValueError(f"Unsupported l_stage_form={l_stage_form!r}.")
        if reset_box_mode not in _VALID_RESET_BOX_MODES:
            raise ValueError(f"Unsupported reset_box_mode={reset_box_mode!r}.")
        if region_profile not in _VALID_REGION_PROFILES:
            raise ValueError(f"Unsupported region_profile={region_profile!r}.")
        if h_theta_denom <= 0.0:
            raise ValueError("h_theta_denom must be > 0.")
        if l_q < 0.0 or l_q_center < 0.0 or l_q_out < 0.0 or l_kappa <= 0.0 or l_stage_reward < 0.0:
            raise ValueError("l_q, l_q_center, l_q_out, l_kappa, and l_stage_reward must be valid.")
        if (
            _curriculum_override_active(init_curriculum_h_min, init_curriculum_h_min_end)
            or _curriculum_override_active(init_curriculum_h_max, init_curriculum_h_max_end)
        ):
            if (
                (not np.isnan(float(init_curriculum_h_min)))
                and (not np.isnan(float(init_curriculum_h_max)))
                and (float(init_curriculum_h_min) >= float(init_curriculum_h_max))
            ):
                raise ValueError("init_curriculum_h_min must be < init_curriculum_h_max when both are specified.")
            if (
                (not np.isnan(float(init_curriculum_h_min_end)))
                and (not np.isnan(float(init_curriculum_h_max_end)))
                and (float(init_curriculum_h_min_end) > float(init_curriculum_h_max_end))
            ):
                raise ValueError(
                    "init_curriculum_h_min_end must be <= init_curriculum_h_max_end when both are specified."
                )
        if (
            _curriculum_override_active(init_curriculum_box2_h_min, init_curriculum_box2_h_min_end)
            or _curriculum_override_active(init_curriculum_box2_h_max, init_curriculum_box2_h_max_end)
        ):
            if (
                (not np.isnan(float(init_curriculum_box2_h_min)))
                and (not np.isnan(float(init_curriculum_box2_h_max)))
                and (float(init_curriculum_box2_h_min) >= float(init_curriculum_box2_h_max))
            ):
                raise ValueError(
                    "init_curriculum_box2_h_min must be < init_curriculum_box2_h_max when both are specified."
                )
            if (
                (not np.isnan(float(init_curriculum_box2_h_min_end)))
                and (not np.isnan(float(init_curriculum_box2_h_max_end)))
                and (float(init_curriculum_box2_h_min_end) > float(init_curriculum_box2_h_max_end))
            ):
                raise ValueError(
                    "init_curriculum_box2_h_min_end must be <= init_curriculum_box2_h_max_end when both are specified."
                )
        if _curriculum_override_active(init_curriculum_pe_abs_min, init_curriculum_pe_abs_min_end):
            for name, value in [
                ("init_curriculum_pe_abs_min", init_curriculum_pe_abs_min),
                ("init_curriculum_pe_abs_min_end", init_curriculum_pe_abs_min_end),
            ]:
                if (not np.isnan(float(value))) and float(value) < 0.0:
                    raise ValueError(f"{name} must be >= 0 when specified.")
        if (
            _curriculum_override_active(init_curriculum_pe_abs_min, init_curriculum_pe_abs_min_end)
            or _curriculum_override_active(init_curriculum_pe_abs_max, init_curriculum_pe_abs_max_end)
        ):
            if (
                (not np.isnan(float(init_curriculum_pe_abs_min)))
                and (not np.isnan(float(init_curriculum_pe_abs_max)))
                and (float(init_curriculum_pe_abs_min) > float(init_curriculum_pe_abs_max))
            ):
                raise ValueError(
                    "init_curriculum_pe_abs_min must be <= init_curriculum_pe_abs_max when both are specified."
                )
            if (
                (not np.isnan(float(init_curriculum_pe_abs_min_end)))
                and (not np.isnan(float(init_curriculum_pe_abs_max_end)))
                and (float(init_curriculum_pe_abs_min_end) > float(init_curriculum_pe_abs_max_end))
            ):
                raise ValueError(
                    "init_curriculum_pe_abs_min_end must be <= init_curriculum_pe_abs_max_end when both are specified."
                )
        if (
            _curriculum_override_active(init_curriculum_theta_lo, init_curriculum_theta_lo_end)
            or _curriculum_override_active(init_curriculum_theta_hi, init_curriculum_theta_hi_end)
        ):
            if (
                (not np.isnan(float(init_curriculum_theta_lo)))
                and (not np.isnan(float(init_curriculum_theta_hi)))
                and (float(init_curriculum_theta_lo) > float(init_curriculum_theta_hi))
            ):
                raise ValueError(
                    "init_curriculum_theta_lo must be <= init_curriculum_theta_hi when both are specified."
                )
            if (
                (not np.isnan(float(init_curriculum_theta_lo_end)))
                and (not np.isnan(float(init_curriculum_theta_hi_end)))
                and (float(init_curriculum_theta_lo_end) > float(init_curriculum_theta_hi_end))
            ):
                raise ValueError(
                    "init_curriculum_theta_lo_end must be <= init_curriculum_theta_hi_end when both are specified."
                )
        if (
            _curriculum_override_active(init_curriculum_box2_theta_lo, init_curriculum_box2_theta_lo_end)
            or _curriculum_override_active(init_curriculum_box2_theta_hi, init_curriculum_box2_theta_hi_end)
        ):
            if (
                (not np.isnan(float(init_curriculum_box2_theta_lo)))
                and (not np.isnan(float(init_curriculum_box2_theta_hi)))
                and (float(init_curriculum_box2_theta_lo) > float(init_curriculum_box2_theta_hi))
            ):
                raise ValueError(
                    "init_curriculum_box2_theta_lo must be <= init_curriculum_box2_theta_hi when both are specified."
                )
            if (
                (not np.isnan(float(init_curriculum_box2_theta_lo_end)))
                and (not np.isnan(float(init_curriculum_box2_theta_hi_end)))
                and (float(init_curriculum_box2_theta_lo_end) > float(init_curriculum_box2_theta_hi_end))
            ):
                raise ValueError(
                    "init_curriculum_box2_theta_lo_end must be <= init_curriculum_box2_theta_hi_end when both are specified."
                )
        if (
            _curriculum_override_active(init_curriculum_alpha_lo, init_curriculum_alpha_lo_end)
            or _curriculum_override_active(init_curriculum_alpha_hi, init_curriculum_alpha_hi_end)
        ):
            if (
                (not np.isnan(float(init_curriculum_alpha_lo)))
                and (not np.isnan(float(init_curriculum_alpha_hi)))
                and (float(init_curriculum_alpha_lo) > float(init_curriculum_alpha_hi))
            ):
                raise ValueError(
                    "init_curriculum_alpha_lo must be <= init_curriculum_alpha_hi when both are specified."
                )
            if (
                (not np.isnan(float(init_curriculum_alpha_lo_end)))
                and (not np.isnan(float(init_curriculum_alpha_hi_end)))
                and (float(init_curriculum_alpha_lo_end) > float(init_curriculum_alpha_hi_end))
            ):
                raise ValueError(
                    "init_curriculum_alpha_lo_end must be <= init_curriculum_alpha_hi_end when both are specified."
                )
        for name, lo, hi in [
            ("init_curriculum_ab_alpha_h", init_curriculum_ab_alpha_h_lo, init_curriculum_ab_alpha_h_hi),
            ("init_curriculum_ab_alpha_h_end", init_curriculum_ab_alpha_h_lo_end, init_curriculum_ab_alpha_h_hi_end),
            ("init_curriculum_ab_beta_h", init_curriculum_ab_beta_h_lo, init_curriculum_ab_beta_h_hi),
            ("init_curriculum_ab_beta_h_end", init_curriculum_ab_beta_h_lo_end, init_curriculum_ab_beta_h_hi_end),
        ]:
            if lo > hi:
                raise ValueError(f"{name}_lo must be <= {name}_hi.")
            if hi > 0.0:
                raise ValueError(f"{name}_hi must be <= 0.0 so boundary resets stay inside the safety set.")
        for name, value in [
            ("init_curriculum_beta_abs_max", init_curriculum_beta_abs_max),
            ("init_curriculum_beta_abs_max_end", init_curriculum_beta_abs_max_end),
            ("init_curriculum_p_abs_max", init_curriculum_p_abs_max),
            ("init_curriculum_p_abs_max_end", init_curriculum_p_abs_max_end),
            ("init_curriculum_q_abs_max", init_curriculum_q_abs_max),
            ("init_curriculum_q_abs_max_end", init_curriculum_q_abs_max_end),
            ("init_curriculum_r_abs_max", init_curriculum_r_abs_max),
            ("init_curriculum_r_abs_max_end", init_curriculum_r_abs_max_end),
        ]:
            if (not np.isnan(float(value))) and value < 0.0:
                raise ValueError(f"{name} must be >= 0.")

        self.task_spec_version = TASK_SPEC_VERSION_V6
        self.safety_gap_mode = str(safety_gap_mode)
        self.obs_task_feats_mode = str(obs_task_feats_mode)
        self.h_form = str(h_form)
        self.l_form = str(l_form)
        self.l_q = float(l_q)
        self.l_q_center = float(l_q_center)
        self.l_q_out = float(l_q_out)
        self.l_kappa = float(l_kappa)
        self.l_stage_form = str(l_stage_form)
        self.l_stage_reward = float(l_stage_reward)
        self.reset_box_mode = str(reset_box_mode)
        self.region_profile = str(region_profile)
        self.reset_requires_safe = bool(reset_requires_safe)
        self.safe_alpha_hi = float(safe_alpha_hi)
        self.safe_beta = float(safe_beta)
        self.train_safe_h_min_requested = float(train_safe_h_min)
        self.train_safe_h_max_requested = float(train_safe_h_max)
        self.train_safe_alpha_lo_requested = float(train_safe_alpha_lo)
        self.train_safe_alpha_hi_requested = float(train_safe_alpha_hi)
        self.train_safe_beta_requested = float(train_safe_beta)
        self.alpha_counts_as_invalid_dynamics = bool(alpha_counts_as_invalid_dynamics)
        self.beta_counts_as_invalid_dynamics = bool(beta_counts_as_invalid_dynamics)
        self.clip_relaxed_alpha_beta_to_terminate = bool(clip_relaxed_alpha_beta_to_terminate)
        self.max_episode_steps = int(max_episode_steps)
        self.goal_dwell_steps = int(goal_dwell_steps)
        self.terminate_on_crash = bool(terminate_on_crash)

        geometry = resolve_region_geometry(self.region_profile)
        self.safe_h_min = float(geometry["safe_h_min"])
        self.safe_h_max = float(geometry["safe_h_max"])
        self.safe_theta = float(geometry["safe_theta"])
        self.safe_pe = float(geometry["safe_pe"])
        self.safe_p = float(geometry["safe_p"])
        self.terminate_alpha_lo = float(_TERMINATE_ALPHA_LO)
        self.terminate_alpha_hi = float(_TERMINATE_ALPHA_HI)
        self.terminate_beta = float(_TERMINATE_BETA)
        self.terminate_theta = float(geometry["terminate_theta"])
        self.terminate_p = float(geometry["terminate_p"])
        self.terminate_h_min = float(geometry["terminate_h_min"])
        self.terminate_h_max = float(geometry["terminate_h_max"])
        self.terminate_pe_limit = float(geometry["terminate_pe_limit"])
        self.safe_h_min = _optional_float(train_safe_h_min, self.safe_h_min)
        self.safe_h_max = _optional_float(train_safe_h_max, self.safe_h_max)
        self.safe_alpha_lo = _optional_float(train_safe_alpha_lo, _SAFE_ALPHA_LO)
        self.safe_alpha_hi = _optional_float(train_safe_alpha_hi, self.safe_alpha_hi)
        self.safe_beta = _optional_float(train_safe_beta, self.safe_beta)
        if not (self.terminate_h_min < self.safe_h_min < self.safe_h_max < self.terminate_h_max):
            raise ValueError(
                "Train/eval H safety bounds must satisfy "
                "terminate_h_min < safe_h_min < safe_h_max < terminate_h_max."
            )
        if not (self.terminate_alpha_lo < self.safe_alpha_lo < self.safe_alpha_hi < self.terminate_alpha_hi):
            raise ValueError(
                "Train/eval alpha safety bounds must satisfy "
                "terminate_alpha_lo < safe_alpha_lo < safe_alpha_hi < terminate_alpha_hi."
            )
        if not (0.0 < self.safe_beta < self.terminate_beta):
            raise ValueError("Train/eval beta safety bound must satisfy 0 < safe_beta < terminate_beta.")
        self.h_theta_denom = resolve_h_theta_denom(self.region_profile, h_theta_denom)

        self.dt = 0.05
        self.state_enc_dim = int(_OBS_MEAN.shape[0])
        self.task_feat_labels = (
            _TASK_FEAT_LABELS_AGGREGATE
            if self.obs_task_feats_mode == OBS_TASK_FEATS_MODE_AGGREGATE
            else _TASK_FEAT_LABELS_PER_AXIS
        )
        self.task_feat_dim = len(self.task_feat_labels)
        self.obs_dim = self.state_enc_dim + self.task_feat_dim
        self.action_dim = NU
        self.goal_h_min = _GOAL_H_MIN
        self.goal_h_max = _GOAL_H_MAX
        self.goal_h_mid = _GOAL_H_MID
        self.goal_h_halfwidth = _GOAL_H_HALFWIDTH

        self.init_curriculum = bool(init_curriculum)
        self.init_curriculum_mode = str(init_curriculum_mode)
        self.init_curriculum_frac = float(init_curriculum_frac)
        self.init_curriculum_frac_end = float(init_curriculum_frac_end)
        self.init_curriculum_box2_frac = float(init_curriculum_box2_frac)
        self.init_curriculum_box2_frac_end = float(init_curriculum_box2_frac_end)
        self.init_curriculum_anneal_start = int(init_curriculum_anneal_start)
        self.init_curriculum_anneal_end = int(init_curriculum_anneal_end)
        self.init_curriculum_h_min = float(init_curriculum_h_min)
        self.init_curriculum_h_max = float(init_curriculum_h_max)
        self.init_curriculum_h_min_end = float(init_curriculum_h_min_end)
        self.init_curriculum_h_max_end = float(init_curriculum_h_max_end)
        self.init_curriculum_box2_h_min = float(init_curriculum_box2_h_min)
        self.init_curriculum_box2_h_max = float(init_curriculum_box2_h_max)
        self.init_curriculum_box2_h_min_end = float(init_curriculum_box2_h_min_end)
        self.init_curriculum_box2_h_max_end = float(init_curriculum_box2_h_max_end)
        self.init_curriculum_pe_abs_min = float(init_curriculum_pe_abs_min)
        self.init_curriculum_pe_abs_max = float(init_curriculum_pe_abs_max)
        self.init_curriculum_pe_abs_min_end = float(init_curriculum_pe_abs_min_end)
        self.init_curriculum_pe_abs_max_end = float(init_curriculum_pe_abs_max_end)
        self.init_curriculum_theta_lo = float(init_curriculum_theta_lo)
        self.init_curriculum_theta_hi = float(init_curriculum_theta_hi)
        self.init_curriculum_theta_lo_end = float(init_curriculum_theta_lo_end)
        self.init_curriculum_theta_hi_end = float(init_curriculum_theta_hi_end)
        self.init_curriculum_box2_theta_lo = float(init_curriculum_box2_theta_lo)
        self.init_curriculum_box2_theta_hi = float(init_curriculum_box2_theta_hi)
        self.init_curriculum_box2_theta_lo_end = float(init_curriculum_box2_theta_lo_end)
        self.init_curriculum_box2_theta_hi_end = float(init_curriculum_box2_theta_hi_end)
        self.init_curriculum_alpha_lo = float(init_curriculum_alpha_lo)
        self.init_curriculum_alpha_hi = float(init_curriculum_alpha_hi)
        self.init_curriculum_alpha_lo_end = float(init_curriculum_alpha_lo_end)
        self.init_curriculum_alpha_hi_end = float(init_curriculum_alpha_hi_end)
        self.init_curriculum_beta_abs_max = float(init_curriculum_beta_abs_max)
        self.init_curriculum_beta_abs_max_end = float(init_curriculum_beta_abs_max_end)
        self.init_curriculum_ab_kind = str(init_curriculum_ab_kind)
        self.init_curriculum_ab_joint_frac = float(init_curriculum_ab_joint_frac)
        self.init_curriculum_ab_alpha_h_lo = float(init_curriculum_ab_alpha_h_lo)
        self.init_curriculum_ab_alpha_h_hi = float(init_curriculum_ab_alpha_h_hi)
        self.init_curriculum_ab_alpha_h_lo_end = float(init_curriculum_ab_alpha_h_lo_end)
        self.init_curriculum_ab_alpha_h_hi_end = float(init_curriculum_ab_alpha_h_hi_end)
        self.init_curriculum_ab_beta_h_lo = float(init_curriculum_ab_beta_h_lo)
        self.init_curriculum_ab_beta_h_hi = float(init_curriculum_ab_beta_h_hi)
        self.init_curriculum_ab_beta_h_lo_end = float(init_curriculum_ab_beta_h_lo_end)
        self.init_curriculum_ab_beta_h_hi_end = float(init_curriculum_ab_beta_h_hi_end)
        self.init_curriculum_p_abs_max = float(init_curriculum_p_abs_max)
        self.init_curriculum_p_abs_max_end = float(init_curriculum_p_abs_max_end)
        self.init_curriculum_q_abs_max = float(init_curriculum_q_abs_max)
        self.init_curriculum_q_abs_max_end = float(init_curriculum_q_abs_max_end)
        self.init_curriculum_r_abs_max = float(init_curriculum_r_abs_max)
        self.init_curriculum_r_abs_max_end = float(init_curriculum_r_abs_max_end)
        self.init_curriculum_allow_in_goal = bool(init_curriculum_allow_in_goal)
        self.init_curriculum_require_train_safe = bool(init_curriculum_require_train_safe)

        self._rng = np.random.RandomState(seed)
        self._f16 = F16()
        self._eval_mode = False
        self._train_step = 0
        self._last_paper_sampler_metadata = SamplerMetadata(tries=0, rejects=0)

        self.control_low = np.array([-10.0, -10.0, -10.0, 0.0], dtype=np.float32)
        self.control_high = np.array([15.0, 10.0, 10.0, 1.0], dtype=np.float32)

        self.observation_space = gym_spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym_spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self._x = None
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._step_count = 0

        self._ep_reward = 0.0
        self._ep_task_cost = 0.0
        self._ep_unsafe_steps = 0.0
        self._ep_length = 0
        self._ep_goal_steps = 0
        self._ep_goal_streak = 0
        self._ep_max_goal_streak = 0
        self._ep_success = False
        self._ep_success_step = None
        self._ep_terminated = False
        self._ep_crash_step = None
        self._ep_crash_cause = None
        self._ep_terminate_reason = None
        self._ep_final_goal_distance = np.inf
        self._ep_min_altitude = np.inf
        self._ep_peak_h = -np.inf
        self._ep_peak_h_components = {label: -np.inf for label in _TASK_H_LABELS}
        self._ep_max_abs_beta = 0.0
        self._ep_max_abs_theta = 0.0
        self._ep_safe_steps = 0
        self._ep_safe_in_goal_steps = 0
        self._ep_init_goal = 0.0
        self._ep_init_safe = 0.0
        self._ep_init_h = -np.inf
        self._ep_init_h_alpha = -np.inf
        self._ep_init_h_beta = -np.inf
        self._ep_init_alpha = 0.0
        self._ep_init_abs_beta = 0.0
        self._ep_init_pe_abs = 0.0

        self._last_reset_curriculum_target_frac = 0.0
        self._last_reset_curriculum_box1_frac = 0.0
        self._last_reset_curriculum_box2_frac = 0.0
        self._last_reset_curriculum_branch = 0.0
        self._last_reset_curriculum_active = 0.0
        self._last_reset_curriculum_fallback = 0.0
        self._last_reset_curriculum_reject_rate = 0.0
        self.episode_info = {}
        self._v5_sanity_check()

    # ------------------------------------------------------------------ mode
    def set_eval_mode(self):
        self._eval_mode = True

    def set_train_mode(self):
        self._eval_mode = False

    def set_train_step(self, step: int):
        self._train_step = max(int(step), 0)

    # ------------------------------------------------------------- task math
    def _task_l_value(self, x: np.ndarray) -> float:
        z = (float(x[IDX_H]) - self.goal_h_mid) / self.goal_h_halfwidth
        d_out = max(abs(z) - 1.0, 0.0)
        l_center = float(self.l_q_center * min(z**2, 1.0))
        if self.l_form == L_FORM_SPLIT_LINEAR:
            phi = d_out
        elif self.l_form == L_FORM_SPLIT_LOG:
            phi = np.log1p(self.l_kappa * d_out) / self.l_kappa
        elif self.l_form == L_FORM_SPLIT_ATAN:
            phi = np.arctan(self.l_kappa * d_out) / self.l_kappa
        else:
            raise ValueError(f"Unsupported v5 l_form={self.l_form!r}.")
        l_base = float(l_center + self.l_q_out * phi)

        if self.l_stage_form == L_STAGE_FORM_BAND_BOWL:
            stage_bonus = self.l_stage_reward * max(1.0 - z**2, 0.0)
            return float(l_base - stage_bonus)
        return l_base

    def _task_goal_distance(self, x: np.ndarray) -> float:
        return max(abs(float(x[IDX_H]) - self.goal_h_mid) - self.goal_h_halfwidth, 0.0) / 250.0

    def _task_goal_bool(self, x: np.ndarray) -> bool:
        return self._task_goal_distance(x) <= 0.0

    def _task_h_components_raw(self, x: np.ndarray) -> np.ndarray:
        if self.h_form != H_FORM_LINEAR:
            raise ValueError(f"Unsupported v5 h_form={self.h_form!r}; v5 is linear-only.")
        h_altitude = max(
            (self.safe_h_min - float(x[IDX_H])) / (self.safe_h_min - self.terminate_h_min),
            (float(x[IDX_H]) - self.safe_h_max) / (self.terminate_h_max - self.safe_h_max),
        )
        h_alpha = max(
            (self.safe_alpha_lo - float(x[IDX_ALPHA])) / (self.safe_alpha_lo - self.terminate_alpha_lo),
            (float(x[IDX_ALPHA]) - self.safe_alpha_hi) / (self.terminate_alpha_hi - self.safe_alpha_hi),
        )
        h_beta = (abs(float(x[IDX_BETA])) - self.safe_beta) / (self.terminate_beta - self.safe_beta)
        h_theta = (abs(float(x[IDX_THETA])) - self.safe_theta) / self.h_theta_denom
        h_pe = (abs(float(x[IDX_PE])) - self.safe_pe) / (self.terminate_pe_limit - self.safe_pe)
        h_p = (abs(float(x[IDX_P])) - self.safe_p) / (self.terminate_p - self.safe_p)
        return np.array([h_altitude, h_alpha, h_beta, h_theta, h_pe, h_p], dtype=np.float64)

    def _task_h_components(self, x: np.ndarray) -> np.ndarray:
        h_raw = self._task_h_components_raw(x)
        if self.safety_gap_mode == "nogap":
            return h_raw
        return np.where(h_raw >= 0.0, h_raw + 0.5, h_raw - 0.5)

    def _task_safety_margin(self, x: np.ndarray) -> float:
        return float(np.max(self._task_h_components(x)))

    def _task_safety_bool(self, x: np.ndarray) -> bool:
        return self._task_safety_margin(x) <= 0.0

    def _task_obs_features(self, x: np.ndarray) -> np.ndarray:
        h_components = self._task_h_components(x)
        h_margin = float(np.max(h_components))
        goal_distance = self._task_goal_distance(x)
        if self.obs_task_feats_mode == OBS_TASK_FEATS_MODE_AGGREGATE:
            return np.asarray(
                [
                    np.tanh(h_margin),
                    np.tanh(goal_distance),
                ],
                dtype=np.float32,
            )
        return np.asarray(
            [
                np.tanh(h_margin),
                np.tanh(goal_distance),
                *np.tanh(h_components),
            ],
            dtype=np.float32,
        )

    def _task_reset_box(self) -> np.ndarray:
        # Main box-based reset entrypoint. Training resets and paper-style eval
        # both follow the active reset_box_mode through this path.
        return self._task_train_reset_box()

    def _task_train_reset_box(self) -> np.ndarray:
        if self.reset_box_mode == RESET_BOX_MODE_OURS:
            return _BASE_BOX_V5.copy()
        if self.reset_box_mode == RESET_BOX_MODE_EFPPO_TRAIN:
            return _BASE_BOX_EFPPO_TRAIN.copy()
        raise ValueError(f"Unsupported reset_box_mode={self.reset_box_mode!r}.")


    def _curriculum_anneal_alpha(self) -> float:
        start = self.init_curriculum_anneal_start
        end = self.init_curriculum_anneal_end
        if end <= start:
            return 1.0 if self._train_step >= end else 0.0
        if self._train_step <= start:
            return 0.0
        if self._train_step >= end:
            return 1.0
        return float((self._train_step - start) / float(end - start))

    def _interp_curriculum(self, start_value: float, end_value: float) -> float:
        alpha = self._curriculum_anneal_alpha()
        return float((1.0 - alpha) * start_value + alpha * end_value)

    def _interp_optional_curriculum(self, start_value: float, end_value: float, base_value: float) -> float | None:
        bounds = _resolve_optional_curriculum_pair(start_value, end_value, base_value)
        if bounds is None:
            return None
        lo, hi = bounds
        return self._interp_curriculum(lo, hi)

    # --------------------------------------------------------- state checks
    def _validity_flags_v5(self, x: np.ndarray) -> Dict[str, bool]:
        finite_state = bool(np.all(np.isfinite(x)))
        alpha_invalid = not (self.terminate_alpha_lo <= float(x[IDX_ALPHA]) <= self.terminate_alpha_hi)
        beta_invalid = abs(float(x[IDX_BETA])) > self.terminate_beta
        theta_invalid = abs(float(x[IDX_THETA])) >= self.terminate_theta
        p_invalid = abs(float(x[IDX_P])) >= self.terminate_p
        h_invalid = not np.isfinite(float(x[IDX_H]))
        pe_invalid = not np.isfinite(float(x[IDX_PE]))
        invalid_dynamics = bool(
            (not finite_state)
            or (self.alpha_counts_as_invalid_dynamics and alpha_invalid)
            or (self.beta_counts_as_invalid_dynamics and beta_invalid)
            or theta_invalid
            or p_invalid
            or h_invalid
            or pe_invalid
        )
        return {
            "finite_state": finite_state,
            "alpha_invalid": bool(alpha_invalid),
            "beta_invalid": bool(beta_invalid),
            "theta_invalid": bool(theta_invalid),
            "p_invalid": bool(p_invalid),
            "h_invalid": bool(h_invalid),
            "pe_invalid": bool(pe_invalid),
            "invalid_dynamics": invalid_dynamics,
        }


    def _clip_relaxed_alpha_beta_state_v5(self, x: np.ndarray, validity_flags: Dict[str, bool]) -> np.ndarray:
        if not self.clip_relaxed_alpha_beta_to_terminate:
            return x
        clipped = np.array(x, dtype=np.float64, copy=True)
        if (not self.alpha_counts_as_invalid_dynamics) and bool(validity_flags["alpha_invalid"]):
            clipped[IDX_ALPHA] = float(np.clip(clipped[IDX_ALPHA], self.terminate_alpha_lo, self.terminate_alpha_hi))
        if (not self.beta_counts_as_invalid_dynamics) and bool(validity_flags["beta_invalid"]):
            clipped[IDX_BETA] = float(np.clip(clipped[IDX_BETA], -self.terminate_beta, self.terminate_beta))
        return clipped

    def _hits_hard_terminal_v5(self, x: np.ndarray) -> bool:
        return (
            float(x[IDX_H]) <= self.terminate_h_min
            or float(x[IDX_H]) >= self.terminate_h_max
            or abs(float(x[IDX_PE])) >= self.terminate_pe_limit
        )

    def _classify_crash_cause_v5(self, x_raw: np.ndarray) -> str:
        if float(x_raw[IDX_ALPHA]) <= self.terminate_alpha_lo or float(x_raw[IDX_ALPHA]) >= self.terminate_alpha_hi:
            return "alpha"
        if abs(float(x_raw[IDX_P])) >= self.terminate_p:
            return "p"
        if float(x_raw[IDX_H]) <= self.terminate_h_min or float(x_raw[IDX_H]) >= self.terminate_h_max:
            return "alt"
        if abs(float(x_raw[IDX_PE])) >= self.terminate_pe_limit:
            return "pe"
        if abs(float(x_raw[IDX_BETA])) >= self.terminate_beta:
            return "beta"
        if abs(float(x_raw[IDX_THETA])) >= self.terminate_theta:
            return "theta"
        return "other"

    def _termination_reason_from_flags_v5(self, invalid_dynamics: bool, hard_terminal: bool) -> str:
        if invalid_dynamics and hard_terminal:
            return "invalid_and_hard"
        if hard_terminal:
            return "hard_terminal"
        if invalid_dynamics:
            return "invalid_dynamics"
        return "none"

    # --------------------------------------------------------------- samplers
    def _current_curriculum_frac_value(self, frac0: float, frac1: float) -> float:
        start = self.init_curriculum_anneal_start
        end = self.init_curriculum_anneal_end
        if end <= start:
            return frac1 if self._train_step >= end else frac0
        if self._train_step <= start:
            return frac0
        if self._train_step >= end:
            return frac1
        alpha = (self._train_step - start) / float(end - start)
        return float((1.0 - alpha) * frac0 + alpha * frac1)

    def current_init_curriculum_frac(self) -> float:
        if (not self.init_curriculum) or self._eval_mode:
            return 0.0
        return self._current_curriculum_frac_value(self.init_curriculum_frac, self.init_curriculum_frac_end)

    def current_init_curriculum_box2_frac(self) -> float:
        if (
            (not self.init_curriculum)
            or self._eval_mode
            or self.init_curriculum_mode != INIT_CURRICULUM_MODE_BOX_MIXTURE
        ):
            return 0.0
        return self._current_curriculum_frac_value(
            self.init_curriculum_box2_frac,
            self.init_curriculum_box2_frac_end,
        )

    def _apply_reset_overrides(self, x: np.ndarray, options: Dict | None):
        if not options:
            return
        for key, idx in _RESET_OVERRIDE_KEYS.items():
            if key in options and options[key] is not None:
                x[idx] = float(options[key])

    def _sample_upper_alpha_from_h(self, h_lo: float, h_hi: float) -> float:
        h_val = self._rng.uniform(h_lo, h_hi)
        alpha = self.safe_alpha_hi + h_val * (self.terminate_alpha_hi - self.safe_alpha_hi)
        return float(np.clip(alpha, self.safe_alpha_lo, self.safe_alpha_hi))

    def _sample_abs_beta_from_h(self, h_lo: float, h_hi: float) -> float:
        h_val = self._rng.uniform(h_lo, h_hi)
        beta_abs = self.safe_beta + h_val * (self.terminate_beta - self.safe_beta)
        return float(np.clip(beta_abs, 0.0, self.safe_beta))

    def _draw_ab_boundary_kind(self) -> str:
        if self.init_curriculum_ab_kind != "mixed":
            return self.init_curriculum_ab_kind
        if self._rng.rand() < self.init_curriculum_ab_joint_frac:
            return "joint"
        return "alpha" if self._rng.rand() < 0.5 else "beta"

    def _apply_ab_boundary_curriculum(self, x: np.ndarray):
        alpha_h_lo = self._interp_curriculum(
            self.init_curriculum_ab_alpha_h_lo, self.init_curriculum_ab_alpha_h_lo_end
        )
        alpha_h_hi = self._interp_curriculum(
            self.init_curriculum_ab_alpha_h_hi, self.init_curriculum_ab_alpha_h_hi_end
        )
        beta_h_lo = self._interp_curriculum(
            self.init_curriculum_ab_beta_h_lo, self.init_curriculum_ab_beta_h_lo_end
        )
        beta_h_hi = self._interp_curriculum(
            self.init_curriculum_ab_beta_h_hi, self.init_curriculum_ab_beta_h_hi_end
        )
        ab_kind = self._draw_ab_boundary_kind()
        if ab_kind in ("alpha", "joint"):
            x[IDX_ALPHA] = self._sample_upper_alpha_from_h(alpha_h_lo, alpha_h_hi)
        if ab_kind in ("beta", "joint"):
            beta_abs = self._sample_abs_beta_from_h(beta_h_lo, beta_h_hi)
            x[IDX_BETA] = beta_abs if self._rng.rand() < 0.5 else -beta_abs

    def _sample_theta_tail_proxy_state_v5(self) -> np.ndarray:
        """Fixed-theta/height proxy reset sampler.

        This branch starts from the diagnostic-grid nominal state and only
        widens theta/H, with a tiny alpha band to avoid overfitting to a
        single trim point. The curriculum fraction anneal controls how much of
        training uses this proxy; the proxy geometry itself stays fixed.
        """

        x = self.nominal_state_v5().copy()
        if self._rng.rand() < 0.80:
            x[IDX_THETA] = self._rng.uniform(0.4, 1.0)
        else:
            x[IDX_THETA] = self._rng.uniform(-1.2, -0.4)
        x[IDX_H] = self._rng.uniform(self.safe_h_min, self.safe_h_max)
        x[IDX_ALPHA] = self._rng.uniform(-0.02, 0.05)
        return x

    def _sample_easy_to_hard_state_v5(self) -> np.ndarray:
        """Sample from a direct easy-to-hard training curriculum.

        At alpha=0 this starts near a benign trim/goal-approach slice with
        small attitude, rate, and PE variation. At alpha=1 it has widened to
        the active training reset box. The usual safety check is still applied
        by the caller, but early curriculum samples are designed to sit well
        inside the safe set instead of relying on rejection to find them.
        """

        base = self._task_reset_box()
        easy = base.copy()

        def set_interval(idx: int, lo: float, hi: float) -> None:
            lo_eff = max(float(lo), float(base[idx, 0]))
            hi_eff = min(float(hi), float(base[idx, 1]))
            if lo_eff > hi_eff:
                mid = float(np.clip(0.5 * (float(lo) + float(hi)), base[idx, 0], base[idx, 1]))
                lo_eff = hi_eff = mid
            easy[idx, 0] = lo_eff
            easy[idx, 1] = hi_eff

        safe_h_span = max(float(self.safe_h_max - self.safe_h_min), 1.0)
        easy_h_lo = max(float(self.goal_h_max + 20.0), float(self.safe_h_min + 0.12 * safe_h_span))
        easy_h_hi = min(float(self.safe_h_min + 0.35 * safe_h_span), float(self.safe_h_max - 0.05 * safe_h_span))
        if easy_h_hi <= easy_h_lo:
            easy_h_lo = float(self.safe_h_min + 0.20 * safe_h_span)
            easy_h_hi = float(self.safe_h_min + 0.45 * safe_h_span)

        set_interval(IDX_VT, 450.0, 540.0)
        set_interval(IDX_ALPHA, -0.02, 0.07)
        set_interval(IDX_BETA, -0.03, 0.03)
        set_interval(IDX_PHI, -0.20, 0.20)
        set_interval(IDX_THETA, -0.15, 0.15)
        set_interval(IDX_PSI, -1.0e-4, 1.0e-4)
        set_interval(IDX_P, -0.10, 0.10)
        set_interval(IDX_Q, -0.10, 0.10)
        set_interval(IDX_R, -0.30, 0.30)
        set_interval(IDX_PN, -100.0, 100.0)
        set_interval(IDX_PE, -20.0, 20.0)
        set_interval(IDX_H, easy_h_lo, easy_h_hi)
        set_interval(IDX_POW, 6.0, 9.0)
        set_interval(IDX_NZINT, -0.20, 0.20)
        set_interval(IDX_PSINT, -0.20, 0.20)
        set_interval(IDX_NYRINT, -0.20, 0.20)

        alpha = self._curriculum_anneal_alpha()
        lo = (1.0 - alpha) * easy[:, 0] + alpha * base[:, 0]
        hi = (1.0 - alpha) * easy[:, 1] + alpha * base[:, 1]
        return self._rng.uniform(lo, hi).astype(np.float64)

    def _reject_initial_state_v5(self, x: np.ndarray) -> bool:
        if not np.all(np.isfinite(x)):
            return True
        if self.reset_requires_safe and (self._task_safety_margin(x) >= 0.0):
            return True
        return False

    def _sample_from_box_v5(
        self,
        box: np.ndarray,
        rng_np: np.random.RandomState,
        reject_fn,
        num_samples: int = 1,
        max_tries: int = 10000,
        options: Dict | None = None,
    ) -> Tuple[List[np.ndarray], SamplerMetadata]:
        lo, hi = box[:, 0], box[:, 1]
        samples: List[np.ndarray] = []
        tries = 0
        rejects = 0
        while len(samples) < num_samples and tries < max_tries:
            tries += 1
            x = rng_np.uniform(lo, hi).astype(np.float64)
            self._apply_reset_overrides(x, options)
            if reject_fn(x):
                rejects += 1
                continue
            samples.append(x)
        if len(samples) < num_samples:
            raise RuntimeError("Failed to sample valid F16 v5 initial states within max_tries.")
        return samples, SamplerMetadata(tries=tries, rejects=rejects)

    def _sample_curriculum_state_v5(
        self,
        options: Dict | None = None,
        curriculum_box: int = 1,
    ) -> Tuple[np.ndarray, SamplerMetadata, bool]:
        if curriculum_box not in (1, 2):
            raise ValueError(f"Unsupported curriculum_box={curriculum_box!r}.")
        box = self._task_reset_box()
        rejects = 0
        tries = 0
        while tries < 10000:
            tries += 1
            if self.init_curriculum_mode == INIT_CURRICULUM_MODE_THETA_TAIL_PROXY and curriculum_box == 1:
                x = self._sample_theta_tail_proxy_state_v5()
            elif self.init_curriculum_mode == INIT_CURRICULUM_MODE_EASY_TO_HARD and curriculum_box == 1:
                x = self._sample_easy_to_hard_state_v5()
            else:
                x = self._rng.uniform(box[:, 0], box[:, 1]).astype(np.float64)
                h_min = self.init_curriculum_h_min
                h_min_end = self.init_curriculum_h_min_end
                h_max = self.init_curriculum_h_max
                h_max_end = self.init_curriculum_h_max_end
                theta_lo_start = self.init_curriculum_theta_lo
                theta_lo_end = self.init_curriculum_theta_lo_end
                theta_hi_start = self.init_curriculum_theta_hi
                theta_hi_end = self.init_curriculum_theta_hi_end
                if curriculum_box == 2:
                    h_min = self.init_curriculum_box2_h_min
                    h_min_end = self.init_curriculum_box2_h_min_end
                    h_max = self.init_curriculum_box2_h_max
                    h_max_end = self.init_curriculum_box2_h_max_end
                    theta_lo_start = self.init_curriculum_box2_theta_lo
                    theta_lo_end = self.init_curriculum_box2_theta_lo_end
                    theta_hi_start = self.init_curriculum_box2_theta_hi
                    theta_hi_end = self.init_curriculum_box2_theta_hi_end
                h_lo = self._interp_optional_curriculum(
                    h_min,
                    h_min_end,
                    box[IDX_H, 0],
                )
                h_hi = self._interp_optional_curriculum(
                    h_max,
                    h_max_end,
                    box[IDX_H, 1],
                )
                if (h_lo is not None) or (h_hi is not None):
                    if h_lo is None:
                        h_lo = float(box[IDX_H, 0])
                    if h_hi is None:
                        h_hi = float(box[IDX_H, 1])
                    x[IDX_H] = self._rng.uniform(h_lo, h_hi)

                pe_abs_min = self._interp_optional_curriculum(
                    self.init_curriculum_pe_abs_min,
                    self.init_curriculum_pe_abs_min_end,
                    0.0,
                )
                pe_abs_max = self._interp_optional_curriculum(
                    self.init_curriculum_pe_abs_max,
                    self.init_curriculum_pe_abs_max_end,
                    max(abs(float(box[IDX_PE, 0])), abs(float(box[IDX_PE, 1]))),
                )
                if (pe_abs_min is not None) or (pe_abs_max is not None):
                    if pe_abs_min is None:
                        pe_abs_min = 0.0
                    if pe_abs_max is None:
                        pe_abs_max = max(abs(float(box[IDX_PE, 0])), abs(float(box[IDX_PE, 1])))
                    pe_abs = self._rng.uniform(pe_abs_min, pe_abs_max)
                    x[IDX_PE] = pe_abs if self._rng.rand() < 0.5 else -pe_abs

                theta_lo = self._interp_optional_curriculum(
                    theta_lo_start,
                    theta_lo_end,
                    box[IDX_THETA, 0],
                )
                theta_hi = self._interp_optional_curriculum(
                    theta_hi_start,
                    theta_hi_end,
                    box[IDX_THETA, 1],
                )
                if (theta_lo is not None) or (theta_hi is not None):
                    if theta_lo is None:
                        theta_lo = float(box[IDX_THETA, 0])
                    if theta_hi is None:
                        theta_hi = float(box[IDX_THETA, 1])
                    x[IDX_THETA] = self._rng.uniform(theta_lo, theta_hi)

                if self.init_curriculum_mode == INIT_CURRICULUM_MODE_AB_BOUNDARY:
                    self._apply_ab_boundary_curriculum(x)
                else:
                    alpha_lo = self._interp_optional_curriculum(
                        self.init_curriculum_alpha_lo,
                        self.init_curriculum_alpha_lo_end,
                        box[IDX_ALPHA, 0],
                    )
                    alpha_hi = self._interp_optional_curriculum(
                        self.init_curriculum_alpha_hi,
                        self.init_curriculum_alpha_hi_end,
                        box[IDX_ALPHA, 1],
                    )
                    if (alpha_lo is not None) or (alpha_hi is not None):
                        if alpha_lo is None:
                            alpha_lo = float(box[IDX_ALPHA, 0])
                        if alpha_hi is None:
                            alpha_hi = float(box[IDX_ALPHA, 1])
                        x[IDX_ALPHA] = self._rng.uniform(alpha_lo, alpha_hi)

                    beta_abs_max = self._interp_optional_curriculum(
                        self.init_curriculum_beta_abs_max,
                        self.init_curriculum_beta_abs_max_end,
                        max(abs(float(box[IDX_BETA, 0])), abs(float(box[IDX_BETA, 1]))),
                    )
                    if beta_abs_max is not None:
                        x[IDX_BETA] = self._rng.uniform(-beta_abs_max, beta_abs_max)

                p_abs_max = self._interp_optional_curriculum(
                    self.init_curriculum_p_abs_max,
                    self.init_curriculum_p_abs_max_end,
                    max(abs(float(box[IDX_P, 0])), abs(float(box[IDX_P, 1]))),
                )
                if p_abs_max is not None:
                    x[IDX_P] = self._rng.uniform(-p_abs_max, p_abs_max)

                q_abs_max = self._interp_optional_curriculum(
                    self.init_curriculum_q_abs_max,
                    self.init_curriculum_q_abs_max_end,
                    max(abs(float(box[IDX_Q, 0])), abs(float(box[IDX_Q, 1]))),
                )
                if q_abs_max is not None:
                    x[IDX_Q] = self._rng.uniform(-q_abs_max, q_abs_max)

                r_abs_max = self._interp_optional_curriculum(
                    self.init_curriculum_r_abs_max,
                    self.init_curriculum_r_abs_max_end,
                    max(abs(float(box[IDX_R, 0])), abs(float(box[IDX_R, 1]))),
                )
                if r_abs_max is not None:
                    x[IDX_R] = self._rng.uniform(-r_abs_max, r_abs_max)
            self._apply_reset_overrides(x, options)
            if self._reject_initial_state_v5(x):
                rejects += 1
                continue
            if (not self.init_curriculum_allow_in_goal) and self._task_goal_bool(x):
                rejects += 1
                continue
            if self.init_curriculum_require_train_safe and (not self._task_safety_bool(x)):
                rejects += 1
                continue
            return x, SamplerMetadata(tries=tries, rejects=rejects), False
        fallback_samples, meta = self._sample_from_box_v5(
            self._task_reset_box(),
            self._rng,
            self._reject_initial_state_v5,
            num_samples=1,
            options=options,
        )
        return fallback_samples[0], SamplerMetadata(tries=tries + meta.tries, rejects=rejects + meta.rejects), True

    def _sample_train_state_v5(self, options: Dict | None = None) -> np.ndarray:
        target_frac1 = self.current_init_curriculum_frac()
        target_frac2 = self.current_init_curriculum_box2_frac()
        target_frac = min(1.0, target_frac1 + target_frac2)
        curriculum_eligible = bool(
            self.init_curriculum
            and (not self._eval_mode)
            and not any(
                key in (options or {}) and (options or {})[key] is not None
                for key in _RESET_OVERRIDE_KEYS
            )
        )
        curriculum_box = 0
        if curriculum_eligible:
            u = self._rng.rand()
            if self.init_curriculum_mode == INIT_CURRICULUM_MODE_BOX_MIXTURE:
                if u < target_frac1:
                    curriculum_box = 1
                elif u < target_frac1 + target_frac2:
                    curriculum_box = 2
            elif u < target_frac1:
                curriculum_box = 1
        use_curriculum = curriculum_box in (1, 2)

        self._last_reset_curriculum_target_frac = float(target_frac)
        self._last_reset_curriculum_box1_frac = float(target_frac1)
        self._last_reset_curriculum_box2_frac = float(target_frac2)
        self._last_reset_curriculum_branch = float(curriculum_box)
        self._last_reset_curriculum_active = 0.0
        self._last_reset_curriculum_fallback = 0.0
        self._last_reset_curriculum_reject_rate = 0.0

        if use_curriculum:
            x, meta, fallback = self._sample_curriculum_state_v5(options, curriculum_box=curriculum_box)
            self._last_reset_curriculum_active = float(not fallback)
            self._last_reset_curriculum_fallback = float(fallback)
            self._last_reset_curriculum_reject_rate = meta.reject_rate
        else:
            samples, meta = self._sample_from_box_v5(
                self._task_reset_box(),
                self._rng,
                self._reject_initial_state_v5,
                num_samples=1,
                options=options,
            )
            x = samples[0]
            self._last_reset_curriculum_reject_rate = meta.reject_rate

        self._ep_init_goal = float(self._task_goal_bool(x))
        self._ep_init_safe = float(self._task_safety_bool(x))
        self._ep_init_pe_abs = float(abs(x[IDX_PE]))
        self._ep_init_h = float(self._task_safety_margin(x))
        h_raw = self._task_h_components_raw(x)
        self._ep_init_h_alpha = float(h_raw[1])
        self._ep_init_h_beta = float(h_raw[2])
        self._ep_init_alpha = float(x[IDX_ALPHA])
        self._ep_init_abs_beta = float(abs(x[IDX_BETA]))
        return x


    def nominal_state_v5(self) -> np.ndarray:
        return np.array(
            [
                5.02089669e02,
                2.61709746e-02,
                0.0,
                0.0,
                2.61709746e-02,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                500.0,
                7.60335468e00,
                2.43004861e-02,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )


    # ---------------------------------------------------------------- reset
    def reset(self, options: Dict | None = None):
        options = dict(options or {})
        self._x = self._sample_train_state_v5(options)
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._step_count = 0

        self._ep_reward = 0.0
        self._ep_task_cost = 0.0
        self._ep_unsafe_steps = 0.0
        self._ep_length = 0
        self._ep_goal_steps = 0
        self._ep_goal_streak = 0
        self._ep_max_goal_streak = 0
        self._ep_success = False
        self._ep_success_step = None
        self._ep_terminated = False
        self._ep_crash_step = None
        self._ep_crash_cause = None
        self._ep_terminate_reason = None
        self._ep_final_goal_distance = self._task_goal_distance(self._x)
        self._ep_min_altitude = float(self._x[IDX_H])
        self._ep_peak_h = self._task_safety_margin(self._x)
        self._ep_peak_h_components = {
            label: value for label, value in zip(_TASK_H_LABELS, self._task_h_components(self._x))
        }
        self._ep_max_abs_beta = abs(float(self._x[IDX_BETA]))
        self._ep_max_abs_theta = abs(float(self._x[IDX_THETA]))
        self._ep_safe_steps = 0
        self._ep_safe_in_goal_steps = 0
        self.episode_info = {}
        return self._get_obs(self._x)

    # ----------------------------------------------------------------- step
    def decode_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float32)
        action_clipped = np.clip(action, -1.0, 1.0)
        control = 0.5 * (self.control_high - self.control_low) * action_clipped
        control = control + 0.5 * (self.control_high + self.control_low)
        return action_clipped.astype(np.float32), np.asarray(control, dtype=np.float32)

    def simulate_transition(self, state: np.ndarray, action: np.ndarray) -> Dict[str, object]:
        action_clipped, control = self.decode_action(action)
        x = jnp.array(state)
        u = jnp.array(control)
        k1 = self._f16.xdot(x, u)
        k2 = self._f16.xdot(x + 0.5 * self.dt * k1, u)
        k3 = self._f16.xdot(x + 0.5 * self.dt * k2, u)
        k4 = self._f16.xdot(x + self.dt * k3, u)
        x_raw = x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_raw_np = np.asarray(x_raw, dtype=np.float64)
        validity_flags = self._validity_flags_v5(x_raw_np)
        x_next_np = self._clip_relaxed_alpha_beta_state_v5(x_raw_np, validity_flags)

        h_components = self._task_h_components(x_next_np)
        h_margin = float(np.max(h_components))
        task_cost = self._task_l_value(x_next_np)
        goal_distance = self._task_goal_distance(x_next_np)
        in_goal = bool(goal_distance <= 0.0)
        safe = bool(h_margin <= 0.0)
        invalid_dynamics = bool(validity_flags["invalid_dynamics"])
        hard_terminal = bool(self._hits_hard_terminal_v5(x_next_np))
        terminated = bool(invalid_dynamics or hard_terminal)
        terminated_reason = self._termination_reason_from_flags_v5(invalid_dynamics, hard_terminal)
        crash_cause = self._classify_crash_cause_v5(x_raw_np) if terminated else "none"

        return {
            "next_state": x_next_np,
            "raw_next_state": x_raw_np,
            "action_clipped": action_clipped,
            "control": control,
            "h_components": h_components,
            "h_margin": h_margin,
            "task_cost": task_cost,
            "reward": float(-task_cost),
            "binary_cost": float(h_margin > 0.0),
            "goal_distance": goal_distance,
            "in_goal": in_goal,
            "safe": safe,
            "invalid_dynamics": invalid_dynamics,
            "would_invalid_alpha": bool(validity_flags["alpha_invalid"]),
            "would_invalid_beta": bool(validity_flags["beta_invalid"]),
            "would_invalid_theta": bool(validity_flags["theta_invalid"]),
            "would_invalid_p": bool(validity_flags["p_invalid"]),
            "hard_terminal": hard_terminal,
            "terminated": terminated,
            "terminated_reason": terminated_reason,
            "crash_cause": crash_cause,
        }

    def step(self, action: np.ndarray):
        transition = self.simulate_transition(self._x, action)
        x_next = transition["next_state"]
        x_raw = transition["raw_next_state"]
        reward = float(transition["reward"])
        h_margin = float(transition["h_margin"])
        binary_cost = float(transition["binary_cost"])
        in_goal = bool(transition["in_goal"])
        safe = bool(transition["safe"])
        terminated = bool(transition["terminated"])
        invalid_dynamics = bool(transition["invalid_dynamics"])

        self._step_count += 1
        self._x = x_next
        self._prev_action = np.asarray(transition["action_clipped"], dtype=np.float32)

        self._ep_reward += reward
        self._ep_task_cost += float(transition["task_cost"])
        self._ep_unsafe_steps += binary_cost
        self._ep_length += 1
        self._ep_final_goal_distance = float(transition["goal_distance"])
        self._ep_min_altitude = min(self._ep_min_altitude, float(x_raw[IDX_H]))
        self._ep_peak_h = max(self._ep_peak_h, h_margin)
        for label, value in zip(_TASK_H_LABELS, transition["h_components"]):
            self._ep_peak_h_components[label] = max(self._ep_peak_h_components[label], float(value))
        self._ep_max_abs_beta = max(self._ep_max_abs_beta, abs(float(x_raw[IDX_BETA])))
        self._ep_max_abs_theta = max(self._ep_max_abs_theta, abs(float(x_raw[IDX_THETA])))
        self._ep_safe_steps += int(safe)
        self._ep_safe_in_goal_steps += int(safe and in_goal)

        if in_goal:
            self._ep_goal_steps += 1
            self._ep_goal_streak += 1
        else:
            self._ep_goal_streak = 0
        self._ep_max_goal_streak = max(self._ep_max_goal_streak, self._ep_goal_streak)

        success_now = False
        if (not self._ep_success) and (self._ep_goal_streak >= self.goal_dwell_steps):
            self._ep_success = True
            self._ep_success_step = self._step_count
            success_now = True

        self._ep_terminated = self._ep_terminated or terminated
        if terminated and self._ep_crash_step is None:
            self._ep_crash_step = self._step_count
        if terminated and self._ep_crash_cause is None:
            self._ep_crash_cause = str(transition["crash_cause"])
        if terminated and self._ep_terminate_reason is None:
            self._ep_terminate_reason = str(transition["terminated_reason"])

        timeout = self._step_count >= self.max_episode_steps
        done = bool(timeout or (self.terminate_on_crash and terminated))

        obs = self._get_obs(self._x)
        info = {
            "terminated": terminated,
            "timeout": timeout,
            "success": self._ep_success,
            "success_now": success_now,
            "in_goal": in_goal,
            "safe": safe,
            "goal_distance": float(transition["goal_distance"]),
            "invalid_dynamics": invalid_dynamics,
            "hard_terminal": bool(transition["hard_terminal"]),
            "crashed": terminated,
            "terminated_reason": str(transition["terminated_reason"]),
            "crash_cause": str(transition["crash_cause"]),
            "pn": float(x_raw[IDX_PN]),
            "pe": float(x_raw[IDX_PE]),
            "h_alt": float(x_raw[IDX_H]),
            "alpha": float(x_raw[IDX_ALPHA]),
            "beta": float(x_raw[IDX_BETA]),
            "phi": float(x_raw[IDX_PHI]),
            "theta": float(x_raw[IDX_THETA]),
            "p_rate": float(x_raw[IDX_P]),
            "q_rate": float(x_raw[IDX_Q]),
            "r_rate": float(x_raw[IDX_R]),
            "max_goal_streak": self._ep_max_goal_streak,
            "goal_steps": self._ep_goal_steps,
        }

        if done:
            self._finalize_episode_info()

        return obs, reward, h_margin, binary_cost, done, info

    # --------------------------------------------------------- episode stats
    def _curriculum_reset_metadata(self) -> Dict[str, float]:
        return {
            "curriculum_target_frac": self._last_reset_curriculum_target_frac,
            "curriculum_box1_target_frac": self._last_reset_curriculum_box1_frac,
            "curriculum_box2_target_frac": self._last_reset_curriculum_box2_frac,
            "curriculum_branch": self._last_reset_curriculum_branch,
            "curriculum_active": self._last_reset_curriculum_active,
            "curriculum_fallback": self._last_reset_curriculum_fallback,
            "curriculum_reject_rate": self._last_reset_curriculum_reject_rate,
            "init_goal": self._ep_init_goal,
            "init_safe": self._ep_init_safe,
            "init_h": self._ep_init_h,
            "init_h_alpha": self._ep_init_h_alpha,
            "init_h_beta": self._ep_init_h_beta,
            "init_alpha": self._ep_init_alpha,
            "init_abs_beta": self._ep_init_abs_beta,
            "init_pe_abs": self._ep_init_pe_abs,
        }

    def _finalize_episode_info(self):
        crash_cause = self._ep_crash_cause or "none"
        terminate_reason = self._ep_terminate_reason or "none"
        episode_success = self._ep_success and (not self._ep_terminated)
        info = {
            "task_spec_version": self.task_spec_version,
            "reward": self._ep_reward,
            "task_cost": self._ep_task_cost,
            "unsafe_step_count": self._ep_unsafe_steps,
            "length": self._ep_length,
            "terminated": self._ep_terminated,
            "success": episode_success,
            "goal_steps": self._ep_goal_steps,
            "goal_step_frac": self._ep_goal_steps / max(self._ep_length, 1),
            "safe_step_frac": self._ep_safe_steps / max(self._ep_length, 1),
            "safe_in_goal_step_frac": self._ep_safe_in_goal_steps / max(self._ep_length, 1),
            "max_goal_streak": self._ep_max_goal_streak,
            "time_to_success": self._ep_success_step if self._ep_success_step is not None else self.max_episode_steps,
            "crash_step": self._ep_crash_step if self._ep_crash_step is not None else self._ep_length,
            "crash_cause": crash_cause,
            "crash_cause_alpha": float(crash_cause == "alpha"),
            "crash_cause_p": float(crash_cause == "p"),
            "crash_cause_alt": float(crash_cause == "alt"),
            "crash_cause_pe": float(crash_cause == "pe"),
            "crash_cause_beta": float(crash_cause == "beta"),
            "crash_cause_theta": float(crash_cause == "theta"),
            "crash_cause_other": float(crash_cause == "other"),
            "terminate_reason": terminate_reason,
            "terminate_invalid_dynamics": float(terminate_reason == "invalid_dynamics"),
            "terminate_hard_terminal": float(terminate_reason == "hard_terminal"),
            "terminate_invalid_and_hard": float(terminate_reason == "invalid_and_hard"),
            "final_goal_distance": self._ep_final_goal_distance,
            "peak_h": self._ep_peak_h,
            "h_alt_peak": self._ep_peak_h_components["alt"],
            "h_alpha_peak": self._ep_peak_h_components["alpha"],
            "h_beta_peak": self._ep_peak_h_components["beta"],
            "h_theta_peak": self._ep_peak_h_components["theta"],
            "h_pe_peak": self._ep_peak_h_components["pe"],
            "h_p_peak": self._ep_peak_h_components["p"],
            "min_altitude": self._ep_min_altitude,
            "max_abs_beta": self._ep_max_abs_beta,
            "max_abs_theta": self._ep_max_abs_theta,
        }
        info.update(self._curriculum_reset_metadata())
        self.episode_info = info

    def _v5_sanity_check(self):
        rng = np.random.RandomState(42)
        samples, _ = self._sample_from_box_v5(
            self._task_train_reset_box(),
            rng,
            reject_fn=lambda _: False,
            num_samples=1000,
            max_tries=1000,
        )
        h_components_all = np.asarray([self._task_h_components(x) for x in samples], dtype=np.float64)
        h_arr = np.max(h_components_all, axis=1)
        accept_rate = float(np.mean(h_arr < 0.0))
        h_max = float(np.max(h_arr))
        if self.reset_box_mode == RESET_BOX_MODE_OURS:
            if accept_rate <= 0.98:
                raise ValueError(f"v5 reset sanity failed: accept_rate={accept_rate:.3f} <= 0.98")
            if h_max >= 0.0:
                raise ValueError(
                    f"v5 reset box violates SAFE set: max(init_h)={h_max:.3f} >= 0.0"
                )
            return
        if not self.reset_requires_safe:
            return
        accepted_samples, _ = self._sample_from_box_v5(
            self._task_train_reset_box(),
            rng,
            reject_fn=self._reject_initial_state_v5,
            num_samples=256,
            max_tries=20000,
        )
        accepted_h = np.asarray(
            [self._task_safety_margin(x) for x in accepted_samples],
            dtype=np.float64,
        )
        if np.max(accepted_h) >= 0.0:
            raise ValueError(
                "efppo_train accepted reset sampler violates SAFE set: "
                f"max(init_h)={np.max(accepted_h):+.3f} >= 0.0"
            )

    # -------------------------------------------------------------- observe
    def _get_obs(self, x: np.ndarray) -> np.ndarray:
        x_jnp = jnp.array(x)
        angles = x_jnp[_ANGLE_IDXS]
        other = x_jnp[_OTHER_IDXS]
        angles_enc = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)])
        state_enc = jnp.concatenate([other, angles_enc])
        vel_feats = _compute_vel_angles(x_jnp)
        state_enc = jnp.concatenate([state_enc, vel_feats])
        state_enc = (state_enc - _OBS_MEAN) / _OBS_STD
        state_enc = jnp.clip(state_enc, -10.0, 10.0)

        task_feats = jnp.asarray(self._task_obs_features(x))
        obs = jnp.concatenate([state_enc, task_feats])
        return np.asarray(obs, dtype=np.float32)

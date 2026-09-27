#!/usr/bin/env python
"""Train SSM on the Quad2D obstacle stabilize-avoid task."""

from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List, Tuple

os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from tqdm import tqdm

from ssm.tasks.quad2d_stabilize.env import (
    FIXED_PAPER_START,
    ACT_DIM,
    STATE_DIM,
    OBS_FEATURE_MODES,
    SUCCESS_MODES,
    Quad2DStabEnv,
    get_layout,
    mirror_action,
    mirror_state,
)


FLAGS = flags.FLAGS


def env_obs_feature_dim_from_flags() -> int:
    return Quad2DStabEnv._obs_feature_dim_for_mode(FLAGS.obs_feature_mode, get_layout(FLAGS.layout_name))

flags.DEFINE_string("project_name", "safefm-hj", "wandb project name.")
flags.DEFINE_string("entity", "bol025-uc-san-diego", "wandb entity.")
flags.DEFINE_string("run_name", "", "wandb run name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("max_steps", 2_000_000, "Total environment steps.")
flags.DEFINE_integer("start_training", 10_000, "Random warmup steps.")
flags.DEFINE_integer("batch_size", 512, "Batch size.")
flags.DEFINE_integer("eval_interval", 20_000, "Steps between evaluations.")
flags.DEFINE_integer("eval_episodes", 20, "Random-reset eval episodes.")
flags.DEFINE_integer("fixed_eval_repeats", 10, "Repeats per fixed start.")
flags.DEFINE_integer("log_interval", 1000, "Steps between logging.")
flags.DEFINE_boolean("wandb", True, "Enable wandb logging.")
flags.DEFINE_boolean("tqdm_bar", True, "Show tqdm progress bar.")
flags.DEFINE_enum("env_backend", "python", ["python", "p2"], "Environment step backend.")
flags.DEFINE_enum("eval_backend", "python", ["python", "p123"], "Training-time eval rollout backend.")

flags.DEFINE_string(
    "agent",
    "dzfill_qhstage_v_2",
    "Agent variant: nofill, dzfill, dzfill_qhstage_v_2, dzfill_adaptive, unified.",
)
flags.DEFINE_float("M_q", 10.0, "QSM scaling factor.")
flags.DEFINE_float("alpha_r", 1.0, "Reward gradient weight.")
flags.DEFINE_float("beta_safety", 5.0, "Safety gradient weight.")
flags.DEFINE_float("beta_max", 50.0, "Max beta(s) for adaptive agent.")
flags.DEFINE_float("beta_init", 5.0, "Initial beta(s) value.")
flags.DEFINE_float("beta_reg", 1.0, "Quadratic anchor regularizer strength.")
flags.DEFINE_float("beta_lr", 3e-4, "Learning rate for beta(s) network.")
flags.DEFINE_float("actor_lr", 3e-4, "Actor/score-model learning rate.")
flags.DEFINE_float("critic_lr", 3e-4, "Reward critic learning rate.")
flags.DEFINE_float("safe_critic_lr", 3e-4, "Safety Q_h critic learning rate.")
flags.DEFINE_float("safe_value_lr", 3e-4, "Safety V_h value learning rate.")
flags.DEFINE_float("delta", 0.0, "Safe set threshold.")
flags.DEFINE_float("discount", 0.99, "Reward discount.")
flags.DEFINE_float("discount_h", 0.999, "Safety discount.")
flags.DEFINE_float("tau", 0.005, "EMA rate.")
flags.DEFINE_float("cost_critic_hyperparam", 0.9, "Reversed expectile for V_h.")
flags.DEFINE_enum("qh_stage_mode", "stage_b", ["stage_a", "stage_b"], "Sampled-Q_h mode.")
flags.DEFINE_enum(
    "qh_candidate_source",
    "policy_plus_gaussian",
    ["policy_only", "policy_plus_gaussian"],
    "Candidate source for sampled-Q_h stage_b.",
)
flags.DEFINE_integer("qh_min_k_samples", 8, "Candidate count K for sampled-Q_h stage_b.")
flags.DEFINE_float("qh_min_k_sigma", 0.3, "Gaussian proposal std for sampled-Q_h stage_b.")
flags.DEFINE_string("beta_schedule", "vp", "DDPM beta schedule.")
flags.DEFINE_integer("T", 5, "DDPM diffusion steps.")
flags.DEFINE_integer("actor_delay", 1, "Actor update every N critic updates.")
flags.DEFINE_string("actor_hidden", "512,512,512", "Actor hidden dims.")
flags.DEFINE_string("critic_hidden", "512,512", "Critic hidden dims.")
flags.DEFINE_boolean("pure_qsm", False, "Pure QSM ablation.")
flags.DEFINE_boolean("adaptive_beta", False, "Use adaptive beta(s) for unified.")
flags.DEFINE_float("ddpm_temperature", 0.2, "DDPM sampling temperature.")
flags.DEFINE_float("fuse_clip_min", -10000.0, "Bellman target clip lower bound.")
flags.DEFINE_float("fuse_clip_max", 100.0, "Bellman target clip upper bound.")
flags.DEFINE_float("huber_delta", 10.0, "Huber delta for reward critic. 0=MSE.")

flags.DEFINE_string("layout_name", "corridor_v2", "Quad2D stab layout preset.")
flags.DEFINE_enum(
    "obs_feature_mode",
    "state",
    list(OBS_FEATURE_MODES),
    "Observation task features: state, h, h_components, h_components_closing, or h_components_closing_reach.",
)
flags.DEFINE_float("obs_h_scale", 1.0, "Scale for tanh-normalized h observation features.")
flags.DEFINE_float("obs_hdot_scale", 1.0, "Scale for tanh-normalized one-step hdot observation feature.")
flags.DEFINE_integer("max_episode_steps", 480, "Episode horizon.")
flags.DEFINE_float("q_pos", 10.0, "Position weight for x/z.")
flags.DEFINE_float("q_x", -1.0, "Optional x-position weight. Negative means use q_pos.")
flags.DEFINE_float("q_z", -1.0, "Optional z-position weight. Negative means use q_pos.")
flags.DEFINE_float("q_vel", 1.0, "Velocity weight for xdot/zdot.")
flags.DEFINE_float("q_angle", 0.2, "Angle and angular-rate weight.")
flags.DEFINE_float("action_penalty", 1e-3, "Action penalty on hover-centered raw thrust.")
flags.DEFINE_string("reward_norm", "l2", "Reward norm: l2 or l1.")
flags.DEFINE_float("dwell_bonus", 0.0, "Per-step reward bonus while inside the dwell goal box.")
flags.DEFINE_float("goal_pos_radius", 0.30, "Position radius for dwell success.")
flags.DEFINE_float("goal_vel_radius", 0.45, "Velocity radius for dwell success.")
flags.DEFINE_float("goal_angle_radius", 0.25, "Theta radius for dwell success.")
flags.DEFINE_integer("goal_dwell_steps", 45, "Consecutive goal steps needed for success.")
flags.DEFINE_enum(
    "success_mode",
    "full_state_terminal",
    list(SUCCESS_MODES),
    "Success definition. full_state_terminal matches Quad3D: terminal ||x_T - x_ref|| < success_radius.",
)
flags.DEFINE_float("success_radius", 0.50, "Terminal full-state norm radius for full_state_terminal success.")
flags.DEFINE_boolean("terminate_on_success", False, "End episode once dwell success is reached in dwell_box mode.")
flags.DEFINE_enum(
    "reset_mode",
    "start_band",
    [
        "start_band",
        "global_safe",
        "mix",
        "route_mix",
        "under_mix",
        "low_mix",
        "gap_mix",
        "simple_under3",
        "safety_under50",
    ],
    "Training reset sampler.",
)
flags.DEFINE_float("reset_mix_start_frac", 0.70, "Mix reset fraction for start band.")
flags.DEFINE_float("reset_mix_global_frac", 0.20, "Mix reset fraction for global safe states.")
flags.DEFINE_float("reset_mix_goal_frac", 0.10, "Mix reset fraction for near-goal states.")
flags.DEFINE_float("reset_route_global_frac", 0.35, "Route-mix fraction for global safe states.")
flags.DEFINE_float("reset_route_near_frac", 0.30, "Route-mix fraction for wide near-goal/red-region states.")
flags.DEFINE_float("reset_route_gap_frac", 0.35, "Route-mix fraction for route gap and detour states.")
flags.DEFINE_float("reset_under_global_frac", 0.35, "Under-mix fraction for global safe states.")
flags.DEFINE_float("reset_under_frac", 0.65, "Under-mix fraction for obstacle-under curriculum states.")
flags.DEFINE_float("reset_low_frac", 0.80, "Low-mix fraction for z<0.5 bottleneck states.")
flags.DEFINE_float("reset_low_mid_frac", 0.15, "Low-mix fraction for mid corridor continuation states.")
flags.DEFINE_float("reset_low_near_frac", 0.05, "Low-mix fraction for near-goal states.")
flags.DEFINE_float("reset_low_paper_jitter_frac", 0.30, "Low component fraction for paper-jitter states.")
flags.DEFINE_float("reset_low_paper_pinpoint_vel_frac", 0.0, "Low component fraction for paper-center pinpoint velocity states.")
flags.DEFINE_float("reset_low_under_bar_frac", 0.20, "Low component fraction for horizontal-bar underside states.")
flags.DEFINE_float("reset_low_under_blocks_frac", 0.20, "Low component fraction for side-block underside states.")
flags.DEFINE_float("reset_low_inner_gaps_frac", 0.20, "Low component fraction for inner-gap states.")
flags.DEFINE_float("reset_low_global_frac", 0.10, "Low component fraction for broad low safe states.")
flags.DEFINE_float("reset_simple_bar_frac", 0.30, "Simple-under3 fraction for horizontal-bar underside states.")
flags.DEFINE_float("reset_simple_left_block_frac", 0.10, "Simple-under3 fraction for left-block underside states.")
flags.DEFINE_float("reset_simple_right_block_frac", 0.10, "Simple-under3 fraction for right-block underside states.")
flags.DEFINE_float("reset_simple_gap_frac", 0.10, "Simple-under3 fraction for inner-gap entry states.")
flags.DEFINE_float("reset_simple_low_uniform_frac", 0.25, "Simple-under3 fraction for broad z<0.5 safe states.")
flags.DEFINE_float("reset_simple_near_frac", 0.15, "Simple-under3 fraction for near-goal states.")
flags.DEFINE_float("reset_simple_side_frac", 0.0, "Simple-under3 fraction for goal-side recovery states.")
flags.DEFINE_float("reset_simple_fork_frac", 0.0, "Simple-under3 fraction for fixed-start pre-fork decision-band states.")
flags.DEFINE_float("reset_safety_bar_frac", 0.20, "Safety-under50 fraction for horizontal-bar underside states.")
flags.DEFINE_float("reset_safety_left_block_frac", 0.15, "Safety-under50 fraction for left-block underside states.")
flags.DEFINE_float("reset_safety_right_block_frac", 0.15, "Safety-under50 fraction for right-block underside states.")
flags.DEFINE_float("reset_safety_gap_frac", 0.15, "Safety-under50 fraction for inner-gap entry states.")
flags.DEFINE_float("reset_safety_low_uniform_frac", 0.25, "Safety-under50 fraction for broad z<0.5 safe states.")
flags.DEFINE_float("reset_safety_mid_uniform_frac", 0.05, "Safety-under50 fraction for mid corridor continuation states.")
flags.DEFINE_float("reset_safety_near_frac", 0.05, "Safety-under50 fraction for near-goal states.")
flags.DEFINE_float("simple_under_vx_abs", 0.50, "Simple-under3 obstacle-under absolute x velocity range.")
flags.DEFINE_float("simple_under_vz_low", 0.00, "Simple-under3 obstacle-under minimum z velocity.")
flags.DEFINE_float("simple_under_vz_high", 0.70, "Simple-under3 obstacle-under maximum z velocity.")
flags.DEFINE_float("simple_bar_vz_low", -1.0, "Simple-under3 bar-under minimum z velocity. Negative means use simple_under_vz_low.")
flags.DEFINE_float("simple_bar_vz_high", -1.0, "Simple-under3 bar-under maximum z velocity. Negative means use simple_under_vz_high.")
flags.DEFINE_float("simple_block_vz_low", -1.0, "Simple-under3 block-under minimum z velocity. Negative means use simple_under_vz_low.")
flags.DEFINE_float("simple_block_vz_high", -1.0, "Simple-under3 block-under maximum z velocity. Negative means use simple_under_vz_high.")
flags.DEFINE_float("simple_under_xdot_zero_frac", 0.0, "Simple-under3 obstacle-under fraction with exactly zero initial x velocity.")
flags.DEFINE_float("simple_bar_xdot_zero_frac", -1.0, "Simple-under3 bar-under zero-xdot fraction. Negative means use simple_under_xdot_zero_frac.")
flags.DEFINE_float("simple_block_xdot_zero_frac", -1.0, "Simple-under3 block-under zero-xdot fraction. Negative means use simple_under_xdot_zero_frac.")
flags.DEFINE_float("simple_gap_vx_abs", 0.40, "Simple-under3 gap-entry absolute x velocity range.")
flags.DEFINE_float("simple_gap_vz_low", -0.20, "Simple-under3 gap-entry minimum z velocity.")
flags.DEFINE_float("simple_gap_vz_high", 0.50, "Simple-under3 gap-entry maximum z velocity.")
flags.DEFINE_float("simple_low_uniform_vx_abs", 0.35, "Simple-under3 low-uniform absolute x velocity range.")
flags.DEFINE_float("simple_low_uniform_vz_abs", 0.35, "Simple-under3 low-uniform absolute z velocity range.")
flags.DEFINE_float("simple_fork_x_low", 0.05, "Pre-fork reset absolute x lower bound.")
flags.DEFINE_float("simple_fork_x_high", 0.35, "Pre-fork reset absolute x upper bound.")
flags.DEFINE_float("simple_fork_z_low", -1.00, "Pre-fork reset z lower bound.")
flags.DEFINE_float("simple_fork_z_high", -0.30, "Pre-fork reset z upper bound.")
flags.DEFINE_float("simple_fork_vx_low", 0.10, "Pre-fork reset side-biased x velocity lower bound.")
flags.DEFINE_float("simple_fork_vx_high", 0.55, "Pre-fork reset side-biased x velocity upper bound.")
flags.DEFINE_float("simple_fork_vz_low", -0.05, "Pre-fork reset z velocity lower bound.")
flags.DEFINE_float("simple_fork_vz_high", 0.45, "Pre-fork reset z velocity upper bound.")
flags.DEFINE_float("simple_fork_theta_abs", 0.12, "Pre-fork reset absolute theta range.")
flags.DEFINE_float("simple_fork_omega_abs", 0.15, "Pre-fork reset absolute theta-dot range.")
flags.DEFINE_float("low_paper_jitter_vx_abs", 0.25, "Paper-jitter absolute x velocity reset range.")
flags.DEFINE_float("low_paper_jitter_vz_abs", 0.25, "Paper-jitter absolute z velocity reset range.")
flags.DEFINE_float("low_paper_pinpoint_vx_abs", 1.00, "Paper-pinpoint absolute x velocity reset range.")
flags.DEFINE_float("low_paper_pinpoint_vz_abs", 0.30, "Paper-pinpoint absolute z velocity reset range.")
flags.DEFINE_float("low_paper_pinpoint_theta_abs", 0.10, "Paper-pinpoint absolute theta reset range.")
flags.DEFINE_float("low_paper_pinpoint_omega_abs", 0.05, "Paper-pinpoint absolute theta-dot reset range.")
flags.DEFINE_float("low_paper_pinpoint_x_abs", 0.05, "Paper-pinpoint half-width in x.")
flags.DEFINE_float("low_paper_pinpoint_z_center", -1.08, "Paper-pinpoint z center.")
flags.DEFINE_float("low_paper_pinpoint_z_halfwidth", 0.02, "Paper-pinpoint z half-width.")
flags.DEFINE_float("low_under_bar_vx_abs", 0.25, "Under-bar absolute x velocity reset range.")
flags.DEFINE_float("low_under_bar_vz_abs", 0.25, "Under-bar absolute z velocity reset range.")
flags.DEFINE_float("low_under_blocks_vx_abs", 0.25, "Under-blocks absolute x velocity reset range.")
flags.DEFINE_float("low_under_blocks_vz_abs", 0.25, "Under-blocks absolute z velocity reset range.")
flags.DEFINE_float("low_inner_gaps_vx_abs", 0.25, "Inner-gap absolute x velocity reset range.")
flags.DEFINE_float("low_inner_gaps_vz_abs", 0.25, "Inner-gap absolute z velocity reset range.")
flags.DEFINE_enum(
    "eval_reset_mode",
    "start_band",
    [
        "start_band",
        "global_safe",
        "mix",
        "route_mix",
        "under_mix",
        "low_mix",
        "gap_mix",
        "simple_under3",
        "safety_under50",
    ],
    "Reset mode used by random online eval episodes.",
)
flags.DEFINE_enum(
    "eval_suite",
    "default",
    ["default", "mixed50", "lowz050_hrej", "lowz055_hrej"],
    "Online eval suite. mixed50 uses paper/low/gap/global/near; lowz*_hrej uses broad low-z h-rejected starts.",
)
flags.DEFINE_float("init_h_threshold", 0.0, "Require h_phys < threshold on random reset.")

flags.DEFINE_string("sdf_mode", "raw", "SDF shaping mode: raw, hard, hard_slope, hard_scale.")
flags.DEFINE_float("sdf_unsafe_const", 0.0, "Unsafe constant for SDF shaping.")
flags.DEFINE_float("sdf_unsafe_slope", 0.0, "Unsafe slope for hard_slope shaping.")
flags.DEFINE_float("sdf_global_scale", 0.0, "Global SDF scale for hard_scale shaping.")
flags.DEFINE_float("gate_lambda", 0.0, "Symmetric horizontal-bar gate reward weight. 0 disables gate shaping.")
flags.DEFINE_float("gate_x_max", 0.90, "Maximum required |x| for gate reward shaping.")
flags.DEFINE_float("gate_z_center", 0.30, "Gate reward z center.")
flags.DEFINE_float("gate_z_width", 1.20, "Gate reward triangular half-width in z.")

flags.DEFINE_boolean("boundary_replay", False, "Enable boundary replay oversampling.")
flags.DEFINE_enum("boundary_mode", "near_unsafe", ["abs", "near_unsafe"], "Boundary replay criterion.")
flags.DEFINE_float("boundary_eps", 0.05, "Boundary replay threshold on h_phys.")
flags.DEFINE_float("boundary_frac", 0.25, "Target boundary fraction in replay samples.")
flags.DEFINE_integer("boundary_min_count", 32, "Minimum boundary samples before oversampling activates.")
flags.DEFINE_float("hard_margin", 0.0, "Safety routing margin. 0 uses the raw h/Q_h=0 boundary.")
flags.DEFINE_float("h_hardgap", 0.0, "Optional unsafe-side hard gap for Q_h targets. 0 disables it.")
flags.DEFINE_boolean("use_action_margin", False, "Apply hard_margin to action-level Q_h indicator.")
flags.DEFINE_boolean("mirror_augment", False, "Insert left/right mirrored transitions into replay.")
flags.DEFINE_string("init_checkpoint", "", "Optional checkpoint path for warm-starting network params and obs stats.")
flags.DEFINE_string(
    "fixed_eval_starts",
    "paper_center,paper_left,paper_right,near_goal",
    "Comma-separated fixed-start labels to evaluate when fixed_eval_repeats > 0.",
)


class SimpleReplayBuffer:
    """Numpy replay buffer with optional h_phys boundary oversampling."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int = 1_000_000,
        boundary_replay: bool = False,
        boundary_mode: str = "near_unsafe",
        boundary_eps: float = 0.05,
        boundary_frac: float = 0.25,
        boundary_min_count: int = 32,
    ) -> None:
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.boundary_replay = bool(boundary_replay)
        self.boundary_mode = str(boundary_mode)
        self.boundary_eps = float(boundary_eps)
        self.boundary_frac = float(boundary_frac)
        self.boundary_min_count = int(boundary_min_count)

        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.costs = np.zeros(max_size, dtype=np.float32)
        self.h_phys = np.zeros(max_size, dtype=np.float32)
        self.masks = np.zeros(max_size, dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)

        self._boundary_slots = np.empty(max_size, dtype=np.int32)
        self._boundary_pos = np.full(max_size, -1, dtype=np.int32)
        self._boundary_size = 0
        self._unsafe_slots = np.empty(max_size, dtype=np.int32)
        self._unsafe_pos = np.full(max_size, -1, dtype=np.int32)
        self._unsafe_size = 0
        self._rng = np.random.default_rng(0)
        self.last_sample_info = {
            "boundary_active": 0.0,
            "boundary_draw_frac": 0.0,
            "boundary_frac_in_buffer": 0.0,
            "unsafe_frac_in_buffer": 0.0,
            "boundary_frac_in_batch": 0.0,
            "unsafe_frac_in_batch": 0.0,
            "costs_mean_in_batch": 0.0,
        }

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _boundary_mask(self, h_values):
        h_values = np.asarray(h_values)
        if self.boundary_mode == "abs":
            return np.abs(h_values) <= self.boundary_eps
        return h_values >= -self.boundary_eps

    def _update_membership(self, idx, is_member, slots, positions, count):
        pos = positions[idx]
        if is_member:
            if pos < 0:
                slots[count] = idx
                positions[idx] = count
                count += 1
            return count
        if pos < 0:
            return count
        last_idx = slots[count - 1]
        slots[pos] = last_idx
        positions[last_idx] = pos
        positions[idx] = -1
        return count - 1

    def insert(self, data: Dict[str, object]) -> None:
        i = self.ptr
        self.observations[i] = data["observations"]
        self.actions[i] = data["actions"]
        self.rewards[i] = data["rewards"]
        self.costs[i] = data["costs"]
        self.h_phys[i] = data["h_phys"]
        self.masks[i] = data["masks"]
        self.next_observations[i] = data["next_observations"]

        h = float(self.h_phys[i])
        cost = float(self.costs[i])
        self._boundary_size = self._update_membership(
            i,
            bool(self._boundary_mask(h)),
            self._boundary_slots,
            self._boundary_pos,
            self._boundary_size,
        )
        self._unsafe_size = self._update_membership(
            i,
            bool(h > 0.0 or cost > 0.0),
            self._unsafe_slots,
            self._unsafe_pos,
            self._unsafe_size,
        )
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        target_boundary = int(round(batch_size * self.boundary_frac))
        boundary_active = (
            self.boundary_replay
            and self.boundary_eps > 0.0
            and target_boundary > 0
            and self._boundary_size >= self.boundary_min_count
        )
        if boundary_active:
            num_boundary = min(target_boundary, self._boundary_size, batch_size)
            boundary_pos = self._rng.choice(self._boundary_size, size=num_boundary, replace=False)
            boundary_idxs = self._boundary_slots[boundary_pos]
            uniform_idxs = self._rng.integers(0, self.size, size=batch_size - num_boundary)
            idxs = np.concatenate([boundary_idxs, uniform_idxs], axis=0)
            self._rng.shuffle(idxs)
        else:
            num_boundary = 0
            idxs = self._rng.integers(0, self.size, size=batch_size)

        batch_h = self.h_phys[idxs]
        batch_costs = self.costs[idxs]
        self.last_sample_info = {
            "boundary_active": float(boundary_active),
            "boundary_draw_frac": num_boundary / batch_size,
            "boundary_frac_in_buffer": self._boundary_size / max(self.size, 1),
            "unsafe_frac_in_buffer": self._unsafe_size / max(self.size, 1),
            "boundary_frac_in_batch": float(np.mean(self._boundary_mask(batch_h))),
            "unsafe_frac_in_batch": float(np.mean(batch_h > 0.0)),
            "costs_mean_in_batch": float(np.mean(batch_costs)),
        }
        return {
            "observations": jnp.asarray(self.observations[idxs]),
            "actions": jnp.asarray(self.actions[idxs]),
            "rewards": jnp.asarray(self.rewards[idxs]),
            "costs": jnp.asarray(self.costs[idxs]),
            "masks": jnp.asarray(self.masks[idxs]),
            "next_observations": jnp.asarray(self.next_observations[idxs]),
        }


def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x)


def get_agent_class(version):
    if version not in ("dzfill_qhstage", "dzfill_qhstage_v_2", "dzfill_qhstage_v2"):
        raise ValueError("This example uses the sampled-Qh SSM learner.")
    from ssm.agents.ssm_agent import SSMOnlineAgent
    return SSMOnlineAgent


def resolved_q_weights() -> Tuple[float, float]:
    q_x = FLAGS.q_pos if FLAGS.q_x < 0.0 else FLAGS.q_x
    q_z = FLAGS.q_pos if FLAGS.q_z < 0.0 else FLAGS.q_z
    return float(q_x), float(q_z)


def q_axis_split_active() -> bool:
    q_x, q_z = resolved_q_weights()
    return not (np.isclose(q_x, FLAGS.q_pos) and np.isclose(q_z, FLAGS.q_pos))


def resolved_simple_under_vz_ranges() -> Tuple[float, float, float, float]:
    bar_low = FLAGS.simple_under_vz_low if FLAGS.simple_bar_vz_low < 0.0 else FLAGS.simple_bar_vz_low
    bar_high = FLAGS.simple_under_vz_high if FLAGS.simple_bar_vz_high < 0.0 else FLAGS.simple_bar_vz_high
    block_low = FLAGS.simple_under_vz_low if FLAGS.simple_block_vz_low < 0.0 else FLAGS.simple_block_vz_low
    block_high = FLAGS.simple_under_vz_high if FLAGS.simple_block_vz_high < 0.0 else FLAGS.simple_block_vz_high
    return float(bar_low), float(bar_high), float(block_low), float(block_high)


def resolved_simple_under_xdot_zero_fracs() -> Tuple[float, float]:
    bar_frac = (
        FLAGS.simple_under_xdot_zero_frac
        if FLAGS.simple_bar_xdot_zero_frac < 0.0
        else FLAGS.simple_bar_xdot_zero_frac
    )
    block_frac = (
        FLAGS.simple_under_xdot_zero_frac
        if FLAGS.simple_block_xdot_zero_frac < 0.0
        else FLAGS.simple_block_xdot_zero_frac
    )
    return float(np.clip(bar_frac, 0.0, 1.0)), float(np.clip(block_frac, 0.0, 1.0))


def make_env(seed: int, for_eval: bool = False) -> Quad2DStabEnv:
    env_cls = Quad2DStabEnv
    if FLAGS.env_backend == "p2":
        from ssm.tasks.quad2d_stabilize.accelerated import Quad2DStabEnvP2

        env_cls = Quad2DStabEnvP2
    env = env_cls(
        seed=seed,
        layout_name=FLAGS.layout_name,
        max_episode_steps=FLAGS.max_episode_steps,
        q_pos=FLAGS.q_pos,
        q_x=resolved_q_weights()[0],
        q_z=resolved_q_weights()[1],
        q_vel=FLAGS.q_vel,
        q_angle=FLAGS.q_angle,
        action_penalty=FLAGS.action_penalty,
        reward_norm=FLAGS.reward_norm,
        dwell_bonus=FLAGS.dwell_bonus,
        goal_pos_radius=FLAGS.goal_pos_radius,
        goal_vel_radius=FLAGS.goal_vel_radius,
        goal_angle_radius=FLAGS.goal_angle_radius,
        goal_dwell_steps=FLAGS.goal_dwell_steps,
        success_mode=FLAGS.success_mode,
        success_radius=FLAGS.success_radius,
        terminate_on_success=FLAGS.terminate_on_success,
        reset_mode=(FLAGS.eval_reset_mode if for_eval else FLAGS.reset_mode),
        reset_mix_start_frac=FLAGS.reset_mix_start_frac,
        reset_mix_global_frac=FLAGS.reset_mix_global_frac,
        reset_mix_goal_frac=FLAGS.reset_mix_goal_frac,
        reset_route_global_frac=FLAGS.reset_route_global_frac,
        reset_route_near_frac=FLAGS.reset_route_near_frac,
        reset_route_gap_frac=FLAGS.reset_route_gap_frac,
        reset_under_global_frac=FLAGS.reset_under_global_frac,
        reset_under_frac=FLAGS.reset_under_frac,
        reset_low_frac=FLAGS.reset_low_frac,
        reset_low_mid_frac=FLAGS.reset_low_mid_frac,
        reset_low_near_frac=FLAGS.reset_low_near_frac,
        reset_low_paper_jitter_frac=FLAGS.reset_low_paper_jitter_frac,
        reset_low_paper_pinpoint_vel_frac=FLAGS.reset_low_paper_pinpoint_vel_frac,
        reset_low_under_bar_frac=FLAGS.reset_low_under_bar_frac,
        reset_low_under_blocks_frac=FLAGS.reset_low_under_blocks_frac,
        reset_low_inner_gaps_frac=FLAGS.reset_low_inner_gaps_frac,
        reset_low_global_frac=FLAGS.reset_low_global_frac,
        reset_simple_bar_frac=FLAGS.reset_simple_bar_frac,
        reset_simple_left_block_frac=FLAGS.reset_simple_left_block_frac,
        reset_simple_right_block_frac=FLAGS.reset_simple_right_block_frac,
        reset_simple_gap_frac=FLAGS.reset_simple_gap_frac,
        reset_simple_low_uniform_frac=FLAGS.reset_simple_low_uniform_frac,
        reset_simple_near_frac=FLAGS.reset_simple_near_frac,
        reset_simple_side_frac=FLAGS.reset_simple_side_frac,
        reset_simple_fork_frac=FLAGS.reset_simple_fork_frac,
        reset_safety_bar_frac=FLAGS.reset_safety_bar_frac,
        reset_safety_left_block_frac=FLAGS.reset_safety_left_block_frac,
        reset_safety_right_block_frac=FLAGS.reset_safety_right_block_frac,
        reset_safety_gap_frac=FLAGS.reset_safety_gap_frac,
        reset_safety_low_uniform_frac=FLAGS.reset_safety_low_uniform_frac,
        reset_safety_mid_uniform_frac=FLAGS.reset_safety_mid_uniform_frac,
        reset_safety_near_frac=FLAGS.reset_safety_near_frac,
        simple_under_vx_abs=FLAGS.simple_under_vx_abs,
        simple_under_vz_low=FLAGS.simple_under_vz_low,
        simple_under_vz_high=FLAGS.simple_under_vz_high,
        simple_bar_vz_low=resolved_simple_under_vz_ranges()[0],
        simple_bar_vz_high=resolved_simple_under_vz_ranges()[1],
        simple_block_vz_low=resolved_simple_under_vz_ranges()[2],
        simple_block_vz_high=resolved_simple_under_vz_ranges()[3],
        simple_under_xdot_zero_frac=FLAGS.simple_under_xdot_zero_frac,
        simple_bar_xdot_zero_frac=resolved_simple_under_xdot_zero_fracs()[0],
        simple_block_xdot_zero_frac=resolved_simple_under_xdot_zero_fracs()[1],
        simple_gap_vx_abs=FLAGS.simple_gap_vx_abs,
        simple_gap_vz_low=FLAGS.simple_gap_vz_low,
        simple_gap_vz_high=FLAGS.simple_gap_vz_high,
        simple_low_uniform_vx_abs=FLAGS.simple_low_uniform_vx_abs,
        simple_low_uniform_vz_abs=FLAGS.simple_low_uniform_vz_abs,
        simple_fork_x_low=FLAGS.simple_fork_x_low,
        simple_fork_x_high=FLAGS.simple_fork_x_high,
        simple_fork_z_low=FLAGS.simple_fork_z_low,
        simple_fork_z_high=FLAGS.simple_fork_z_high,
        simple_fork_vx_low=FLAGS.simple_fork_vx_low,
        simple_fork_vx_high=FLAGS.simple_fork_vx_high,
        simple_fork_vz_low=FLAGS.simple_fork_vz_low,
        simple_fork_vz_high=FLAGS.simple_fork_vz_high,
        simple_fork_theta_abs=FLAGS.simple_fork_theta_abs,
        simple_fork_omega_abs=FLAGS.simple_fork_omega_abs,
        low_paper_jitter_vx_abs=FLAGS.low_paper_jitter_vx_abs,
        low_paper_jitter_vz_abs=FLAGS.low_paper_jitter_vz_abs,
        low_paper_pinpoint_vx_abs=FLAGS.low_paper_pinpoint_vx_abs,
        low_paper_pinpoint_vz_abs=FLAGS.low_paper_pinpoint_vz_abs,
        low_paper_pinpoint_theta_abs=FLAGS.low_paper_pinpoint_theta_abs,
        low_paper_pinpoint_omega_abs=FLAGS.low_paper_pinpoint_omega_abs,
        low_paper_pinpoint_x_abs=FLAGS.low_paper_pinpoint_x_abs,
        low_paper_pinpoint_z_center=FLAGS.low_paper_pinpoint_z_center,
        low_paper_pinpoint_z_halfwidth=FLAGS.low_paper_pinpoint_z_halfwidth,
        low_under_bar_vx_abs=FLAGS.low_under_bar_vx_abs,
        low_under_bar_vz_abs=FLAGS.low_under_bar_vz_abs,
        low_under_blocks_vx_abs=FLAGS.low_under_blocks_vx_abs,
        low_under_blocks_vz_abs=FLAGS.low_under_blocks_vz_abs,
        low_inner_gaps_vx_abs=FLAGS.low_inner_gaps_vx_abs,
        low_inner_gaps_vz_abs=FLAGS.low_inner_gaps_vz_abs,
        init_h_threshold=FLAGS.init_h_threshold,
        sdf_mode=FLAGS.sdf_mode,
        sdf_unsafe_const=FLAGS.sdf_unsafe_const,
        sdf_unsafe_slope=FLAGS.sdf_unsafe_slope,
        sdf_global_scale=FLAGS.sdf_global_scale,
        gate_lambda=FLAGS.gate_lambda,
        gate_x_max=FLAGS.gate_x_max,
        gate_z_center=FLAGS.gate_z_center,
        gate_z_width=FLAGS.gate_z_width,
        obs_feature_mode=FLAGS.obs_feature_mode,
        obs_h_scale=FLAGS.obs_h_scale,
        obs_hdot_scale=FLAGS.obs_hdot_scale,
    )
    if for_eval:
        env.set_eval_mode()
    return env


def build_run_config() -> Dict[str, object]:
    is_adaptive = FLAGS.agent == "dzfill_adaptive" or (FLAGS.agent == "unified" and FLAGS.adaptive_beta)
    q_x, q_z = resolved_q_weights()
    bar_vz_low, bar_vz_high, block_vz_low, block_vz_high = resolved_simple_under_vz_ranges()
    bar_xzero_frac, block_xzero_frac = resolved_simple_under_xdot_zero_fracs()
    return {
        "env": "Quad2DStab",
        "accel_stage": "p123" if (FLAGS.env_backend == "p2" or FLAGS.eval_backend == "p123") else "python",
        "env_backend": FLAGS.env_backend,
        "eval_backend": FLAGS.eval_backend,
        "fused_update": True,
        "seed": FLAGS.seed,
        "agent": FLAGS.agent,
        "M_q": FLAGS.M_q,
        "alpha_r": FLAGS.alpha_r,
        "beta_safety": FLAGS.beta_safety,
        "beta_max": FLAGS.beta_max,
        "beta_init": FLAGS.beta_init,
        "beta_reg": FLAGS.beta_reg,
        "beta_lr": FLAGS.beta_lr,
        "actor_lr": FLAGS.actor_lr,
        "critic_lr": FLAGS.critic_lr,
        "safe_critic_lr": FLAGS.safe_critic_lr,
        "safe_value_lr": FLAGS.safe_value_lr,
        "delta": FLAGS.delta,
        "discount": FLAGS.discount,
        "discount_h": FLAGS.discount_h,
        "tau": FLAGS.tau,
        "cost_critic_hyperparam": FLAGS.cost_critic_hyperparam,
        "qh_stage_mode": FLAGS.qh_stage_mode,
        "qh_candidate_source": FLAGS.qh_candidate_source,
        "qh_min_k_samples": FLAGS.qh_min_k_samples,
        "qh_min_k_sigma": FLAGS.qh_min_k_sigma,
        "T": FLAGS.T,
        "beta_schedule": FLAGS.beta_schedule,
        "batch_size": FLAGS.batch_size,
        "actor_delay": FLAGS.actor_delay,
        "max_steps": FLAGS.max_steps,
        "start_training": FLAGS.start_training,
        "actor_hidden": FLAGS.actor_hidden,
        "critic_hidden": FLAGS.critic_hidden,
        "pure_qsm": FLAGS.pure_qsm,
        "ddpm_temperature": FLAGS.ddpm_temperature,
        "adaptive_beta": is_adaptive,
        "fuse_clip_min": FLAGS.fuse_clip_min,
        "fuse_clip_max": FLAGS.fuse_clip_max,
        "huber_delta": FLAGS.huber_delta,
        "layout_name": FLAGS.layout_name,
        "obs_feature_mode": FLAGS.obs_feature_mode,
        "obs_h_scale": FLAGS.obs_h_scale,
        "obs_hdot_scale": FLAGS.obs_hdot_scale,
        "obs_dim": STATE_DIM + env_obs_feature_dim_from_flags(),
        "dt": 1.0 / 60.0,
        "max_episode_steps": FLAGS.max_episode_steps,
        "q_pos": FLAGS.q_pos,
        "q_x": q_x,
        "q_z": q_z,
        "q_axis_split": q_axis_split_active(),
        "q_vel": FLAGS.q_vel,
        "q_angle": FLAGS.q_angle,
        "action_penalty": FLAGS.action_penalty,
        "reward_norm": FLAGS.reward_norm,
        "dwell_bonus": FLAGS.dwell_bonus,
        "goal_pos_radius": FLAGS.goal_pos_radius,
        "goal_vel_radius": FLAGS.goal_vel_radius,
        "goal_angle_radius": FLAGS.goal_angle_radius,
        "goal_dwell_steps": FLAGS.goal_dwell_steps,
        "success_mode": FLAGS.success_mode,
        "success_radius": FLAGS.success_radius,
        "terminate_on_success": FLAGS.terminate_on_success,
        "reset_mode": FLAGS.reset_mode,
        "reset_mix_start_frac": FLAGS.reset_mix_start_frac,
        "reset_mix_global_frac": FLAGS.reset_mix_global_frac,
        "reset_mix_goal_frac": FLAGS.reset_mix_goal_frac,
        "reset_route_global_frac": FLAGS.reset_route_global_frac,
        "reset_route_near_frac": FLAGS.reset_route_near_frac,
        "reset_route_gap_frac": FLAGS.reset_route_gap_frac,
        "reset_under_global_frac": FLAGS.reset_under_global_frac,
        "reset_under_frac": FLAGS.reset_under_frac,
        "reset_low_frac": FLAGS.reset_low_frac,
        "reset_low_mid_frac": FLAGS.reset_low_mid_frac,
        "reset_low_near_frac": FLAGS.reset_low_near_frac,
        "reset_low_paper_jitter_frac": FLAGS.reset_low_paper_jitter_frac,
        "reset_low_paper_pinpoint_vel_frac": FLAGS.reset_low_paper_pinpoint_vel_frac,
        "reset_low_under_bar_frac": FLAGS.reset_low_under_bar_frac,
        "reset_low_under_blocks_frac": FLAGS.reset_low_under_blocks_frac,
        "reset_low_inner_gaps_frac": FLAGS.reset_low_inner_gaps_frac,
        "reset_low_global_frac": FLAGS.reset_low_global_frac,
        "reset_simple_bar_frac": FLAGS.reset_simple_bar_frac,
        "reset_simple_left_block_frac": FLAGS.reset_simple_left_block_frac,
        "reset_simple_right_block_frac": FLAGS.reset_simple_right_block_frac,
        "reset_simple_gap_frac": FLAGS.reset_simple_gap_frac,
        "reset_simple_low_uniform_frac": FLAGS.reset_simple_low_uniform_frac,
        "reset_simple_near_frac": FLAGS.reset_simple_near_frac,
        "reset_simple_side_frac": FLAGS.reset_simple_side_frac,
        "reset_simple_fork_frac": FLAGS.reset_simple_fork_frac,
        "reset_safety_bar_frac": FLAGS.reset_safety_bar_frac,
        "reset_safety_left_block_frac": FLAGS.reset_safety_left_block_frac,
        "reset_safety_right_block_frac": FLAGS.reset_safety_right_block_frac,
        "reset_safety_gap_frac": FLAGS.reset_safety_gap_frac,
        "reset_safety_low_uniform_frac": FLAGS.reset_safety_low_uniform_frac,
        "reset_safety_mid_uniform_frac": FLAGS.reset_safety_mid_uniform_frac,
        "reset_safety_near_frac": FLAGS.reset_safety_near_frac,
        "simple_under_vx_abs": FLAGS.simple_under_vx_abs,
        "simple_under_vz_low": FLAGS.simple_under_vz_low,
        "simple_under_vz_high": FLAGS.simple_under_vz_high,
        "simple_bar_vz_low": bar_vz_low,
        "simple_bar_vz_high": bar_vz_high,
        "simple_block_vz_low": block_vz_low,
        "simple_block_vz_high": block_vz_high,
        "simple_under_xdot_zero_frac": FLAGS.simple_under_xdot_zero_frac,
        "simple_bar_xdot_zero_frac": bar_xzero_frac,
        "simple_block_xdot_zero_frac": block_xzero_frac,
        "simple_gap_vx_abs": FLAGS.simple_gap_vx_abs,
        "simple_gap_vz_low": FLAGS.simple_gap_vz_low,
        "simple_gap_vz_high": FLAGS.simple_gap_vz_high,
        "simple_low_uniform_vx_abs": FLAGS.simple_low_uniform_vx_abs,
        "simple_low_uniform_vz_abs": FLAGS.simple_low_uniform_vz_abs,
        "simple_fork_x_low": FLAGS.simple_fork_x_low,
        "simple_fork_x_high": FLAGS.simple_fork_x_high,
        "simple_fork_z_low": FLAGS.simple_fork_z_low,
        "simple_fork_z_high": FLAGS.simple_fork_z_high,
        "simple_fork_vx_low": FLAGS.simple_fork_vx_low,
        "simple_fork_vx_high": FLAGS.simple_fork_vx_high,
        "simple_fork_vz_low": FLAGS.simple_fork_vz_low,
        "simple_fork_vz_high": FLAGS.simple_fork_vz_high,
        "simple_fork_theta_abs": FLAGS.simple_fork_theta_abs,
        "simple_fork_omega_abs": FLAGS.simple_fork_omega_abs,
        "low_paper_jitter_vx_abs": FLAGS.low_paper_jitter_vx_abs,
        "low_paper_jitter_vz_abs": FLAGS.low_paper_jitter_vz_abs,
        "low_paper_pinpoint_vx_abs": FLAGS.low_paper_pinpoint_vx_abs,
        "low_paper_pinpoint_vz_abs": FLAGS.low_paper_pinpoint_vz_abs,
        "low_paper_pinpoint_theta_abs": FLAGS.low_paper_pinpoint_theta_abs,
        "low_paper_pinpoint_omega_abs": FLAGS.low_paper_pinpoint_omega_abs,
        "low_paper_pinpoint_x_abs": FLAGS.low_paper_pinpoint_x_abs,
        "low_paper_pinpoint_z_center": FLAGS.low_paper_pinpoint_z_center,
        "low_paper_pinpoint_z_halfwidth": FLAGS.low_paper_pinpoint_z_halfwidth,
        "low_under_bar_vx_abs": FLAGS.low_under_bar_vx_abs,
        "low_under_bar_vz_abs": FLAGS.low_under_bar_vz_abs,
        "low_under_blocks_vx_abs": FLAGS.low_under_blocks_vx_abs,
        "low_under_blocks_vz_abs": FLAGS.low_under_blocks_vz_abs,
        "low_inner_gaps_vx_abs": FLAGS.low_inner_gaps_vx_abs,
        "low_inner_gaps_vz_abs": FLAGS.low_inner_gaps_vz_abs,
        "eval_reset_mode": FLAGS.eval_reset_mode,
        "eval_suite": FLAGS.eval_suite,
        "init_h_threshold": FLAGS.init_h_threshold,
        "sdf_mode": FLAGS.sdf_mode,
        "sdf_unsafe_const": FLAGS.sdf_unsafe_const,
        "sdf_unsafe_slope": FLAGS.sdf_unsafe_slope,
        "sdf_global_scale": FLAGS.sdf_global_scale,
        "gate_lambda": FLAGS.gate_lambda,
        "gate_x_max": FLAGS.gate_x_max,
        "gate_z_center": FLAGS.gate_z_center,
        "gate_z_width": FLAGS.gate_z_width,
        "boundary_replay": FLAGS.boundary_replay,
        "boundary_mode": FLAGS.boundary_mode,
        "boundary_eps": FLAGS.boundary_eps,
        "boundary_frac": FLAGS.boundary_frac,
        "boundary_min_count": FLAGS.boundary_min_count,
        "hard_margin": FLAGS.hard_margin,
        "h_hardgap": FLAGS.h_hardgap,
        "use_action_margin": FLAGS.use_action_margin,
        "mirror_augment": FLAGS.mirror_augment,
        "init_checkpoint": FLAGS.init_checkpoint,
        "fixed_eval_starts": FLAGS.fixed_eval_starts,
    }


def make_agent(env: Quad2DStabEnv):
    if FLAGS.h_hardgap != 0.0 or FLAGS.use_action_margin:
        raise ValueError("This historical example requires h_hardgap=0 and use_action_margin=false.")
    SSMOnlineAgent = get_agent_class(FLAGS.agent)
    kwargs = dict(
        seed=FLAGS.seed,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor_hidden_dims=parse_hidden_dims(FLAGS.actor_hidden),
        critic_hidden_dims=parse_hidden_dims(FLAGS.critic_hidden),
        actor_lr=FLAGS.actor_lr,
        critic_lr=FLAGS.critic_lr,
        safe_critic_lr=FLAGS.safe_critic_lr,
        safe_value_lr=FLAGS.safe_value_lr,
        M_q=FLAGS.M_q,
        alpha_r=FLAGS.alpha_r,
        beta_safety=FLAGS.beta_safety,
        delta=FLAGS.delta,
        discount=FLAGS.discount,
        discount_h=FLAGS.discount_h,
        tau=FLAGS.tau,
        T=FLAGS.T,
        beta_schedule=FLAGS.beta_schedule,
        cost_critic_hyperparam=FLAGS.cost_critic_hyperparam,
        safety_estimator="sampled_qh",
        pure_qsm=FLAGS.pure_qsm,
        ddpm_temperature=FLAGS.ddpm_temperature,
        fuse_clip_min=FLAGS.fuse_clip_min,
        fuse_clip_max=FLAGS.fuse_clip_max,
        huber_delta=FLAGS.huber_delta,
        hard_margin=FLAGS.hard_margin,
    )
    if FLAGS.agent == "dzfill_adaptive":
        kwargs.update(
            beta_max=FLAGS.beta_max,
            beta_init=FLAGS.beta_init,
            beta_reg=FLAGS.beta_reg,
            beta_lr=FLAGS.beta_lr,
            adaptive_beta=True,
        )
    elif FLAGS.agent == "unified":
        kwargs.update(
            beta_max=FLAGS.beta_max,
            beta_init=FLAGS.beta_init,
            beta_reg=FLAGS.beta_reg,
            beta_lr=FLAGS.beta_lr,
            adaptive_beta=FLAGS.adaptive_beta,
        )
    elif FLAGS.agent in ("dzfill_qhstage", "dzfill_qhstage_v_2", "dzfill_qhstage_v2"):
        kwargs.update(
            stage_mode=FLAGS.qh_stage_mode,
            qh_candidate_source=FLAGS.qh_candidate_source,
            qh_min_k_samples=FLAGS.qh_min_k_samples,
            qh_min_k_sigma=FLAGS.qh_min_k_sigma,
        )
    return SSMOnlineAgent.create(**kwargs)


def save_checkpoint(agent, env: Quad2DStabEnv, path: str) -> None:
    params = {}
    for name in (
        "score_model",
        "critic_1",
        "critic_2",
        "target_critic_1",
        "target_critic_2",
        "safe_critic",
        "safe_target_critic",
        "safe_value",
        "safe_target_value",
        "beta_net",
    ):
        if hasattr(agent, name):
            params[name] = jax.device_get(getattr(agent, name).params)
    data = {
        "params": params,
        "agent": FLAGS.agent,
        "env": "Quad2DStab",
        "rng": jax.device_get(agent.rng),
        "obs_mean": env._obs_mean.copy(),
        "obs_var": env._obs_var.copy(),
        "obs_count": env._obs_count,
        "config": build_run_config(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  checkpoint saved: {path}")


def warm_start_from_checkpoint(agent, env: Quad2DStabEnv):
    if not FLAGS.init_checkpoint:
        return agent
    with open(FLAGS.init_checkpoint, "rb") as f:
        data = pickle.load(f)

    obs_mean = np.asarray(data.get("obs_mean", np.zeros(env.obs_dim)), dtype=np.float64)
    obs_var = np.asarray(data.get("obs_var", np.ones(env.obs_dim)), dtype=np.float64)
    if obs_mean.shape != (env.obs_dim,) or obs_var.shape != (env.obs_dim,):
        raise ValueError(
            f"init_checkpoint obs stats shape mismatch: ckpt mean={obs_mean.shape}, "
            f"var={obs_var.shape}, env obs_dim={env.obs_dim}"
        )
    env.set_obs_stats(obs_mean, obs_var, int(data.get("obs_count", 0)))
    env.set_train_mode()

    replacements = {}
    loaded_names = []
    for name, value in data.get("params", {}).items():
        if hasattr(agent, name) and hasattr(getattr(agent, name), "replace"):
            replacements[name] = getattr(agent, name).replace(params=value)
            loaded_names.append(name)
    if not replacements:
        raise ValueError(f"init_checkpoint {FLAGS.init_checkpoint!r} did not contain loadable params.")
    agent = agent.replace(**replacements)
    print(
        f"  warm-start loaded {len(loaded_names)} modules from {FLAGS.init_checkpoint}; "
        "optimizer state is freshly initialized"
    )
    print(f"  warm-start modules={','.join(loaded_names)}")
    return agent


FIXED_STARTS = [
    ("paper_center", FIXED_PAPER_START),
    ("paper_left", {**FIXED_PAPER_START, "init_x": -0.12}),
    ("paper_right", {**FIXED_PAPER_START, "init_x": 0.12}),
    ("near_goal", {"init_x": 0.0, "init_vx": 0.0, "init_z": 2.18, "init_vz": 0.0, "init_theta": 0.0, "init_omega": 0.0}),
]


def selected_fixed_starts() -> List[Tuple[str, Dict[str, float]]]:
    requested = [name.strip() for name in FLAGS.fixed_eval_starts.split(",") if name.strip()]
    if not requested:
        return []
    by_name = dict(FIXED_STARTS)
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown fixed_eval_starts labels: {unknown}; valid={list(by_name)}")
    return [(name, by_name[name]) for name in requested]


def side_entropy_norm(left_frac: float, right_frac: float) -> float:
    total = float(left_frac + right_frac)
    if total <= 0.0:
        return 0.0
    p = float(left_frac / total)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0))


def obs_from_raw_state_with_current_stats(env: Quad2DStabEnv, state: np.ndarray) -> np.ndarray:
    raw = env._raw_obs(np.asarray(state, dtype=np.float32))
    mean, var, _count = env.get_obs_stats()
    normed = (raw.astype(np.float64) - mean) / np.sqrt(var + 1e-8)
    return np.clip(normed, -env._obs_norm_clip, env._obs_norm_clip).astype(np.float32)


def insert_transition(
    replay_buffer: SimpleReplayBuffer,
    env: Quad2DStabEnv,
    raw_state: np.ndarray,
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    h_val: float,
    h_phys: float,
    mask: float,
    next_raw_state: np.ndarray,
    next_obs: np.ndarray,
) -> None:
    replay_buffer.insert(
        dict(
            observations=obs,
            actions=action,
            rewards=reward,
            costs=h_val,
            h_phys=h_phys,
            masks=mask,
            next_observations=next_obs,
        )
    )
    if not FLAGS.mirror_augment:
        return
    obs_m = obs_from_raw_state_with_current_stats(env, mirror_state(raw_state))
    next_obs_m = obs_from_raw_state_with_current_stats(env, mirror_state(next_raw_state))
    replay_buffer.insert(
        dict(
            observations=obs_m,
            actions=mirror_action(action),
            rewards=reward,
            costs=h_val,
            h_phys=h_phys,
            masks=mask,
            next_observations=next_obs_m,
        )
    )


def _eval_one_episode(agent, env: Quad2DStabEnv, obs: np.ndarray, seed_offset: int):
    ep_agent = agent.replace(rng=jax.random.fold_in(agent.rng, seed_offset))
    done = False
    last_info = {}
    while not done:
        action, ep_agent = ep_agent.eval_actions(obs)
        obs, _reward, _h, _binary_cost, done, last_info = env.step(np.asarray(action))
    return env.episode_info, last_info


def evaluate_agent(agent, train_env: Quad2DStabEnv, num_episodes: int = 20) -> Dict[str, float]:
    if FLAGS.eval_backend == "p123":
        from ssm.tasks.quad2d_stabilize.evaluation import evaluate_agent_stab, evaluate_agent_stab_suite

        mean, var, _count = train_env.get_obs_stats()
        if FLAGS.eval_suite != "default":
            return evaluate_agent_stab_suite(
                agent=agent,
                config=build_run_config(),
                obs_mean=mean,
                obs_var=var,
                episodes=num_episodes,
                init_seed=9999,
                policy_seed=999000,
                suite=FLAGS.eval_suite,
                prefix="eval",
                rollout_backend="p123",
            )
        return evaluate_agent_stab(
            agent=agent,
            config=build_run_config(),
            obs_mean=mean,
            obs_var=var,
            episodes=num_episodes,
            init_seed=9999,
            policy_seed=999000,
            random_starts=True,
            prefix="eval",
            rollout_backend="p123",
        )

    if FLAGS.eval_suite != "default":
        from ssm.tasks.quad2d_stabilize.evaluation import evaluate_agent_stab_suite

        mean, var, _count = train_env.get_obs_stats()
        return evaluate_agent_stab_suite(
            agent=agent,
            config=build_run_config(),
            obs_mean=mean,
            obs_var=var,
            episodes=num_episodes,
            init_seed=9999,
            policy_seed=999000,
            suite=FLAGS.eval_suite,
            prefix="eval",
            rollout_backend="python",
        )

    eval_env = make_env(seed=9999, for_eval=True)
    mean, var, count = train_env.get_obs_stats()
    eval_env.set_obs_stats(mean, var, count)
    eval_env.set_eval_mode()

    rewards, costs, lengths = [], [], []
    successes, crashes, dwell_fracs = [], [], []
    terminal_pos_errors, terminal_vel_norms, terminal_state_norms, terminal_action_error_norms = [], [], [], []
    for ep in range(num_episodes):
        obs = eval_env.reset()
        ep_info, last_info = _eval_one_episode(agent, eval_env, obs, 999000 + ep)
        rewards.append(ep_info["reward"])
        costs.append(ep_info["cost"])
        lengths.append(ep_info["length"])
        successes.append(ep_info["success"])
        crashes.append(1.0 if last_info.get("collision", False) else 0.0)
        dwell_fracs.append(ep_info["goal_dwell_frac"])
        terminal_pos_errors.append(last_info.get("pos_error", np.nan))
        terminal_vel_norms.append(last_info.get("vel_norm", np.nan))
        terminal_state_norms.append(ep_info.get("terminal_state_norm_to_ref", np.nan))
        terminal_action_error_norms.append(ep_info.get("terminal_action_error_norm", np.nan))

    return {
        "eval/reward_mean": float(np.mean(rewards)),
        "eval/reward_std": float(np.std(rewards)),
        "eval/cost_mean": float(np.mean(costs)),
        "eval/length_mean": float(np.mean(lengths)),
        "eval/success_frac": float(np.mean(successes)),
        "eval/crash_frac": float(np.mean(crashes)),
        "eval/goal_dwell_frac": float(np.mean(dwell_fracs)),
        "eval/terminal_state_norm_to_ref": float(np.nanmean(terminal_state_norms)),
        "eval/terminal_state_error_norm": float(np.nanmean(terminal_state_norms)),
        "eval/terminal_action_error_norm": float(np.nanmean(terminal_action_error_norms)),
        "eval/terminal_pos_error": float(np.nanmean(terminal_pos_errors)),
        "eval/terminal_vel_norm": float(np.nanmean(terminal_vel_norms)),
    }


def evaluate_fixed_starts(agent, train_env: Quad2DStabEnv, repeats: int = 10) -> Dict[str, float]:
    if repeats <= 0:
        return {}

    if FLAGS.eval_backend == "p123":
        from ssm.tasks.quad2d_stabilize.evaluation import evaluate_agent_stab

        mean, var, _count = train_env.get_obs_stats()
        config = build_run_config()
        all_rewards, all_costs, all_successes, all_crashes = [], [], [], []
        all_terminal_norms, all_terminal_action_errors = [], []
        result: Dict[str, float] = {}
        for idx, (label, _opts) in enumerate(selected_fixed_starts()):
            metrics = evaluate_agent_stab(
                agent=agent,
                config=config,
                obs_mean=mean,
                obs_var=var,
                episodes=repeats,
                init_seed=8888,
                policy_seed=888000 + idx * 100,
                start=label,
                random_starts=False,
                prefix="eval_tmp",
                rollout_backend="p123",
            )
            reward = float(metrics["eval_tmp/reward_mean"])
            cost = float(metrics["eval_tmp/cost_mean"])
            success = float(metrics["eval_tmp/success_frac"])
            crash = float(metrics["eval_tmp/crash_frac"])
            result[f"eval_fixed/{label}_reward_mean"] = reward
            result[f"eval_fixed/{label}_cost_mean"] = cost
            result[f"eval_fixed/{label}_success_frac"] = success
            result[f"eval_fixed/{label}_crash_frac"] = crash
            result[f"eval_fixed/{label}_terminal_pos_error"] = float(metrics["eval_tmp/terminal_pos_error"])
            result[f"eval_fixed/{label}_terminal_state_norm_to_ref"] = float(
                metrics["eval_tmp/terminal_state_norm_to_ref"]
            )
            result[f"eval_fixed/{label}_terminal_state_error_norm"] = float(
                metrics["eval_tmp/terminal_state_error_norm"]
            )
            result[f"eval_fixed/{label}_terminal_action_error_norm"] = float(
                metrics["eval_tmp/terminal_action_error_norm"]
            )
            route_left = float(metrics.get("eval_tmp/route_left_frac", np.nan))
            route_right = float(metrics.get("eval_tmp/route_right_frac", np.nan))
            succ_route_left = float(metrics.get("eval_tmp/success_route_left_frac", np.nan))
            succ_route_right = float(metrics.get("eval_tmp/success_route_right_frac", np.nan))
            result[f"eval_fixed/{label}_route_left_frac"] = route_left
            result[f"eval_fixed/{label}_route_right_frac"] = route_right
            result[f"eval_fixed/{label}_side_mode_coverage"] = float(metrics.get("eval_tmp/side_mode_coverage", np.nan))
            result[f"eval_fixed/{label}_success_route_left_frac"] = succ_route_left
            result[f"eval_fixed/{label}_success_route_right_frac"] = succ_route_right
            result[f"eval_fixed/{label}_side_mode_coverage_success"] = float(
                metrics.get("eval_tmp/side_mode_coverage_success", np.nan)
            )
            result[f"eval_fixed/{label}_route_split_entropy_norm"] = side_entropy_norm(succ_route_left, succ_route_right)
            if label == "paper_center":
                result["eval_fixed/route_left_frac"] = route_left
                result["eval_fixed/route_right_frac"] = route_right
                result["eval_fixed/success_route_left_frac"] = succ_route_left
                result["eval_fixed/success_route_right_frac"] = succ_route_right
                result["eval_fixed/side_mode_coverage_success"] = result[
                    f"eval_fixed/{label}_side_mode_coverage_success"
                ]
                result["eval_fixed/route_split_entropy_norm"] = result[
                    f"eval_fixed/{label}_route_split_entropy_norm"
                ]
            all_rewards.append(reward)
            all_costs.append(cost)
            all_successes.append(success)
            all_crashes.append(crash)
            all_terminal_norms.append(float(metrics["eval_tmp/terminal_state_norm_to_ref"]))
            all_terminal_action_errors.append(float(metrics["eval_tmp/terminal_action_error_norm"]))
        result.update(
            {
                "eval_fixed/reward_mean": float(np.mean(all_rewards)),
                "eval_fixed/cost_mean": float(np.mean(all_costs)),
                "eval_fixed/success_frac": float(np.mean(all_successes)),
                "eval_fixed/crash_frac": float(np.mean(all_crashes)),
                "eval_fixed/terminal_state_norm_to_ref": float(np.nanmean(all_terminal_norms)),
                "eval_fixed/terminal_state_error_norm": float(np.nanmean(all_terminal_norms)),
                "eval_fixed/terminal_action_error_norm": float(np.nanmean(all_terminal_action_errors)),
            }
        )
        return result

    eval_env = make_env(seed=8888, for_eval=True)
    mean, var, count = train_env.get_obs_stats()
    eval_env.set_obs_stats(mean, var, count)
    eval_env.set_eval_mode()

    all_rewards, all_costs, all_successes, all_crashes = [], [], [], []
    all_terminal_norms, all_terminal_action_errors = [], []
    result: Dict[str, float] = {}
    for idx, (label, opts) in enumerate(selected_fixed_starts()):
        rewards, costs, successes, crashes, pos_errors = [], [], [], [], []
        terminal_norms, terminal_action_errors = [], []
        for rep in range(repeats):
            obs = eval_env.reset(options=opts)
            ep_info, last_info = _eval_one_episode(agent, eval_env, obs, 888000 + idx * 100 + rep)
            rewards.append(ep_info["reward"])
            costs.append(ep_info["cost"])
            successes.append(ep_info["success"])
            crashes.append(1.0 if last_info.get("collision", False) else 0.0)
            pos_errors.append(last_info.get("pos_error", np.nan))
            terminal_norms.append(ep_info.get("terminal_state_norm_to_ref", np.nan))
            terminal_action_errors.append(ep_info.get("terminal_action_error_norm", np.nan))
            all_rewards.append(ep_info["reward"])
            all_costs.append(ep_info["cost"])
            all_successes.append(ep_info["success"])
            all_crashes.append(1.0 if last_info.get("collision", False) else 0.0)
            all_terminal_norms.append(ep_info.get("terminal_state_norm_to_ref", np.nan))
            all_terminal_action_errors.append(ep_info.get("terminal_action_error_norm", np.nan))
        result[f"eval_fixed/{label}_reward_mean"] = float(np.mean(rewards))
        result[f"eval_fixed/{label}_cost_mean"] = float(np.mean(costs))
        result[f"eval_fixed/{label}_success_frac"] = float(np.mean(successes))
        result[f"eval_fixed/{label}_crash_frac"] = float(np.mean(crashes))
        result[f"eval_fixed/{label}_terminal_pos_error"] = float(np.nanmean(pos_errors))
        result[f"eval_fixed/{label}_terminal_state_norm_to_ref"] = float(np.nanmean(terminal_norms))
        result[f"eval_fixed/{label}_terminal_state_error_norm"] = float(np.nanmean(terminal_norms))
        result[f"eval_fixed/{label}_terminal_action_error_norm"] = float(np.nanmean(terminal_action_errors))

    result.update(
        {
            "eval_fixed/reward_mean": float(np.mean(all_rewards)),
            "eval_fixed/cost_mean": float(np.mean(all_costs)),
            "eval_fixed/success_frac": float(np.mean(all_successes)),
            "eval_fixed/crash_frac": float(np.mean(all_crashes)),
            "eval_fixed/terminal_state_norm_to_ref": float(np.nanmean(all_terminal_norms)),
            "eval_fixed/terminal_state_error_norm": float(np.nanmean(all_terminal_norms)),
            "eval_fixed/terminal_action_error_norm": float(np.nanmean(all_terminal_action_errors)),
        }
    )
    return result


def auto_run_name() -> str:
    if FLAGS.run_name:
        return FLAGS.run_name
    q_x, q_z = resolved_q_weights()
    q_part = f"qxz{q_x:g}_{q_z:g}" if q_axis_split_active() else f"qpv{FLAGS.q_pos:g}_{FLAGS.q_vel:g}_{FLAGS.q_angle:g}"
    parts = [
        "quad2d_stab",
        FLAGS.layout_name,
        "p123" if (FLAGS.env_backend == "p2" or FLAGS.eval_backend == "p123") else "",
        FLAGS.agent,
        f"{FLAGS.qh_stage_mode}_k{FLAGS.qh_min_k_samples}s{FLAGS.qh_min_k_sigma:g}"
        if FLAGS.agent in ("dzfill_qhstage", "dzfill_qhstage_v_2", "dzfill_qhstage_v2")
        else f"b{FLAGS.beta_safety:g}",
        q_part,
        f"qv{FLAGS.q_vel:g}_qa{FLAGS.q_angle:g}" if q_axis_split_active() else "",
        f"ap{FLAGS.action_penalty:g}",
        f"fs{FLAGS.success_radius:g}" if FLAGS.success_mode == "full_state_terminal" else f"dwell{FLAGS.goal_dwell_steps}",
        f"h{FLAGS.max_episode_steps}",
        f"mq{FLAGS.M_q:g}",
    ]
    if FLAGS.reset_mode != "start_band":
        parts.append(f"reset{FLAGS.reset_mode}")
    if FLAGS.reset_simple_fork_frac > 0.0:
        parts.append(f"fork{FLAGS.reset_simple_fork_frac:g}")
    if FLAGS.mirror_augment:
        parts.append("mirror")
    if FLAGS.obs_feature_mode != "state":
        parts.append(f"obs{FLAGS.obs_feature_mode}")
    if FLAGS.gate_lambda > 0.0:
        parts.append(f"gate{FLAGS.gate_lambda:g}_xm{FLAGS.gate_x_max:g}")
    if FLAGS.dwell_bonus > 0.0:
        parts.append(f"db{FLAGS.dwell_bonus:g}")
    if FLAGS.boundary_replay:
        parts.append(f"br{FLAGS.boundary_mode}_e{FLAGS.boundary_eps:g}_f{FLAGS.boundary_frac:g}")
    parts.append(f"s{FLAGS.seed}")
    return "_".join(p for p in parts if p)


def main(_):
    run_name = auto_run_name()
    env = make_env(seed=FLAGS.seed)
    q_x, q_z = resolved_q_weights()
    print(f"Env: Quad2DStab obs={env.observation_space.shape} act={env.action_space.shape}")
    print(
        f"  layout={FLAGS.layout_name}, horizon={FLAGS.max_episode_steps}, "
        f"goal={env.goal.tolist()}, reset={FLAGS.reset_mode}"
    )
    print(
        f"  reward q_pos={FLAGS.q_pos}, q_x={q_x}, q_z={q_z}, q_vel={FLAGS.q_vel}, q_angle={FLAGS.q_angle}, "
        f"action_penalty={FLAGS.action_penalty}, dwell_bonus={FLAGS.dwell_bonus}"
    )
    print(
        f"  success_mode={FLAGS.success_mode}, success_radius={FLAGS.success_radius}, "
        f"dwell={FLAGS.goal_dwell_steps}, terminate_on_success={FLAGS.terminate_on_success}"
    )
    print(
        "  low reset top="
        f"low/mid/near=({FLAGS.reset_low_frac:g},{FLAGS.reset_low_mid_frac:g},{FLAGS.reset_low_near_frac:g}) "
        "components="
        f"paper_jitter={FLAGS.reset_low_paper_jitter_frac:g}, "
        f"pinpoint_vel={FLAGS.reset_low_paper_pinpoint_vel_frac:g}, "
        f"under_bar={FLAGS.reset_low_under_bar_frac:g}, "
        f"under_blocks={FLAGS.reset_low_under_blocks_frac:g}, "
        f"inner_gaps={FLAGS.reset_low_inner_gaps_frac:g}, "
        f"low_global={FLAGS.reset_low_global_frac:g}"
    )
    print(
        "  low reset velocity="
        f"paper_jitter_v=({FLAGS.low_paper_jitter_vx_abs:g},{FLAGS.low_paper_jitter_vz_abs:g}), "
        f"pinpoint_v=({FLAGS.low_paper_pinpoint_vx_abs:g},{FLAGS.low_paper_pinpoint_vz_abs:g}), "
        f"under_bar_v=({FLAGS.low_under_bar_vx_abs:g},{FLAGS.low_under_bar_vz_abs:g}), "
        f"under_blocks_v=({FLAGS.low_under_blocks_vx_abs:g},{FLAGS.low_under_blocks_vz_abs:g}), "
        f"inner_gaps_v=({FLAGS.low_inner_gaps_vx_abs:g},{FLAGS.low_inner_gaps_vz_abs:g})"
    )
    bar_vz_low, bar_vz_high, block_vz_low, block_vz_high = resolved_simple_under_vz_ranges()
    bar_xzero_frac, block_xzero_frac = resolved_simple_under_xdot_zero_fracs()
    print(
        "  simple_under3 reset="
        f"bar/left/right/gap/low/near/side/fork=({FLAGS.reset_simple_bar_frac:g},"
        f"{FLAGS.reset_simple_left_block_frac:g},{FLAGS.reset_simple_right_block_frac:g},"
        f"{FLAGS.reset_simple_gap_frac:g},{FLAGS.reset_simple_low_uniform_frac:g},"
        f"{FLAGS.reset_simple_near_frac:g},{FLAGS.reset_simple_side_frac:g},{FLAGS.reset_simple_fork_frac:g}), "
        f"under_vx={FLAGS.simple_under_vx_abs:g}, "
        f"bar_vz=[{bar_vz_low:g},{bar_vz_high:g}], "
        f"block_vz=[{block_vz_low:g},{block_vz_high:g}], "
        f"xzero_bar/block=({bar_xzero_frac:g},{block_xzero_frac:g}), "
        f"gap_vx={FLAGS.simple_gap_vx_abs:g}, gap_vz=[{FLAGS.simple_gap_vz_low:g},{FLAGS.simple_gap_vz_high:g}], "
        f"low_uniform_v=({FLAGS.simple_low_uniform_vx_abs:g},{FLAGS.simple_low_uniform_vz_abs:g}), "
        f"fork_x=[{FLAGS.simple_fork_x_low:g},{FLAGS.simple_fork_x_high:g}], "
        f"fork_z=[{FLAGS.simple_fork_z_low:g},{FLAGS.simple_fork_z_high:g}]"
    )
    print(
        "  safety_under50 reset="
        f"bar/left/right/gap/low/mid/near=({FLAGS.reset_safety_bar_frac:g},"
        f"{FLAGS.reset_safety_left_block_frac:g},{FLAGS.reset_safety_right_block_frac:g},"
        f"{FLAGS.reset_safety_gap_frac:g},{FLAGS.reset_safety_low_uniform_frac:g},"
        f"{FLAGS.reset_safety_mid_uniform_frac:g},{FLAGS.reset_safety_near_frac:g})"
    )
    print(
        f"  safety routing delta={FLAGS.delta}, hard_margin={FLAGS.hard_margin}, "
        f"use_action_margin={FLAGS.use_action_margin}, h_hardgap={FLAGS.h_hardgap}, "
        f"init_h_threshold={FLAGS.init_h_threshold}"
    )
    print(
        f"  gate_lambda={FLAGS.gate_lambda}, gate_x_max={FLAGS.gate_x_max}, "
        f"gate_z_center={FLAGS.gate_z_center}, gate_z_width={FLAGS.gate_z_width}"
    )
    print(
        f"  obs_feature_mode={FLAGS.obs_feature_mode}, obs_dim={env.observation_space.shape[0]}, "
        f"obs_h_scale={FLAGS.obs_h_scale}, obs_hdot_scale={FLAGS.obs_hdot_scale}"
    )
    print(
        f"  lr actor/critic/safe_q/safe_v=({FLAGS.actor_lr:g},{FLAGS.critic_lr:g},"
        f"{FLAGS.safe_critic_lr:g},{FLAGS.safe_value_lr:g}), "
        f"mirror_augment={FLAGS.mirror_augment}, init_checkpoint={FLAGS.init_checkpoint or 'none'}, "
        f"fixed_eval_starts={FLAGS.fixed_eval_starts}"
    )

    agent = make_agent(env)
    agent = warm_start_from_checkpoint(agent, env)
    replay_buffer = SimpleReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=ACT_DIM,
        max_size=1_000_000,
        boundary_replay=FLAGS.boundary_replay,
        boundary_mode=FLAGS.boundary_mode,
        boundary_eps=FLAGS.boundary_eps,
        boundary_frac=FLAGS.boundary_frac,
        boundary_min_count=FLAGS.boundary_min_count,
    )
    replay_buffer.seed(FLAGS.seed + 123)

    if FLAGS.wandb:
        import wandb

        wandb.init(project=FLAGS.project_name, entity=FLAGS.entity, name=run_name, config=build_run_config())

    obs = env.reset()
    start_time = time.time()
    for step in tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm_bar):
        if step < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(obs)
            action = np.asarray(action)

        raw_state = env.state.copy()
        next_obs, reward, h_val, _binary_cost, done, info = env.step(action)
        next_raw_state = env.state.copy()
        success_cuts_bootstrap = FLAGS.success_mode == "dwell_box" and info.get("success_now", False)
        mask = 0.0 if (info.get("collision", False) or success_cuts_bootstrap) else 1.0
        insert_transition(
            replay_buffer,
            env,
            raw_state,
            obs,
            action,
            reward,
            h_val,
            info["h_phys"],
            mask,
            next_raw_state,
            next_obs,
        )
        obs = next_obs

        if done:
            ep_info = env.episode_info
            log_dict = {
                "train/episode_reward": ep_info["reward"],
                "train/episode_cost": ep_info["cost"],
                "train/episode_length": ep_info["length"],
                "train/success": ep_info["success"],
                "train/collision": float(info.get("collision", False)),
                "train/goal_dwell_frac": ep_info["goal_dwell_frac"],
                "train/max_goal_streak": ep_info["max_goal_streak"],
            }
            if FLAGS.wandb:
                import wandb

                wandb.log(log_dict, step=step)
            obs = env.reset()

        if step >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            grad_step = step - FLAGS.start_training
            if FLAGS.actor_delay <= 1 or grad_step % FLAGS.actor_delay == 0:
                agent, update_info = agent.update(batch)
            else:
                agent, update_info = agent.update_critic_only(batch)

            if step % FLAGS.log_interval == 0:
                log_dict = {f"train/{k}": float(v) for k, v in update_info.items()}
                log_dict["train/step"] = step
                log_dict["train/sps"] = step / (time.time() - start_time)
                log_dict.update({f"replay/{k}": float(v) for k, v in replay_buffer.last_sample_info.items()})
                if FLAGS.wandb:
                    import wandb

                    wandb.log(log_dict, step=step)
                else:
                    print(
                        f"[{step}] actor={float(update_info.get('actor_loss', 0.0)):.3f} "
                        f"sf={float(update_info.get('safe_frac', 0.0)):.3f} "
                        f"vh={float(update_info.get('vh_mean', 0.0)):.3f} "
                        f"br={replay_buffer.last_sample_info['boundary_frac_in_batch']:.3f}"
                    )

        if step >= FLAGS.start_training and step % FLAGS.eval_interval == 0:
            eval_info = evaluate_agent(agent, env, FLAGS.eval_episodes)
            fixed_info = evaluate_fixed_starts(agent, env, FLAGS.fixed_eval_repeats)
            if FLAGS.wandb:
                import wandb

                wandb.log(eval_info, step=step)
                if fixed_info:
                    wandb.log(fixed_info, step=step)
            ckpt_dir = f"checkpoints/{run_name}"
            save_checkpoint(agent, env, f"{ckpt_dir}/step_{step}.pkl")
            fixed_summary = (
                f"| Fixed Succ={fixed_info['eval_fixed/success_frac']:.2f} "
                f"Crash={fixed_info['eval_fixed/crash_frac']:.2f} "
                f"Cov={fixed_info.get('eval_fixed/side_mode_coverage_success', float('nan')):.2f} "
                f"H={fixed_info.get('eval_fixed/route_split_entropy_norm', float('nan')):.2f}"
                if fixed_info
                else "| Fixed skipped"
            )
            print(
                f"\n[Eval @ {step}] "
                f"R={eval_info['eval/reward_mean']:.0f} "
                f"C={eval_info['eval/cost_mean']:.1f} "
                f"Succ={eval_info['eval/success_frac']:.2f} "
                f"Crash={eval_info['eval/crash_frac']:.2f} "
                f"Comp={eval_info.get('eval/composite_score', float('nan')):.2f} "
                f"Safe={eval_info.get('eval/safety_score', float('nan')):.2f} "
                f"P={eval_info.get('eval/paper_success_frac', float('nan')):.2f} "
                f"L={eval_info.get('eval/low_success_frac', float('nan')):.2f} "
                f"{fixed_summary}\n"
            )

    print("Training complete.")
    if FLAGS.wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    app.run(main)

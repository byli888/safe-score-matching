#!/usr/bin/env python3
"""Train posterior-score SSM on velocity-constrained MuJoCo tasks."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

from swimmer_velocity_bootstrap import install as _install_swimmer_velocity

_install_swimmer_velocity()

import jax
import jax.numpy as jnp
import numpy as np

from lp_ps_bridge_mis import BRIDGE_COMPONENT_WEIGHTINGS
from lp_ps_checkpoint import (
    load_checkpoint_payload,
    load_lp_checkpoint,
    save_lp_checkpoint,
)
from lp_ps_accel import (
    BEHAVIOR_GUARD_KEYS,
    BEHAVIOR_INFO_KEYS,
    CRITIC_GUARD_KEYS,
    UPDATE_GUARD_KEYS,
    GuardStatus,
    joint_paired_update_with_guard,
    pack_behavior_guard,
    pack_update_guard,
    sample_paired_replay_batches,
)
from swimmer_role_particles.particle_lp_ps_ssm_agent import (
    LP_ACTOR_SAFETY_MODES,
    LP_ACTOR_SAFETY_QH_ROUTED,
    LP_ACTOR_SAFETY_REWARD_ONLY,
    LP_ACTOR_COORDINATE,
    LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE,
    LP_LEGACY_AUTO_PROPOSAL_MODE,
    LP_TARGET_ESTIMATOR,
    LP_PARTICLE_MODE_S1_MIN,
    LP_PARTICLE_HJ_FULL_CONSISTENT,
    LP_PARTICLE_MODES,
    LP_REWARD_BACKUP_MODES,
    LP_REWARD_BACKUP_ONLINE_K1,
    LP_REWARD_TILT_FIXED_ALPHA,
    LP_REWARD_TILT_KL_BUDGET,
    LP_REWARD_TILT_BATCH_KL,
    LP_REWARD_TILT_CAPPED_BATCH_KL,
    LP_REWARD_TILT_MODES,
    PARTICLE_BEHAVIOR_INFO_KEYS,
    LPPSAgent,
    lp_proposal_mode,
)
from mujoco_velocity import make_velocity_env
from safe_replay_buffer import SafeReplayBuffer
from ssm_agent_v51_hard_dzfill_qhstage_v_2 import SSMOnlineAgent


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}.")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="SafetyHalfCheetahVelocity-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--critic-start-training", type=int, default=10_000)
    parser.add_argument("--start-training", type=int, default=20_000)
    parser.add_argument(
        "--obs-stats-calibration-steps",
        type=int,
        default=0,
        help=(
            "Random-policy calibration interactions before observation "
            "statistics are frozen and stationary replay collection begins. "
            "Zero preserves the historical continuously-updated behavior."
        ),
    )
    parser.add_argument("--critic-batch-size", type=int, default=256)
    parser.add_argument("--actor-batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=int, default=20_000)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--particle-mode",
        choices=LP_PARTICLE_MODES,
        default=LP_PARTICLE_MODE_S1_MIN,
        help=(
            "s1_min uses raw K=1 roles; role_separated uses N=8 for data, "
            "reward TD and actor outer while retaining an independent "
            "policy+Gaussian HJ set; full_consistent also replaces the HJ set."
        ),
    )
    parser.add_argument("--num-particles-data", type=int, default=1)
    parser.add_argument("--num-particles-target", type=int, default=1)
    parser.add_argument("--num-particles-train", type=int, default=1)
    parser.add_argument(
        "--data-particle-ess-fraction", type=float, default=0.5
    )
    parser.add_argument(
        "--target-particle-ess-fraction", type=float, default=0.5
    )
    parser.add_argument(
        "--train-particle-ess-fraction", type=float, default=0.5
    )

    parser.add_argument("--actor-hidden", default="256,256,256")
    parser.add_argument("--critic-hidden", default="256,256")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--safe-critic-lr", type=float, default=3e-4)
    parser.add_argument("--actor-grad-clip", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--ddpm-temperature", type=float, default=0.2)
    parser.add_argument("--posterior-samples", type=int, default=64)
    parser.add_argument("--proposal-chunk-size", type=int, default=64)
    parser.add_argument("--likelihood-fraction", type=float, default=1.0)
    parser.add_argument(
        "--proposal-family",
        choices=(
            LP_LEGACY_AUTO_PROPOSAL_MODE,
            LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE,
        ),
        default=LP_LEGACY_AUTO_PROPOSAL_MODE,
    )
    parser.add_argument("--bridge-likelihood-fraction", type=float)
    parser.add_argument("--bridge-reference-samples", type=int)
    parser.add_argument("--bridge-latent-std", type=float)
    parser.add_argument(
        "--bridge-component-weighting",
        choices=BRIDGE_COMPONENT_WEIGHTINGS,
    )
    parser.add_argument("--epsilon-action", type=float, default=1e-5)
    parser.add_argument("--alpha-reward", type=float, default=1.0)
    parser.add_argument("--beta-safety", type=float, default=3.0)
    parser.add_argument(
        "--reward-tilt-mode",
        choices=LP_REWARD_TILT_MODES,
        default=LP_REWARD_TILT_FIXED_ALPHA,
    )
    parser.add_argument("--reward-kl-delta", type=float, default=0.25)
    parser.add_argument("--reward-row-kl-cap", type=float, default=1.0)
    parser.add_argument("--safety-threshold", type=float, default=0.0)
    parser.add_argument(
        "--actor-safety-mode",
        choices=LP_ACTOR_SAFETY_MODES,
        default=LP_ACTOR_SAFETY_QH_ROUTED,
        help=(
            "qh_routed uses Q_h for branch routing/masking; reward_only "
            "does not evaluate Q_h in the actor update"
        ),
    )
    parser.add_argument(
        "--qh-critic-mode",
        choices=("single", "twin_head0", "twin_max"),
        default="twin_max",
    )
    parser.add_argument(
        "--eta",
        "--rho-anchor",
        dest="eta",
        type=float,
        default=0.5,
        help=(
            "proximal reference/posterior interpolation; --eta is retained "
            "as the checkpoint-compatible legacy spelling"
        ),
    )
    parser.add_argument("--reference-refresh-interval", type=int, default=500)
    parser.add_argument(
        "--reward-backup-mode",
        choices=LP_REWARD_BACKUP_MODES,
        default=LP_REWARD_BACKUP_ONLINE_K1,
    )
    parser.add_argument("--reward-backup-samples", type=int, default=1)
    parser.add_argument(
        "--qh-backup-uses-reference", type=_bool_arg, default=False
    )

    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--discount-h", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--hard-margin", type=float, default=0.0)
    parser.add_argument(
        "--h-hardgap",
        type=float,
        required=True,
        help=(
            "Task-specific safety gap: 0.5 for HalfCheetah, 0.0 for Swimmer; "
            "supplied by the task YAML."
        ),
    )
    parser.add_argument("--qh-k", type=int, default=8)
    parser.add_argument("--qh-sigma", type=float, default=0.3)
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--fuse-clip-min", type=float, default=-200.0)
    parser.add_argument("--fuse-clip-max", type=float, default=1000.0)

    parser.add_argument("--invalid-warning-threshold", type=float, default=0.05)
    parser.add_argument("--invalid-stop-threshold", type=float, default=0.10)
    parser.add_argument("--invalid-window", type=int, default=10)

    parser.add_argument("--wandb", type=_bool_arg, default=False)
    parser.add_argument("--wandb-project", default="safe-score-matching")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-group", default=None
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the locked config without creating artifacts.",
    )
    return parser.parse_args()


def _parse_hidden(value: str) -> tuple[int, ...]:
    hidden = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError(f"Invalid hidden dimensions: {value!r}.")
    return hidden


def _strict_warmup_action(action: Any) -> np.ndarray:
    """Keep Box-sampled warmup actions in the mathematically open box.

    Gym's continuous sampler is normally upper-exclusive.  The endpoint guard
    handles implementation/version edge cases without shrinking the warmup
    distribution or hiding an out-of-range learned policy action.
    """

    value = np.asarray(action, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("Warmup sampler produced a non-finite action.")
    upper = np.nextafter(np.float32(1.0), np.float32(0.0))
    lower = np.nextafter(np.float32(-1.0), np.float32(0.0))
    return np.clip(value, lower, upper)


def _validate_args(args: argparse.Namespace) -> None:
    supported_envs = {
        "SafetyHalfCheetahVelocity-v1",
        "SafetySwimmerVelocity-v1",
    }
    if args.env_id not in supported_envs:
        raise ValueError(
            "The LP-PS velocity runner only supports "
            f"non-terminating v1 tasks {sorted(supported_envs)}, got "
            f"{args.env_id!r}."
        )
    particle_counts = (
        args.num_particles_data,
        args.num_particles_target,
        args.num_particles_train,
    )
    if args.particle_mode == LP_PARTICLE_MODE_S1_MIN:
        if particle_counts != (1, 1, 1):
            raise ValueError(
                "s1_min requires N_data=N_target=N_train=1."
            )
    elif particle_counts != (8, 8, 8):
        raise ValueError(
            f"{args.particle_mode} requires N_data=N_target=N_train=8."
        )
    for name in (
        "data_particle_ess_fraction",
        "target_particle_ess_fraction",
        "train_particle_ess_fraction",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must lie in (0,1].")
    positive = {
        "max_steps": args.max_steps,
        "critic_start_training": args.critic_start_training,
        "start_training": args.start_training,
        "critic_batch_size": args.critic_batch_size,
        "actor_batch_size": args.actor_batch_size,
        "replay_size": args.replay_size,
        "updates_per_step": args.updates_per_step,
        "log_interval": args.log_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "posterior_samples": args.posterior_samples,
        "proposal_chunk_size": args.proposal_chunk_size,
        "reference_refresh_interval": args.reference_refresh_interval,
        "reward_backup_samples": args.reward_backup_samples,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if args.posterior_samples % args.proposal_chunk_size:
        raise ValueError("posterior_samples must be divisible by proposal_chunk_size.")
    if args.critic_start_training > args.start_training:
        raise ValueError(
            "critic_start_training must be <= actor start_training."
        )
    if args.obs_stats_calibration_steps < 0:
        raise ValueError("obs_stats_calibration_steps must be nonnegative.")
    if args.reward_backup_mode == LP_REWARD_BACKUP_ONLINE_K1:
        if args.reward_backup_samples != 1 or args.qh_backup_uses_reference:
            raise ValueError(
                "online_policy_k1 requires reward_backup_samples=1 and "
                "qh_backup_uses_reference=false."
            )
    if not 0.0 <= args.likelihood_fraction <= 1.0:
        raise ValueError("likelihood_fraction must lie in [0,1].")
    proposal_mode = lp_proposal_mode(
        args.likelihood_fraction,
        getattr(args, "proposal_family", LP_LEGACY_AUTO_PROPOSAL_MODE),
    )
    bridge_values = {
        "bridge_likelihood_fraction": getattr(
            args, "bridge_likelihood_fraction", None
        ),
        "bridge_reference_samples": getattr(
            args, "bridge_reference_samples", None
        ),
        "bridge_latent_std": getattr(args, "bridge_latent_std", None),
        "bridge_component_weighting": getattr(
            args, "bridge_component_weighting", None
        ),
    }
    if proposal_mode == LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE:
        if args.likelihood_fraction != 1.0:
            raise ValueError(
                "Bridge mode requires legacy likelihood_fraction=1.0; use "
                "bridge_likelihood_fraction for proposal omega."
            )
        missing = [name for name, value in bridge_values.items() if value is None]
        if missing:
            raise ValueError(
                "Bridge proposal requires explicit values for "
                + ", ".join(missing)
                + "."
            )
        if not 0.0 <= args.bridge_likelihood_fraction <= 1.0:
            raise ValueError("bridge_likelihood_fraction must lie in [0,1].")
        if args.bridge_reference_samples <= 0:
            raise ValueError("bridge_reference_samples must be positive.")
        if args.bridge_latent_std <= 0.0:
            raise ValueError("bridge_latent_std must be positive.")
    elif any(value is not None for value in bridge_values.values()):
        raise ValueError(
            "Bridge-only arguments are invalid unless proposal_family is "
            f"{LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE!r}."
        )
    if not 0.0 <= args.eta <= 1.0:
        raise ValueError("eta must lie in [0,1].")
    if args.alpha_reward <= 0.0 or args.beta_safety <= 0.0:
        raise ValueError("alpha_reward and beta_safety must be positive.")
    if args.reward_kl_delta < 0.0 or not math.isfinite(args.reward_kl_delta):
        raise ValueError("reward_kl_delta must be finite and non-negative.")
    if args.reward_row_kl_cap < 0.0 or not math.isfinite(
        args.reward_row_kl_cap
    ):
        raise ValueError(
            "reward_row_kl_cap must be finite and non-negative."
        )
    if (
        args.reward_tilt_mode == LP_REWARD_TILT_CAPPED_BATCH_KL
        and args.reward_row_kl_cap < args.reward_kl_delta
    ):
        raise ValueError(
            "capped_batch_kl requires reward_row_kl_cap >= reward_kl_delta."
        )
    if not 0.0 < args.epsilon_action < 0.01:
        raise ValueError("epsilon_action must lie in (0,0.01).")
    output = args.output_root.resolve()
    if output == Path("/"):
        raise ValueError("output_root cannot be the filesystem root")


def make_agent(args: argparse.Namespace, env) -> LPPSAgent:
    qh_critic_mode = getattr(args, "qh_critic_mode", "twin_max")
    actor_safety_mode = getattr(
        args, "actor_safety_mode", LP_ACTOR_SAFETY_QH_ROUTED
    )
    core = SSMOnlineAgent.create(
        seed=args.seed,
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor_hidden_dims=_parse_hidden(args.actor_hidden),
        critic_hidden_dims=_parse_hidden(args.critic_hidden),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        safe_critic_lr=args.safe_critic_lr,
        safe_value_lr=args.safe_critic_lr,
        discount=args.discount,
        discount_h=args.discount_h,
        tau=args.tau,
        M_q=1.0,
        alpha_r=args.alpha_reward,
        beta_safety=args.beta_safety,
        delta=args.delta,
        hard_margin=args.hard_margin,
        h_hardgap=args.h_hardgap,
        fuse_clip_min=args.fuse_clip_min,
        fuse_clip_max=args.fuse_clip_max,
        huber_delta=args.huber_delta,
        stage_mode="stage_b",
        qh_critic_mode=qh_critic_mode,
        qh_candidate_source="policy_plus_gaussian",
        qh_min_k_samples=args.qh_k,
        qh_min_k_sigma=args.qh_sigma,
        T=args.T,
        clip_sampler=False,
        ddpm_temperature=args.ddpm_temperature,
        beta_schedule="vp",
        decay_steps=None,
        use_action_margin=False,
    )
    return LPPSAgent.from_core(
        core,
        actor_lr=args.actor_lr,
        actor_grad_clip_norm=args.actor_grad_clip,
        posterior_samples=args.posterior_samples,
        proposal_chunk_size=args.proposal_chunk_size,
        likelihood_fraction=args.likelihood_fraction,
        proposal_mode=getattr(
            args, "proposal_family", LP_LEGACY_AUTO_PROPOSAL_MODE
        ),
        bridge_likelihood_fraction=(
            getattr(args, "bridge_likelihood_fraction", None)
            if getattr(args, "bridge_likelihood_fraction", None) is not None
            else 0.5
        ),
        bridge_reference_samples=(
            getattr(args, "bridge_reference_samples", None) or 0
        ),
        bridge_latent_std=(getattr(args, "bridge_latent_std", None) or 0.0),
        bridge_component_weighting=(
            getattr(args, "bridge_component_weighting", None)
            or BRIDGE_COMPONENT_WEIGHTINGS[0]
        ),
        eta=args.eta,
        reference_refresh_interval=args.reference_refresh_interval,
        safety_threshold=args.safety_threshold,
        epsilon_action=args.epsilon_action,
        actor_safety_mode=actor_safety_mode,
        particle_mode=args.particle_mode,
        num_particles_data=args.num_particles_data,
        num_particles_target=args.num_particles_target,
        num_particles_train=args.num_particles_train,
        data_particle_ess_fraction=args.data_particle_ess_fraction,
        target_particle_ess_fraction=args.target_particle_ess_fraction,
        train_particle_ess_fraction=args.train_particle_ess_fraction,
        # Retain defaults for checkpoints predating the reward-backup fields.
        reward_backup_mode=getattr(
            args, "reward_backup_mode", LP_REWARD_BACKUP_ONLINE_K1
        ),
        reward_backup_samples=getattr(args, "reward_backup_samples", 1),
        qh_backup_uses_reference=getattr(
            args, "qh_backup_uses_reference", False
        ),
        reward_tilt_mode=getattr(
            args, "reward_tilt_mode", LP_REWARD_TILT_FIXED_ALPHA
        ),
        reward_kl_delta=getattr(args, "reward_kl_delta", 0.25),
        reward_row_kl_cap=getattr(args, "reward_row_kl_cap", 1.0),
    )


@jax.jit
def _sample_action_guarded(agent: LPPSAgent, observation: jax.Array):
    if agent.particle_mode == LP_PARTICLE_MODE_S1_MIN:
        action, agent, behavior_info = agent.action_with_diagnostics(
            observation
        )
    else:
        action, agent, behavior_info = (
            agent.data_particle_action_with_diagnostics(observation)
        )
    packed_guard = pack_behavior_guard(action, behavior_info)
    return action, agent, behavior_info, packed_guard




@jax.jit
def _update_critics_guarded(agent: LPPSAgent, batch):
    agent, update_info = agent.update_critic_only(batch)
    return agent, update_info, pack_update_guard(update_info, actor_active=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _append_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = next(csv.reader(handle), None)
        if not fields:
            raise ValueError(f"Existing CSV has no header: {path}.")
    else:
        fields = list(rows[0])
    unknown = sorted(set().union(*(set(row) for row in rows)) - set(fields))
    if unknown:
        raise ValueError(
            f"Refusing CSV schema drift in {path}; new columns: {unknown}."
        )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval=float("nan"))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)








def _config(args: argparse.Namespace) -> dict[str, Any]:
    # ``resume`` identifies the transport checkpoint used for this process; it
    # is not part of the immutable experiment definition.  Keeping it out of
    # the locked config lets a checkpoint produced after one resume be resumed
    # again without silently changing any algorithm or stopping parameter.
    proposal_mode = lp_proposal_mode(
        args.likelihood_fraction,
        getattr(args, "proposal_family", LP_LEGACY_AUTO_PROPOSAL_MODE),
    )
    bridge_only_keys = {
        "proposal_family",
        "bridge_likelihood_fraction",
        "bridge_reference_samples",
        "bridge_latent_std",
        "bridge_component_weighting",
    }
    is_bridge = proposal_mode == LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE
    qh_critic_mode = getattr(args, "qh_critic_mode", "twin_max")
    actor_safety_mode = getattr(
        args, "actor_safety_mode", LP_ACTOR_SAFETY_QH_ROUTED
    )
    config = {
        key: _jsonable(value)
        for key, value in vars(args).items()
        if key not in {"resume", "dry_run"}
        and (is_bridge or key not in bridge_only_keys)
    }
    config.update(
        {
            "actor_coordinate": LP_ACTOR_COORDINATE,
            "target_estimator": (
                LP_TARGET_ESTIMATOR
                if args.eta < 1.0
                else "kl_constrained_posterior_snis_full_projection_v1"
            ),
            "proposal_mode": proposal_mode,
            "proposal_note": (
                "defensive bridge-MIS changes only the finite-M proposal; "
                "the clean HJ-Gibbs target and noise expectation are unchanged"
                if is_bridge
                else "rho=1 forward-likelihood proposal is the preregistered main; "
                "rho=0.5 balance-MIS remains a separately labelled diagnostic"
            ),
            "include_action_jacobian": True,
            "score_network_heads": 1,
            "score_semantic_branches": (
                ["reward"]
                if actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY
                else ["feasible", "recovery"]
            ),
            "legacy_dzfill_actor_update": False,
            "action_q_gradients": False,
            "qh_critic_mode": qh_critic_mode,
            "actor_safety_mode": actor_safety_mode,
            "qh_actor_role": (
                "diagnostic_critic_only"
                if actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY
                else "branch_routing_mask_and_recovery_energy"
            ),
            "stage_mode": "stage_b",
            "h_semantics": "raw_signed_sdf",
            "qh_candidate_source": (
                "independent_full_policy_n8"
                if args.particle_mode == LP_PARTICLE_HJ_FULL_CONSISTENT
                else "policy_plus_gaussian"
            ),
            "legacy_core_qh_candidate_source": "policy_plus_gaussian",
            "particle_mode": args.particle_mode,
            "particle_training_scope": args.particle_mode,
            "particle_hj_semantics": (
                "independent_policy_plus_gaussian_k8_sigma0.3"
                if args.particle_mode
                != LP_PARTICLE_HJ_FULL_CONSISTENT
                else "same_independent_full_policy_n8_candidate_minimum"
            ),
            "particle_candidates_reused_for_hj": (
                args.particle_mode == LP_PARTICLE_HJ_FULL_CONSISTENT
            ),
            "particle_roles": {
                "data": args.num_particles_data,
                "reward_td_target": args.num_particles_target,
                "actor_outer": args.num_particles_train,
                "hj_gate_and_backup": (
                    args.num_particles_target
                    if args.particle_mode
                    == LP_PARTICLE_HJ_FULL_CONSISTENT
                    else "independent_policy_plus_gaussian_k8"
                ),
            },
            "particle_selector": (
                "none"
                if args.particle_mode == LP_PARTICLE_MODE_S1_MIN
                else "per_state_mad_safe_softmax_ess_target"
            ),
            "deployment_selector": "none",
            "environment_execution": (
                "raw_k1"
                if args.particle_mode == LP_PARTICLE_MODE_S1_MIN
                else "independent_full_policy_n8_safe_softmax"
            ),
            "reward_tilt_semantics": (
                "fixed_alpha_qr_gibbs"
                if args.reward_tilt_mode == LP_REWARD_TILT_FIXED_ALPHA
                else (
                    "per_row_kl_budget_from_reward_free_feasible_mis_base"
                    if args.reward_tilt_mode == LP_REWARD_TILT_KL_BUDGET
                    else (
                        "shared_batch_kl_budget_on_raw_qr_advantages"
                        if args.reward_tilt_mode == LP_REWARD_TILT_BATCH_KL
                        else (
                            "shared_batch_kl_budget_on_raw_qr_advantages"
                            "_with_per_row_kl_cap"
                        )
                    )
                )
            ),
            "reward_tilt_recovery_semantics": "unchanged_beta_safety_qh",
            "reward_tilt_kl_solver": (
                None
                if args.reward_tilt_mode == LP_REWARD_TILT_FIXED_ALPHA
                else (
                    "standardized_qr_per_row_adaptive_bracket_"
                    "32_step_log_safe_bisection"
                    if args.reward_tilt_mode == LP_REWARD_TILT_KL_BUDGET
                    else (
                        "raw_qr_shared_dual_global_rms_parameterization_"
                        "adaptive_bracket_32_step_log_safe_bisection"
                    )
                )
            ),
            "accel_schema": "swimmer_s1_role_particles_isolated_v1",
            "joint_update": True,
            "paired_replay": True,
            "packed_guards": True,
            "replay_draw_order": "critic_then_actor_independent",
            "update_order": "critic_then_actor",
        }
    )
    if is_bridge:
        config.update(
            {
                "posterior_moment": "noise_expectation",
                "rho_anchor": args.eta,
                "proposal_omega": args.bridge_likelihood_fraction,
                "bridge_density_semantics": (
                    "equal_normalized_product_components"
                    if args.bridge_component_weighting == "uniform"
                    else "reference_kde_exact_conditional"
                ),
                "reference_pool_source": "frozen_actor_independent_full_ddpm",
            }
        )
    return config




def _init_wandb(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    resume_id: str | None,
):
    if not args.wandb:
        return None, {
            "requested": False,
            "initialized": False,
            "run_id": None,
            "run_url": None,
        }
    import wandb

    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "name": args.run_name,
        "config": dict(config),
        "dir": str(wandb_dir),
    }
    if resume_id is not None:
        kwargs.update(id=resume_id, resume="must")
    run = wandb.init(**kwargs)
    return run, {
        "requested": True,
        "initialized": True,
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
        "run_path": str(run.path),
    }


def _float_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in payload.items():
        array = np.asarray(jax.device_get(value))
        if array.shape != ():
            continue
        output[key] = float(array)
    return output


def _host_packed_guard(
    packed_guard: Any, keys: tuple[str, ...]
) -> tuple[int, dict[str, float]]:
    """Decode an already-fetched packed guard without another device sync."""

    status = int(np.asarray(packed_guard.status_code))
    values = np.asarray(packed_guard.values)
    if values.shape != (len(keys),):
        raise ValueError(
            f"Packed guard shape {values.shape!r} does not match {len(keys)} keys."
        )
    return status, {
        key: float(values[index]) for index, key in enumerate(keys)
    }


def _empty_behavior_info(
    particle_mode: str = LP_PARTICLE_MODE_S1_MIN,
) -> dict[str, jax.Array]:
    """Stabilize the actor CSV schema at the warmup/actor boundary.

    At exactly ``start_training`` the original protocol still executes a
    Box-sampled action but performs its first actor update.  Those diagnostics
    are undefined for that single action; explicit NaNs preserve that meaning
    while ensuring every actor-training CSV row has the same columns.  Particle
    arms predeclare their behavior-only columns here because those columns
    appear on the first policy-sampled action one step later.
    """

    if particle_mode not in LP_PARTICLE_MODES:
        raise ValueError(f"Unknown particle_mode={particle_mode!r}.")
    keys = list(BEHAVIOR_INFO_KEYS)
    if particle_mode != LP_PARTICLE_MODE_S1_MIN:
        keys.extend(PARTICLE_BEHAVIOR_INFO_KEYS)
    return {
        key: jnp.asarray(float("nan"), dtype=jnp.float32)
        for key in keys
    }


def _run_state(
    *,
    step: int,
    update_steps: int,
    episodes: int,
    observation: np.ndarray,
    invalid_history: deque[float],
    elapsed_seconds: float,
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "global_env_step": int(step),
        "gradient_update_step": int(update_steps),
        "episodes": int(episodes),
        "current_observation": np.asarray(observation, dtype=np.float32),
        "invalid_history": list(invalid_history),
        "elapsed_seconds": float(elapsed_seconds),
        "stop_reason": stop_reason,
    }




def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.dry_run:
        print(json.dumps(_config(args), indent=2, sort_keys=True))
        return

    run_dir = (args.output_root / args.run_name).resolve()
    if args.resume is None:
        if run_dir.exists():
            raise FileExistsError(f"Refusing to reuse LP-PS output directory {run_dir}.")
        run_dir.mkdir(parents=True)
    else:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume output directory does not exist: {run_dir}.")

    config = _config(args)
    config_path = run_dir / "config.json"
    if args.resume is None:
        _atomic_json(config_path, config)
        locked_config = config
    else:
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        locked_config = existing_config
        # Fail closed on the entire experiment definition, including logging,
        # invalid-posterior and early-stop controls.  Shape compatibility is
        # not enough for an exact continuation.
        if existing_config != config:
            differing = sorted(
                key
                for key in set(existing_config) | set(config)
                if existing_config.get(key) != config.get(key)
            )
            raise ValueError(
                "Resume arguments do not match the locked run config; "
                f"differing keys: {differing}."
            )

    env = make_velocity_env(env_id=args.env_id, seed=args.seed)
    env.action_space.seed(args.seed)
    agent = make_agent(args, env)
    replay = SafeReplayBuffer(
        env.observation_space,
        env.action_space,
        max(args.replay_size, args.max_steps),
    )
    replay.seed(args.seed)

    resume_payload = None
    if args.resume is None:
        observation = env.reset()
        start_step = 0
        update_steps = 0
        episodes = 0
        invalid_history: deque[float] = deque(maxlen=args.invalid_window)
        elapsed_before = 0.0
        resume_wandb_id = None
    else:
        # The caller-supplied checkpoint must itself claim the exact locked
        # run configuration.  Shape-compatible optimizer states from another
        # rho/eta/LR/Bellman run are not a valid resume.
        checkpoint_preview = load_checkpoint_payload(args.resume.resolve())
        checkpoint_config = checkpoint_preview["config"]
        if checkpoint_config != locked_config:
            differing = sorted(
                key
                for key in set(checkpoint_config) | set(locked_config)
                if checkpoint_config.get(key) != locked_config.get(key)
            )
            raise ValueError(
                "Resume checkpoint config does not match the locked run "
                f"config; differing keys: {differing}."
            )
        # Gymnasium creates the wrapper RNG lazily on first reset.  The saved
        # MuJoCo state, wrapper RNGs, observation statistics, and current
        # observation are all restored immediately below, so this reset is
        # initialization only and does not advance the resumed trajectory.
        env.reset()
        agent, replay, runner_state, resume_payload = load_lp_checkpoint(
            args.resume.resolve(),
            agent_template=agent,
            replay_template=replay,
            env=env,
        )
        observation = np.asarray(
            runner_state["current_observation"], dtype=np.float32
        )
        start_step = int(runner_state["global_env_step"])
        update_steps = int(runner_state["gradient_update_step"])
        episodes = int(runner_state["episodes"])
        invalid_history = deque(
            (float(value) for value in runner_state.get("invalid_history", [])),
            maxlen=args.invalid_window,
        )
        if runner_state.get("stop_reason") is not None:
            raise ValueError(
                "Refusing to resume an early-stopped/integrity checkpoint: "
                f"{runner_state['stop_reason']}."
            )
        elapsed_before = float(runner_state.get("elapsed_seconds", 0.0))
        resume_wandb_id = resume_payload.get("wandb", {}).get("run_id")
        if start_step >= args.max_steps:
            raise ValueError(
                f"Checkpoint step {start_step} is already >= max_steps {args.max_steps}."
            )
        if args.obs_stats_calibration_steps > 0:
            calibration_step = args.obs_stats_calibration_steps
            update_stats = bool(getattr(env, "_update_stats", True))
            replay_size = len(replay)
            if start_step <= calibration_step:
                if not update_stats or replay_size != 0:
                    raise ValueError(
                        "A calibration-phase checkpoint must still update "
                        "observation statistics and have an empty replay; "
                        f"got step={start_step}, update_stats={update_stats}, "
                        f"replay_size={replay_size}."
                    )
            else:
                expected_replay_size = start_step - calibration_step
                if update_stats or replay_size != expected_replay_size:
                    raise ValueError(
                        "A post-calibration checkpoint must have frozen "
                        "observation statistics and exactly one stationary-"
                        "coordinate transition per post-calibration step; "
                        f"got step={start_step}, update_stats={update_stats}, "
                        f"replay_size={replay_size}, "
                        f"expected_replay_size={expected_replay_size}."
                    )

    wandb_run, wandb_info = _init_wandb(
        args, locked_config, run_dir, resume_id=resume_wandb_id
    )
    manifest = {
        "status": "running",
        "created_unix": time.time(),
        "run_dir": str(run_dir),
        "config": locked_config,
        "wandb": wandb_info,
        "resumed_from": str(args.resume.resolve()) if args.resume else None,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    _atomic_json(run_dir / "manifest.json", manifest)

    train_metrics_path = run_dir / "training_curves.csv"
    train_jsonl_path = run_dir / "training_metrics.jsonl"
    critic_warmup_path = run_dir / "critic_warmup_curves.csv"
    critic_warmup_jsonl_path = run_dir / "critic_warmup_metrics.jsonl"
    train_episodes_path = run_dir / "train_episodes.csv"
    snis_path = run_dir / "snis_diagnostics.csv"
    refresh_events_path = run_dir / "reference_refresh_events.csv"
    normalization_events_path = run_dir / "normalization_events.jsonl"
    guard_events_path = run_dir / "guard_events.jsonl"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    started = time.time()
    stop_reason: str | None = None
    last_update_metrics: dict[str, float] = {}
    last_checkpoint: Path | None = None
    step = start_step
    try:
        for step in range(start_step + 1, args.max_steps + 1):
            if (
                args.obs_stats_calibration_steps > 0
                and step == args.obs_stats_calibration_steps + 1
            ):
                # Phase 1 (steps 1..calibration) estimates one observation
                # transform under random behavior but contributes no replay
                # data.  ``observation`` was normalized immediately after the
                # final Welford update at the previous step, so it already
                # uses the statistics being frozen; no extra environment reset
                # (and hence no extra RNG advance) is needed.
                env.freeze_obs_stats()
                replay = SafeReplayBuffer(
                    env.observation_space,
                    env.action_space,
                    max(args.replay_size, args.max_steps),
                )
                replay.seed(args.seed)
                obs_mean, obs_var, obs_count = env.get_obs_stats()
                normalization_event = {
                    "event": "freeze_observation_statistics",
                    "frozen_after_step": args.obs_stats_calibration_steps,
                    "effective_from_step": step,
                    "observation_count": int(obs_count),
                    "observation_mean_l2": float(np.linalg.norm(obs_mean)),
                    "observation_std_l2": float(
                        np.linalg.norm(np.sqrt(obs_var + 1e-8))
                    ),
                    "replay_size_after_reset": len(replay),
                    "critic_start_training": args.critic_start_training,
                    "actor_start_training": args.start_training,
                }
                _append_jsonl(normalization_events_path, normalization_event)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "integrity/obs_stats_frozen": 1.0,
                            "integrity/obs_stats_count": float(obs_count),
                            "integrity/replay_size_after_obs_freeze": float(
                                len(replay)
                            ),
                        },
                        step=args.obs_stats_calibration_steps,
                    )
            behavior_info: Mapping[str, Any] = _empty_behavior_info(
                args.particle_mode
            )
            if step <= args.start_training:
                action = _strict_warmup_action(env.action_space.sample())
            else:
                action_jax, agent, behavior_info, packed_behavior = (
                    _sample_action_guarded(
                        agent, jnp.asarray(observation)
                    )
                )
                # One host transfer carries both the environment action and
                # all immediate behavior guards.  The updated agent and full
                # metric mapping stay device-resident.
                action_host, packed_behavior_host = jax.device_get(
                    (action_jax, packed_behavior)
                )
                action = np.asarray(action_host, dtype=np.float32)
                behavior_status, behavior_guard = _host_packed_guard(
                    packed_behavior_host, BEHAVIOR_GUARD_KEYS
                )
                if behavior_status == int(GuardStatus.BEHAVIOR_LATENT_INVALID):
                    raise FloatingPointError(
                        "LP-PS training sampled a non-finite/exploding latent."
                    )
                if behavior_status == int(GuardStatus.BEHAVIOR_ACTION_INVALID):
                    raise FloatingPointError(
                        "LP-PS training produced a non-finite/boundary "
                        "environment action."
                    )
                if behavior_status != int(GuardStatus.OK):
                    raise FloatingPointError(
                        f"Unknown packed behavior guard status {behavior_status}: "
                        f"{behavior_guard}."
                    )
            if not np.all(np.isfinite(action)) or not np.all(np.abs(action) < 1.0):
                raise FloatingPointError(
                    "LP-PS training produced a non-finite/boundary environment action."
                )

            next_observation, reward, h_value, _, done, info = env.step(action)
            if step > args.obs_stats_calibration_steps:
                replay.insert(
                    {
                        "observations": observation,
                        "actions": action,
                        "rewards": reward,
                        "costs": h_value,
                        "masks": 0.0 if info.get("terminated", False) else 1.0,
                        "dones": done,
                        "next_observations": next_observation,
                    }
                )
            observation = next_observation
            if done:
                episodes += 1
                episode = env.episode_info
                episode_row = {
                    "train_step": step,
                    "episode": episodes,
                    "reward": float(episode["reward"]),
                    "cost_binary": float(episode["cost_binary"]),
                    "cost_sdf": float(episode["sdf_violation"]),
                    "avg_velocity": float(episode["avg_velocity"]),
                    "max_velocity": float(episode["max_velocity"]),
                    "length": float(episode["length"]),
                }
                _append_csv(train_episodes_path, [episode_row])
                if wandb_run is not None:
                    wandb_run.log(
                        {f"train_episode/{key}": value for key, value in episode_row.items()},
                        step=step,
                    )
                observation = env.reset()

            if step >= args.critic_start_training:
                actor_active = step >= args.start_training
                for _ in range(args.updates_per_step):
                    if actor_active:
                        paired_batches = sample_paired_replay_batches(
                            replay,
                            args.critic_batch_size,
                            args.actor_batch_size,
                        )
                        agent, core_update_info, packed_update = (
                            joint_paired_update_with_guard(
                                agent,
                                paired_batches.combined_batch,
                                critic_batch_size=args.critic_batch_size,
                            )
                        )
                        guard_keys = UPDATE_GUARD_KEYS
                    else:
                        critic_batch = replay.sample(args.critic_batch_size)
                        agent, core_update_info, packed_update = (
                            _update_critics_guarded(agent, critic_batch)
                        )
                        guard_keys = CRITIC_GUARD_KEYS
                    update_steps += 1
                    update_info = {
                        **core_update_info,
                        **behavior_info,
                    }

                    # Check integrity immediately using one packed vector
                    # instead of synchronizing each scalar independently.
                    packed_update_host = jax.device_get(packed_update)
                    update_status, guard = _host_packed_guard(
                        packed_update_host, guard_keys
                    )
                    missing_or_nonfinite = [
                        key for key, value in guard.items()
                        if not math.isfinite(value)
                    ]
                    if update_status != int(GuardStatus.OK):
                        try:
                            stop_reason = GuardStatus(update_status).name
                        except ValueError:
                            stop_reason = "NONFINITE_TRAIN_METRIC"

                    # invalid_feasible_rate uses all actor rows as its
                    # denominator; also log the conditional-on-feasible rate.
                    invalid = guard.get("invalid_feasible_rate", float("nan"))
                    if actor_active and math.isfinite(invalid):
                        invalid_history.append(invalid)
                    if (
                        stop_reason is None
                        and actor_active
                        and len(invalid_history) >= args.invalid_window
                    ):
                        rolling_invalid = float(np.mean(invalid_history))
                        if rolling_invalid > args.invalid_stop_threshold:
                            stop_reason = "INVALID_FEASIBLE_RATE_GT_10_PERCENT"
                        elif rolling_invalid > args.invalid_warning_threshold:
                            print(
                                f"WARNING step={step}: rolling invalid-feasible "
                                f"rate={rolling_invalid:.4f}",
                                flush=True,
                            )

                    if actor_active and guard.get("reference_refreshed", 0.0) == 1.0:
                        refresh_row = {
                            "train_step": step,
                            "gradient_update_step": update_steps,
                            "actor_update_count": int(guard["actor_update_count"]),
                            "reference_block_count": int(
                                guard["reference_block_count"]
                            ),
                        }
                        _append_csv(refresh_events_path, [refresh_row])
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "train/reference_refresh_event": 1.0,
                                    "train/reference_block_count": guard[
                                        "reference_block_count"
                                    ],
                                },
                                step=step,
                            )

                    if stop_reason is not None:
                        guard_event = {
                            "train_step": step,
                            "gradient_update_step": update_steps,
                            "stop_reason": stop_reason,
                            "missing_or_nonfinite": missing_or_nonfinite,
                            **guard,
                        }
                        _append_jsonl(guard_events_path, guard_event)
                        print(
                            f"ERROR immediate guard at step={step}: {stop_reason}",
                            flush=True,
                        )
                        break

                if step % args.log_interval == 0:
                    last_update_metrics = _float_metrics(update_info)
                    row = {
                        "train_step": step,
                        "gradient_update_step": update_steps,
                        "elapsed_seconds": elapsed_before + time.time() - started,
                        **last_update_metrics,
                    }
                    if actor_active:
                        invalid = last_update_metrics.get(
                            "invalid_feasible_rate", float("nan")
                        )
                        _append_jsonl(train_jsonl_path, row)
                        _append_csv(train_metrics_path, [row])
                        snis_keys = (
                            "posterior_ess_mean",
                            "posterior_ess_over_m_mean",
                            "posterior_max_weight_mean",
                            "posterior_reward_tilt_kl_budget_active",
                            "posterior_reward_tilt_batch_kl_active",
                            "posterior_reward_tilt_capped_batch_kl_active",
                            "posterior_reward_tilt_kl_delta",
                            "posterior_reward_tilt_row_kl_cap",
                            "posterior_reward_tilt_achieved_kl_mean",
                            "posterior_reward_tilt_active_rate",
                            "posterior_reward_tilt_active_kl_mean",
                            "posterior_reward_tilt_active_kl_max",
                            "posterior_reward_tilt_top1_kl_fraction",
                            "posterior_reward_tilt_effective_state_fraction",
                            "posterior_reward_tilt_row_cap_active_rate",
                            "posterior_reward_tilt_target_hit_rate",
                            "posterior_reward_tilt_attainable_rate",
                            "posterior_reward_tilt_ess_over_m_mean",
                            "posterior_reward_tilt_max_weight_mean",
                            "posterior_reward_tilt_q_gain_mean",
                            "posterior_reward_tilt_flat_rate",
                            "posterior_reward_tilt_effective_kappa_mean",
                            "posterior_reward_tilt_tau_proxy_mean",
                            "invalid_feasible_rate",
                            "invalid_feasible_rate_conditional",
                            "invalid_recovery_rate",
                            "invalid_recovery_rate_conditional",
                            "posterior_critic_nonfinite_fraction",
                            "posterior_all_zero_rate",
                            "posterior_feasible_acceptance_mean",
                            "feasible_gate_fraction",
                            "recovery_gate_fraction",
                            "stage_b_candidate_preclip_fraction",
                            "outer_latent_abs_max",
                            "proposal_latent_abs_max",
                            "actor_reference_output_mse",
                            "posterior_feasible_row_count",
                            "posterior_feasible_qr_span_p90_p10",
                            "posterior_feasible_qh_span_p90_p10",
                            "posterior_feasible_alpha_qr_span_p90_p10",
                            "posterior_feasible_beta_qh_span_p90_p10",
                            "posterior_recovery_row_count",
                            "posterior_recovery_qr_span_p90_p10",
                            "posterior_recovery_qh_span_p90_p10",
                            "posterior_recovery_alpha_qr_span_p90_p10",
                            "posterior_recovery_beta_qh_span_p90_p10",
                            "particle_train_active",
                            "particle_train_n",
                            "particle_train_ess_mean",
                            "particle_train_feasible_fraction_mean",
                            "particle_train_no_feasible_rate",
                            "particle_train_invalid_candidate_fraction_mean",
                            "particle_target_active",
                            "particle_target_n",
                            "particle_target_ess_mean",
                            "particle_target_feasible_fraction_mean",
                            "particle_target_no_feasible_rate",
                            "particle_target_invalid_candidate_fraction_mean",
                            "particle_target_rng_routing_valid",
                            "particle_target_key_collision_rate",
                            "particle_train_rng_routing_valid",
                            "particle_train_key_collision_rate",
                            "particle_rng_routing_valid",
                            "particle_key_collision_rate",
                            "particle_hj_role_separated",
                            "particle_hj_full_consistent",
                        )
                        if (
                            lp_proposal_mode(
                                args.likelihood_fraction,
                                getattr(
                                    args,
                                    "proposal_family",
                                    LP_LEGACY_AUTO_PROPOSAL_MODE,
                                ),
                            )
                            == LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE
                        ):
                            snis_keys += (
                                "posterior_likelihood_weight_mass_mean",
                                "posterior_likelihood_fraction_actual",
                                "posterior_likelihood_component_ess_mean",
                                "posterior_reference_bridge_component_ess_mean",
                                "posterior_reference_bridge_weight_mass_mean",
                                "posterior_reference_bridge_fraction_actual",
                                "posterior_likelihood_sample_count",
                                "posterior_reference_bridge_sample_count",
                                "posterior_likelihood_feasible_acceptance_mean",
                                "posterior_reference_bridge_feasible_acceptance_mean",
                                "posterior_ess_feasible_mean",
                                "posterior_ess_recovery_mean",
                                "posterior_reference_bridge_weight_mass_feasible_mean",
                                "posterior_reference_bridge_weight_mass_recovery_mean",
                                "posterior_reference_bridge_component_ess_feasible_mean",
                                "posterior_reference_bridge_component_ess_recovery_mean",
                                "reference_center_latent_norm_mean",
                            )
                        _append_csv(
                            snis_path,
                            [
                                {
                                    "train_step": step,
                                    "gradient_update_step": update_steps,
                                    **{
                                        key: last_update_metrics.get(
                                            key, float("nan")
                                        )
                                        for key in snis_keys
                                    },
                                }
                            ],
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    f"train/{key}": value
                                    for key, value in row.items()
                                },
                                step=step,
                            )
                        print(
                            f"[train {step}] actor_loss="
                            f"{last_update_metrics.get('actor_loss', float('nan')):.5g} "
                            f"ESS={last_update_metrics.get('posterior_ess_mean', float('nan')):.3f} "
                            f"invalid={invalid:.4f}",
                            flush=True,
                        )
                    else:
                        warmup_keys = (
                            "critic_loss_1",
                            "critic_loss_2",
                            "safe_critic_loss",
                            "q1_mean",
                            "q2_mean",
                            "qh_mean",
                            "target_q_mean",
                            "target_qh_mean",
                        )
                        warmup_row = {
                            "train_step": step,
                            "gradient_update_step": update_steps,
                            "elapsed_seconds": row["elapsed_seconds"],
                            **{
                                key: last_update_metrics.get(key, float("nan"))
                                for key in warmup_keys
                            },
                        }
                        _append_jsonl(critic_warmup_jsonl_path, warmup_row)
                        _append_csv(critic_warmup_path, [warmup_row])
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    f"critic_warmup/{key}": value
                                    for key, value in warmup_row.items()
                                },
                                step=step,
                            )
                        print(
                            f"[critic-warmup {step}] "
                            f"Qloss={warmup_row['critic_loss_1']:.5g} "
                            f"Qhloss={warmup_row['safe_critic_loss']:.5g}",
                            flush=True,
                        )


            should_checkpoint = step % args.checkpoint_interval == 0
            if should_checkpoint or stop_reason is not None:
                runner_state = _run_state(
                    step=step,
                    update_steps=update_steps,
                    episodes=episodes,
                    observation=observation,
                    invalid_history=invalid_history,
                    elapsed_seconds=elapsed_before + time.time() - started,
                    stop_reason=stop_reason,
                )
                last_checkpoint = checkpoint_dir / f"step_{step}.pkl"
                save_lp_checkpoint(
                    last_checkpoint,
                    agent=agent,
                    replay=replay,
                    env=env,
                    runner_state=runner_state,
                    config=locked_config,
                    wandb_info=wandb_info,
                )
                print(f"[checkpoint {step}] {last_checkpoint}", flush=True)

            if stop_reason is not None:
                break

        status = (
            "complete"
            if step == args.max_steps and stop_reason is None
            else "early_stopped"
        )

        if last_checkpoint is None or last_checkpoint.name != f"step_{step}.pkl":
            last_checkpoint = checkpoint_dir / f"step_{step}.pkl"
            save_lp_checkpoint(
                last_checkpoint,
                agent=agent,
                replay=replay,
                env=env,
                runner_state=_run_state(
                    step=step,
                    update_steps=update_steps,
                    episodes=episodes,
                    observation=observation,
                    invalid_history=invalid_history,
                    elapsed_seconds=elapsed_before + time.time() - started,
                    stop_reason=stop_reason,
                ),
                config=locked_config,
                wandb_info=wandb_info,
            )

        duration = elapsed_before + time.time() - started
        manifest.update(
            status=status,
            completed_unix=time.time(),
            duration_seconds=duration,
            stop_reason=stop_reason,
            final_step=step,
            final_checkpoint=str(last_checkpoint),
        )
        _atomic_json(run_dir / "manifest.json", manifest)
        if status == "complete":
            (run_dir / "COMPLETE").touch()
        else:
            (run_dir / "EARLY_STOPPED").touch()
    except BaseException as error:
        manifest.update(
            status="failed",
            failed_unix=time.time(),
            error_type=type(error).__name__,
            error=str(error),
            last_step=step,
        )
        _atomic_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        env.env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

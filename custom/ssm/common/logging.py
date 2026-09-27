"""Episode and update metrics for training logs."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import numpy as np


def _as_float(value) -> Optional[float]:
    try:
        scalar = float(np.asarray(value))
    except Exception:
        return None
    if math.isnan(scalar):
        return None
    return scalar


def _first(metrics: Mapping[str, object], *keys: str) -> Optional[float]:
    for key in keys:
        if key in metrics:
            value = _as_float(metrics[key])
            if value is not None:
                return value
    return None


def add_if_present(out: Dict[str, float], key: str, value) -> None:
    scalar = _as_float(value)
    if scalar is not None:
        out[key] = scalar


def compact_episode_log(
    episode_info: Mapping[str, object],
    *,
    episode_count: Optional[int] = None,
    terminated: Optional[bool] = None,
    success: Optional[object] = None,
    collision: Optional[bool] = None,
) -> Dict[str, float]:
    """Curated episode-level W&B metrics."""
    out: Dict[str, float] = {}
    add_if_present(out, "train/return", episode_info.get("reward"))
    add_if_present(out, "train/cost", episode_info.get("cost", episode_info.get("task_cost")))
    add_if_present(out, "train/length", episode_info.get("length"))
    if success is None:
        success = episode_info.get("success")
    if success is not None:
        add_if_present(out, "train/success", success)
    if terminated is None:
        terminated = episode_info.get("terminated")
    if terminated is not None:
        add_if_present(out, "train/terminated", float(bool(terminated)))
    if collision is not None:
        add_if_present(out, "train/collision", float(bool(collision)))
    if episode_count is not None:
        add_if_present(out, "train/episodes", episode_count)
    return out


def compact_update_log(
    update_info: Mapping[str, object],
    *,
    sps: Optional[float] = None,
) -> Dict[str, float]:
    """Summarize update losses, critic values and throughput for W&B."""
    out: Dict[str, float] = {}
    add_if_present(out, "loss/actor", update_info.get("actor_loss"))
    critic_1 = _first(update_info, "critic_loss_1")
    critic_2 = _first(update_info, "critic_loss_2")
    if critic_1 is not None and critic_2 is not None:
        add_if_present(out, "loss/critic", 0.5 * (critic_1 + critic_2))
    else:
        add_if_present(out, "loss/critic", critic_1 if critic_1 is not None else critic_2)
    add_if_present(out, "loss/safe_critic", update_info.get("safe_critic_loss"))
    add_if_present(out, "loss/safe_value", update_info.get("safe_value_loss"))
    add_if_present(out, "train/safe_frac", update_info.get("safe_frac"))
    add_if_present(out, "train/action_safe_frac", update_info.get("indicator_frac"))
    add_if_present(out, "train/reward_coverage", update_info.get("reward_coverage"))
    add_if_present(out, "critic/q_mean", _first(update_info, "q1_mean", "q2_mean"))
    add_if_present(out, "critic/qh_mean", _first(update_info, "qh_gate_mean", "qh_policy_mean", "qc_mean"))
    if sps is not None:
        add_if_present(out, "time/sps", sps)
    return out



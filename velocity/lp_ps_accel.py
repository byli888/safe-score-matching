"""Compiled update and replay helpers for the LP-PS-SSM online runner.

This module intentionally contains orchestration-only optimizations.  It does
not change the LP posterior, critic targets, actor loss, candidate source, or
update-to-data ratio.  In particular:

* critic and actor replay indices are still drawn independently and in the
  legacy NumPy-RNG order (critic first, actor second);
* the joint update executes ``update_critic_only`` before ``update_actor``;
* every immediate integrity guard used by the original runner remains active.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# Keep these orders stable: the packed vector is an on-device ABI between the
# compiled update and the host-side integrity/rolling-window logic.
CRITIC_GUARD_KEYS = (
    "critic_loss_1",
    "critic_loss_2",
    "safe_critic_loss",
    "legacy_dzfill_actor_used",
)

ACTOR_GUARD_KEYS = (
    "actor_loss",
    "actor_grad_norm_raw",
    "actor_output_abs_max",
    "posterior_ess_mean",
    "posterior_max_weight_mean",
    "invalid_feasible_rate",
    "invalid_feasible_rate_conditional",
    "invalid_recovery_rate",
    "posterior_critic_nonfinite_fraction",
    "posterior_valid_rate",
    "feasible_gate_fraction",
    "outer_latent_abs_max",
    "noisy_latent_abs_max",
    "proposal_latent_abs_max",
    "stage_b_candidate_boundary_fraction",
    "reference_refreshed",
    "actor_update_count",
    "reference_block_count",
)

UPDATE_GUARD_KEYS = CRITIC_GUARD_KEYS + ACTOR_GUARD_KEYS
UPDATE_GUARD_INDEX = {name: index for index, name in enumerate(UPDATE_GUARD_KEYS)}

BEHAVIOR_INFO_KEYS = (
    "behavior_latent_abs_max",
    "behavior_latent_norm",
    "behavior_latent_nonfinite_fraction",
    "behavior_action_saturation_fraction",
)

# The last two entries are computed directly from the action on device.  They
# let the host receive the action and all behavior guards in one transfer.
BEHAVIOR_GUARD_KEYS = BEHAVIOR_INFO_KEYS + (
    "behavior_action_nonfinite_fraction",
    "behavior_action_boundary_fraction",
)
BEHAVIOR_GUARD_INDEX = {
    name: index for index, name in enumerate(BEHAVIOR_GUARD_KEYS)
}


class GuardStatus(IntEnum):
    """Stable status codes in the same priority order as the legacy runner."""

    OK = 0
    NONFINITE_TRAIN_METRIC = 1
    LEGACY_DZFILL_PATH_USED = 2
    STAGE_B_BOUNDARY_ACTION = 3
    ACTOR_OUTPUT_EXPLODING = 4
    LATENT_OUTPUT_EXPLODING = 5
    POSTERIOR_CRITIC_NONFINITE = 6
    INVALID_RECOVERY_POSTERIOR = 7

    # Behavior guards run before the environment action is executed.  The old
    # runner reports one latent error and one action error, so these codes keep
    # that distinction without strengthening or weakening the checks.
    BEHAVIOR_LATENT_INVALID = 20
    BEHAVIOR_ACTION_INVALID = 21


class PackedGuard(NamedTuple):
    """Small JAX PyTree returned by a compiled action/update call."""

    status_code: jax.Array
    values: jax.Array


@dataclass(frozen=True)
class PairedReplayBatches:
    """Independent critic/actor batches plus their exact sampled indices."""

    combined_batch: Any
    critic_batch: Any
    actor_batch: Any
    critic_indices: np.ndarray
    actor_indices: np.ndarray


def guard_status_name(status_code: int | np.ndarray | jax.Array) -> str:
    """Decode a scalar status code for logs and guard-event artifacts."""

    scalar = int(np.asarray(jax.device_get(status_code)))
    try:
        return GuardStatus(scalar).name
    except ValueError:
        return f"UNKNOWN_GUARD_STATUS_{scalar}"


def packed_guard_dict(
    packed: PackedGuard, keys: tuple[str, ...]
) -> dict[str, float]:
    """Transfer one packed vector and expose the legacy host-side mapping."""

    _, output = unpack_packed_guard(packed, keys)
    return output


def unpack_packed_guard(
    packed: PackedGuard, keys: tuple[str, ...]
) -> tuple[int, dict[str, float]]:
    """Fetch status and values together in one device-to-host operation."""

    host_status, host_values = jax.device_get(packed)
    status = int(np.asarray(host_status))
    values = np.asarray(host_values)
    if values.shape != (len(keys),):
        raise ValueError(
            f"Packed guard shape {values.shape!r} does not match {len(keys)} keys."
        )
    output = {key: float(values[index]) for index, key in enumerate(keys)}
    return status, output


def _status(value: GuardStatus) -> jax.Array:
    return jnp.asarray(int(value), dtype=jnp.int32)


def pack_update_guard(
    update_info: Mapping[str, Any], *, actor_active: bool = True
) -> PackedGuard:
    """Pack update scalars and reproduce the original immediate-stop priority.

    ``actor_active`` is a Python/static choice.  A missing required metric is a
    programming error and raises while tracing rather than silently weakening
    the guard.  The rolling ten-update invalid-feasible rule intentionally
    remains host-side; its required scalar is included in ``values``.
    """

    keys = UPDATE_GUARD_KEYS if actor_active else CRITIC_GUARD_KEYS
    values = jnp.stack(
        [jnp.asarray(update_info[key], dtype=jnp.float32) for key in keys]
    )
    # Keep the packed ABI stable for legacy checkpoints while still failing
    # closed on every bridge-only scalar.  These values participate in the
    # finite check below but are intentionally not returned in PackedGuard;
    # the full metric dictionary is logged separately.
    bridge_guard_values = []
    if actor_active:
        bridge_guard_values = [
            jnp.asarray(value, dtype=jnp.float32)
            for name, value in update_info.items()
            if name.startswith("posterior_reference_bridge_")
            or name.startswith("posterior_reward_tilt_")
            or name
            in (
                "posterior_likelihood_weight_mass_mean",
                "posterior_likelihood_fraction_actual",
                "posterior_likelihood_component_ess_mean",
                "posterior_likelihood_sample_count",
                "posterior_likelihood_feasible_acceptance_mean",
                "posterior_ess_feasible_mean",
                "posterior_ess_recovery_mean",
                "reference_center_latent_norm_mean",
            )
        ]
    finite_guard_values = (
        jnp.concatenate([values, jnp.stack(bridge_guard_values)])
        if bridge_guard_values
        else values
    )

    status = _status(GuardStatus.OK)
    if actor_active:
        index = UPDATE_GUARD_INDEX
        # Build in reverse order so the first legacy if/elif condition has the
        # final (highest) priority when more than one fault is present.
        status = jnp.where(
            values[index["invalid_recovery_rate"]] > 0.0,
            _status(GuardStatus.INVALID_RECOVERY_POSTERIOR),
            status,
        )
        status = jnp.where(
            values[index["posterior_critic_nonfinite_fraction"]] > 0.0,
            _status(GuardStatus.POSTERIOR_CRITIC_NONFINITE),
            status,
        )
        latent_exploding = (
            (values[index["outer_latent_abs_max"]] > 1e4)
            | (values[index["noisy_latent_abs_max"]] > 1e4)
            | (values[index["proposal_latent_abs_max"]] > 1e4)
        )
        status = jnp.where(
            latent_exploding,
            _status(GuardStatus.LATENT_OUTPUT_EXPLODING),
            status,
        )
        status = jnp.where(
            values[index["actor_output_abs_max"]] > 1e4,
            _status(GuardStatus.ACTOR_OUTPUT_EXPLODING),
            status,
        )
        status = jnp.where(
            values[index["stage_b_candidate_boundary_fraction"]] != 0.0,
            _status(GuardStatus.STAGE_B_BOUNDARY_ACTION),
            status,
        )

    # These two conditions apply to critic-only and actor-active updates.
    legacy_index = keys.index("legacy_dzfill_actor_used")
    status = jnp.where(
        values[legacy_index] != 0.0,
        _status(GuardStatus.LEGACY_DZFILL_PATH_USED),
        status,
    )
    status = jnp.where(
        jnp.any(~jnp.isfinite(finite_guard_values)),
        _status(GuardStatus.NONFINITE_TRAIN_METRIC),
        status,
    )
    return PackedGuard(status_code=status, values=values)


def pack_behavior_guard(
    action: jax.Array, behavior_info: Mapping[str, Any]
) -> PackedGuard:
    """Pack action diagnostics without changing the legacy behavior checks."""

    action = jnp.asarray(action)
    action_nonfinite_fraction = jnp.mean(
        (~jnp.isfinite(action)).astype(jnp.float32)
    )
    action_boundary_fraction = jnp.mean(
        (jnp.abs(action) >= 1.0).astype(jnp.float32)
    )
    values = jnp.stack(
        [
            *(
                jnp.asarray(behavior_info[key], dtype=jnp.float32)
                for key in BEHAVIOR_INFO_KEYS
            ),
            action_nonfinite_fraction,
            action_boundary_fraction,
        ]
    )
    index = BEHAVIOR_GUARD_INDEX
    latent_invalid = (
        (values[index["behavior_latent_nonfinite_fraction"]] != 0.0)
        | ~jnp.isfinite(values[index["behavior_latent_abs_max"]])
        | (values[index["behavior_latent_abs_max"]] > 1e4)
    )
    action_invalid = (
        (values[index["behavior_action_nonfinite_fraction"]] != 0.0)
        | (values[index["behavior_action_boundary_fraction"]] != 0.0)
    )
    status = jnp.where(
        action_invalid,
        _status(GuardStatus.BEHAVIOR_ACTION_INVALID),
        _status(GuardStatus.OK),
    )
    # The original runner checks the latent before it checks action validity.
    status = jnp.where(
        latent_invalid,
        _status(GuardStatus.BEHAVIOR_LATENT_INVALID),
        status,
    )
    return PackedGuard(status_code=status, values=values)


@jax.jit
def joint_update_with_guard(agent, critic_batch, actor_batch):
    """Run the two independent-batch updates in one compiled computation.

    Return the complete metrics and packed integrity guard from the same
    update, avoiding a separate update path on logging steps.
    """

    agent, critic_info = agent.update_critic_only(critic_batch)
    agent, actor_info = agent.update_actor(actor_batch)
    update_info = {**critic_info, **actor_info}
    packed = pack_update_guard(update_info, actor_active=True)
    return agent, update_info, packed


@partial(jax.jit, static_argnames=("critic_batch_size",))
def joint_paired_update_with_guard(
    agent, combined_batch, *, critic_batch_size: int
):
    """Joint update variant that splits one combined device payload on device.

    ``critic_batch_size`` is static and the actor batch is the remaining suffix.
    The host sampler still draws the two index sets independently; concatenating
    their gathered rows only changes transport, not replay semantics.
    """

    critic_batch = jax.tree_util.tree_map(
        lambda value: value[:critic_batch_size], combined_batch
    )
    actor_batch = jax.tree_util.tree_map(
        lambda value: value[critic_batch_size:], combined_batch
    )
    agent, critic_info = agent.update_critic_only(critic_batch)
    agent, actor_info = agent.update_actor(actor_batch)
    update_info = {**critic_info, **actor_info}
    packed = pack_update_guard(update_info, actor_active=True)
    return agent, update_info, packed


def _draw_indices(replay: Any, batch_size: int) -> np.ndarray:
    rng = replay.np_random
    if hasattr(rng, "integers"):
        return np.asarray(rng.integers(len(replay), size=batch_size))
    return np.asarray(rng.randint(len(replay), size=batch_size))


def sample_paired_replay_batches(
    replay: Any,
    critic_batch_size: int,
    actor_batch_size: int,
) -> PairedReplayBatches:
    """Sample independent batches with one combined host gather/H2D payload.

    The two RNG calls exactly mirror

    ``replay.sample(critic_batch_size); replay.sample(actor_batch_size)``.

    Only after both draws are complete are the indices concatenated for one
    explicit-index gather.  This preserves batch contents and the next NumPy
    RNG state while reducing Python indexing and allowing a new runner to
    device-put one combined PyTree.  The existing ``Dataset.sample_jax`` must
    not be substituted here: it closes over a stale snapshot of an online
    replay buffer.
    """

    for name, value in (
        ("critic_batch_size", critic_batch_size),
        ("actor_batch_size", actor_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if len(replay) <= 0:
        raise ValueError("Cannot sample paired batches from an empty replay buffer.")

    critic_size = int(critic_batch_size)
    actor_size = int(actor_batch_size)
    # Do not combine these two calls: their order and boundaries are part of
    # exact checkpoint/resume RNG continuation.
    critic_indices = _draw_indices(replay, critic_size)
    actor_indices = _draw_indices(replay, actor_size)
    combined_indices = np.concatenate((critic_indices, actor_indices), axis=0)
    combined_batch = replay.sample(
        critic_size + actor_size, indx=combined_indices
    )
    critic_batch = jax.tree_util.tree_map(
        lambda value: value[:critic_size], combined_batch
    )
    actor_batch = jax.tree_util.tree_map(
        lambda value: value[critic_size:], combined_batch
    )
    return PairedReplayBatches(
        combined_batch=combined_batch,
        critic_batch=critic_batch,
        actor_batch=actor_batch,
        critic_indices=np.array(critic_indices, copy=True),
        actor_indices=np.array(actor_indices, copy=True),
    )


__all__ = [
    "ACTOR_GUARD_KEYS",
    "BEHAVIOR_GUARD_INDEX",
    "BEHAVIOR_GUARD_KEYS",
    "CRITIC_GUARD_KEYS",
    "GuardStatus",
    "PackedGuard",
    "PairedReplayBatches",
    "UPDATE_GUARD_INDEX",
    "UPDATE_GUARD_KEYS",
    "guard_status_name",
    "joint_paired_update_with_guard",
    "joint_update_with_guard",
    "pack_behavior_guard",
    "pack_update_guard",
    "packed_guard_dict",
    "sample_paired_replay_batches",
    "unpack_packed_guard",
]

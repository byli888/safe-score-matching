"""Robust categorical selector for independent full-policy candidates.

For each state, predicted-feasible candidates receive a reward softmax.  If
none is predicted feasible, all candidates with a finite safety estimate
receive a safety softmin.  Logits are normalized by the row-wise median/MAD,
and a per-row temperature is chosen to target a configurable categorical ESS.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class ParticleDistribution(NamedTuple):
    probabilities: Array
    temperature: Array
    categorical_ess: Array
    target_ess: Array
    feasible_branch: Array
    feasible_fraction: Array
    no_feasible: Array
    eligible_count: Array
    mad: Array
    invalid_candidate_fraction: Array


class ParticleSelection(NamedTuple):
    index: Array
    distribution: ParticleDistribution
    selected_probability: Array
    selected_reward_rank: Array
    selected_safety_rank: Array


def _masked_median(values: Array, mask: Array) -> Array:
    """Return the row-wise median over a non-empty mask."""

    count = jnp.sum(mask, axis=-1).astype(jnp.int32)
    sorted_values = jnp.sort(jnp.where(mask, values, jnp.inf), axis=-1)
    lower = jnp.maximum((count - 1) // 2, 0)
    upper = jnp.maximum(count // 2, 0)
    row = jnp.arange(values.shape[0], dtype=jnp.int32)
    return 0.5 * (
        sorted_values[row, lower] + sorted_values[row, upper]
    )


def _masked_softmax(logits: Array, mask: Array) -> Array:
    probabilities = jax.nn.softmax(
        jnp.where(mask, logits, -jnp.inf), axis=-1
    )
    return jnp.where(mask, probabilities, 0.0)


def _categorical_ess(probabilities: Array) -> Array:
    return jnp.reciprocal(
        jnp.maximum(jnp.sum(jnp.square(probabilities), axis=-1), 1e-12)
    )


def safe_particle_distribution(
    reward_q: Array,
    safety_q: Array,
    *,
    safety_threshold: float = 0.0,
    target_ess_fraction: float = 0.5,
    bisection_steps: int = 24,
) -> ParticleDistribution:
    """Build a feasible-reward-softmax / recovery-safety-softmin policy."""

    reward_q = jnp.asarray(reward_q, dtype=jnp.float32)
    safety_q = jnp.asarray(safety_q, dtype=jnp.float32)
    if reward_q.ndim != 2 or reward_q.shape != safety_q.shape:
        raise ValueError("reward_q and safety_q must have matched [B,N] shape.")
    if reward_q.shape[1] < 1:
        raise ValueError("At least one policy candidate is required.")
    if not 0.0 < target_ess_fraction <= 1.0:
        raise ValueError("target_ess_fraction must lie in (0,1].")

    finite_qr = jnp.isfinite(reward_q)
    finite_qh = jnp.isfinite(safety_q)
    fully_valid = finite_qr & finite_qh
    safety_feasible = finite_qh & (
        safety_q <= jnp.asarray(safety_threshold, dtype=jnp.float32)
    )
    feasible_branch = jnp.any(safety_feasible, axis=-1)
    reward_eligible = safety_feasible & finite_qr
    use_reward = feasible_branch & jnp.any(reward_eligible, axis=-1)
    recovery_eligible = finite_qh
    eligible = jnp.where(
        use_reward[:, None],
        reward_eligible,
        jnp.where(
            feasible_branch[:, None],
            safety_feasible,
            recovery_eligible,
        ),
    )
    eligible_count = jnp.sum(eligible, axis=-1).astype(jnp.float32)
    has_eligible = eligible_count > 0.0
    # Keep diagnostics numerically defined.  The formal runner rejects
    # eligible_count==0 before an environment action may be executed.
    eligible = jnp.where(
        has_eligible[:, None],
        eligible,
        jax.nn.one_hot(
            jnp.zeros((reward_q.shape[0],), dtype=jnp.int32),
            reward_q.shape[1],
            dtype=jnp.bool_,
        ),
    )
    eligible_count_safe = jnp.maximum(
        jnp.sum(eligible, axis=-1), 1
    ).astype(jnp.float32)

    branch_values = jnp.where(use_reward[:, None], reward_q, safety_q)
    median = _masked_median(branch_values, eligible)
    deviation = jnp.abs(branch_values - median[:, None])
    mad = _masked_median(deviation, eligible)
    has_spread = jnp.isfinite(mad) & (mad > 1e-8)
    normalized = jnp.where(
        has_spread[:, None],
        (branch_values - median[:, None])
        / jnp.maximum(1.4826 * mad[:, None], 1e-8),
        0.0,
    )
    signed_logits = jnp.where(
        use_reward[:, None], normalized, -normalized
    )
    signed_logits = jnp.clip(signed_logits, -40.0, 40.0)
    target_ess = jnp.maximum(
        1.0,
        jnp.minimum(
            target_ess_fraction * float(reward_q.shape[1]),
            eligible_count_safe,
        ),
    )

    low = jnp.full_like(target_ess, 1e-3)
    high = jnp.full_like(target_ess, 1e3)
    for _ in range(int(bisection_steps)):
        middle = jnp.sqrt(low * high)
        probabilities = _masked_softmax(
            signed_logits / middle[:, None], eligible
        )
        ess = _categorical_ess(probabilities)
        low = jnp.where(ess < target_ess, middle, low)
        high = jnp.where(ess < target_ess, high, middle)
    temperature = jnp.where(has_spread, jnp.sqrt(low * high), 1.0)
    probabilities = _masked_softmax(
        signed_logits / temperature[:, None], eligible
    )
    return ParticleDistribution(
        probabilities=probabilities,
        temperature=temperature,
        categorical_ess=_categorical_ess(probabilities),
        target_ess=target_ess,
        feasible_branch=feasible_branch,
        feasible_fraction=jnp.mean(
            safety_feasible.astype(jnp.float32), axis=-1
        ),
        no_feasible=~feasible_branch,
        eligible_count=eligible_count,
        mad=mad,
        invalid_candidate_fraction=jnp.mean(
            (~fully_valid).astype(jnp.float32), axis=-1
        ),
    )


def sample_safe_particle(
    reward_q: Array,
    safety_q: Array,
    rng: Array,
    *,
    safety_threshold: float = 0.0,
    target_ess_fraction: float = 0.5,
) -> ParticleSelection:
    """Sample one candidate per row from :func:`safe_particle_distribution`."""

    distribution = safe_particle_distribution(
        reward_q,
        safety_q,
        safety_threshold=safety_threshold,
        target_ess_fraction=target_ess_fraction,
    )
    log_probabilities = jnp.where(
        distribution.probabilities > 0.0,
        jnp.log(distribution.probabilities),
        -jnp.inf,
    )
    index = jax.random.categorical(rng, log_probabilities, axis=-1)
    row = jnp.arange(reward_q.shape[0], dtype=jnp.int32)
    selected_reward = reward_q[row, index]
    selected_safety = safety_q[row, index]
    return ParticleSelection(
        index=index.astype(jnp.int32),
        distribution=distribution,
        selected_probability=distribution.probabilities[row, index],
        selected_reward_rank=1
        + jnp.sum(
            (reward_q > selected_reward[:, None]).astype(jnp.int32), axis=-1
        ),
        selected_safety_rank=1
        + jnp.sum(
            (safety_q < selected_safety[:, None]).astype(jnp.int32), axis=-1
        ),
    )


__all__ = [
    "ParticleDistribution",
    "ParticleSelection",
    "safe_particle_distribution",
    "sample_safe_particle",
]

"""Numerically stable posterior-SNIS targets for bounded SSM actions.

The paper target is a density with respect to Lebesgue measure on the bounded
action box.  For a fixed VP-noised action ``x_t``, the unbounded inverse
likelihood proposal is wasteful because most samples leave the action box at
large noise levels.  Conditioning that Gaussian on the box does not change
the self-normalized clean-energy weights: its normalizer depends on ``x_t``
but not on the sampled clean action.

These target utilities are independent of the online agent.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr


Array = jax.Array


class PosteriorTarget(NamedTuple):
    """Posterior noise mean and per-outer-sample diagnostics."""

    epsilon: Array
    weights: Array
    valid: Array
    ess: Array
    max_weight: Array
    candidate_acceptance: Array


def open_unit_uniform(key: Array, shape: tuple[int, ...]) -> Array:
    """Draw float32 uniforms strictly inside ``(0, 1)``.

    JAX's ordinary float32 uniform may return its inclusive lower endpoint.
    Midpoints of the 23-bit mantissa grid rule out both 0 and 1 before
    inverse, logarithm, or ``atanh`` transforms.
    """
    bits = jax.random.bits(key, shape=shape, dtype=jnp.uint32)
    mantissa = jnp.right_shift(bits, jnp.asarray(9, dtype=jnp.uint32))
    return (mantissa.astype(jnp.float32) + 0.5) * jnp.asarray(
        2.0**-23, dtype=jnp.float32
    )


def normalize_log_weighted_noise(
    log_weight: Array,
    proposal_noise: Array,
    *,
    candidate_acceptance: Array | None = None,
) -> PosteriorTarget:
    """Normalize arbitrary log importance weights and average noise targets.

    Non-finite log weights are treated as zero mass.  All-zero rows are kept
    finite and marked invalid so the caller can remove them from the loss.
    """
    log_weight = jnp.asarray(log_weight, dtype=jnp.float32)
    proposal_noise = jnp.asarray(proposal_noise, dtype=jnp.float32)
    if log_weight.ndim != 2:
        raise ValueError(f"log_weight must have shape [B,M], got {log_weight.shape!r}.")
    if proposal_noise.ndim != 3 or proposal_noise.shape[:2] != log_weight.shape:
        raise ValueError(
            "proposal_noise must have shape [B,M,D] aligned with log_weight; got "
            f"{proposal_noise.shape!r}."
        )

    finite_noise = jnp.all(jnp.isfinite(proposal_noise), axis=-1)
    finite = jnp.isfinite(log_weight) & finite_noise
    valid = jnp.any(finite, axis=1)
    row_max = jnp.max(jnp.where(finite, log_weight, -jnp.inf), axis=1, keepdims=True)
    row_max = jnp.where(valid[:, None], row_max, 0.0)
    unnormalized = jnp.where(finite, jnp.exp(log_weight - row_max), 0.0)
    denominator = jnp.sum(unnormalized, axis=1, keepdims=True)
    weights = jnp.where(
        valid[:, None],
        unnormalized / jnp.maximum(denominator, jnp.finfo(jnp.float32).tiny),
        0.0,
    )
    finite_proposal_noise = jnp.where(
        finite_noise[:, :, None], proposal_noise, 0.0
    )
    epsilon = jnp.sum(weights[:, :, None] * finite_proposal_noise, axis=1)
    squared_weight_sum = jnp.sum(jnp.square(weights), axis=1)
    ess = jnp.where(valid, 1.0 / jnp.maximum(squared_weight_sum, 1e-30), 0.0)
    max_weight = jnp.max(weights, axis=1)
    if candidate_acceptance is None:
        candidate_acceptance = jnp.mean(finite.astype(jnp.float32), axis=1)
    else:
        candidate_acceptance = jnp.asarray(candidate_acceptance, dtype=jnp.float32)
        if candidate_acceptance.shape != (log_weight.shape[0],):
            raise ValueError(
                "candidate_acceptance must have shape [B], got "
                f"{candidate_acceptance.shape!r}."
            )
    return PosteriorTarget(
        epsilon=epsilon,
        weights=weights,
        valid=valid,
        ess=ess,
        max_weight=max_weight,
        candidate_acceptance=candidate_acceptance,
    )


def _as_batch_column(value: Array | float, batch_size: int, name: str) -> Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    if value.ndim == 0:
        return jnp.broadcast_to(value, (batch_size, 1))
    if value.shape == (batch_size,):
        return value[:, None]
    if value.shape == (batch_size, 1):
        return value
    raise ValueError(
        f"{name} must be scalar, [B], or [B,1]; got shape {value.shape!r}."
    )


def sample_box_truncated_likelihood(
    key: Array,
    noisy_actions: Array,
    alpha: Array | float,
    sigma: Array | float,
    num_samples: int,
    *,
    action_low: float = -1.0,
    action_high: float = 1.0,
    max_rejection_iterations: int = 1024,
) -> tuple[Array, Array]:
    r"""Sample the exact likelihood proposal conditioned on the action box.

    For ``x = alpha * a + sigma * epsilon``, this samples

    ``r(a | x) = Normal(x / alpha, sigma^2 / alpha^2 I) | a in [low, high]^d``.

    The returned ``proposal_noise`` is the forward noise implied by each
    sampled action, i.e. ``(x - alpha*a) / sigma``.  It is not standard normal
    after truncation; it is nevertheless exactly the noise whose posterior
    expectation defines the DDPM epsilon target.
    """
    noisy_actions = jnp.asarray(noisy_actions, dtype=jnp.float32)
    if noisy_actions.ndim != 2:
        raise ValueError(
            f"noisy_actions must have shape [B,D], got {noisy_actions.shape!r}."
        )
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if not action_low < action_high:
        raise ValueError(
            f"Expected action_low < action_high, got {action_low}, {action_high}."
        )
    if max_rejection_iterations <= 0:
        raise ValueError(
            "max_rejection_iterations must be positive, got "
            f"{max_rejection_iterations}."
        )
    batch_size, action_dim = noisy_actions.shape
    alpha_col = _as_batch_column(alpha, batch_size, "alpha")
    sigma_col = _as_batch_column(sigma, batch_size, "sigma")
    if isinstance(alpha, (float, int)) and float(alpha) <= 0.0:
        raise ValueError("alpha must be positive.")
    if isinstance(sigma, (float, int)) and float(sigma) <= 0.0:
        raise ValueError("sigma must be positive.")

    # Array-valued schedules may be traced under JIT, so Python exceptions are
    # unavailable for their runtime values.  Mark every invalid batch/dimension
    # lane and use finite surrogates only to keep the rejection loop itself
    # well-defined.  Invalid lanes are emitted as NaN below and are therefore
    # removed by ``normalize_log_weighted_noise`` rather than approximated.
    parameter_valid = (
        jnp.isfinite(alpha_col)
        & jnp.isfinite(sigma_col)
        & (alpha_col > 0.0)
        & (sigma_col > 0.0)
    )
    lane_valid = jnp.isfinite(noisy_actions) & parameter_valid
    safe_alpha = jnp.where(parameter_valid, alpha_col, 1.0)
    safe_sigma = jnp.where(parameter_valid, sigma_col, 1.0)
    safe_noisy_actions = jnp.where(jnp.isfinite(noisy_actions), noisy_actions, 0.0)
    mean = safe_noisy_actions / safe_alpha
    std = safe_sigma / safe_alpha

    lower_z = (jnp.asarray(action_low, dtype=jnp.float32) - mean) / std
    upper_z = (jnp.asarray(action_high, dtype=jnp.float32) - mean) / std
    sample_shape = (batch_size, num_samples, action_dim)
    lower = jnp.broadcast_to(lower_z[:, None, :], sample_shape)
    upper = jnp.broadcast_to(upper_z[:, None, :], sample_shape)

    # Tail-stable rejection sampler for N(0,1) restricted to [lower, upper].
    # Positive/negative tails use an exponential proposal, central narrow
    # intervals use a uniform proposal, and central wide intervals reject a
    # standard normal.  This avoids float32 CDF cancellation and never clips
    # accepted samples into artificial atoms at the action boundary.
    positive_tail = lower >= 0.0
    negative_tail = upper <= 0.0
    central = ~(positive_tail | negative_tail)
    central_mass = ndtr(upper) - ndtr(lower)
    central_normal = central & (central_mass >= 0.25)

    def positive_tail_candidate(
        tail_lower: Array, tail_upper: Array, uniform: Array
    ) -> tuple[Array, Array]:
        rate = 0.5 * (tail_lower + jnp.sqrt(jnp.square(tail_lower) + 4.0))
        truncated_mass = -jnp.expm1(-rate * (tail_upper - tail_lower))
        candidate = tail_lower - jnp.log1p(-uniform * truncated_mass) / rate
        log_acceptance = -0.5 * jnp.square(candidate - rate)
        return candidate, log_acceptance

    def condition(state):
        _, accepted, iteration = state
        return jnp.any(~accepted) & (iteration < max_rejection_iterations)

    def body(state):
        samples, accepted, iteration = state
        iteration_key = jax.random.fold_in(key, iteration)
        proposal_key, acceptance_key, normal_key = jax.random.split(
            iteration_key, 3
        )
        proposal_uniform = open_unit_uniform(proposal_key, sample_shape)
        acceptance_uniform = open_unit_uniform(acceptance_key, sample_shape)

        positive_candidate, positive_log_accept = positive_tail_candidate(
            lower, upper, proposal_uniform
        )
        reflected_candidate, negative_log_accept = positive_tail_candidate(
            -upper, -lower, proposal_uniform
        )
        negative_candidate = -reflected_candidate
        uniform_candidate = lower + (upper - lower) * proposal_uniform
        normal_candidate = jax.random.normal(
            normal_key, sample_shape, dtype=jnp.float32
        )

        candidate = jnp.where(
            positive_tail,
            positive_candidate,
            jnp.where(
                negative_tail,
                negative_candidate,
                jnp.where(central_normal, normal_candidate, uniform_candidate),
            ),
        )
        log_u = jnp.log(acceptance_uniform)
        tail_accept = jnp.where(
            positive_tail,
            log_u <= positive_log_accept,
            log_u <= negative_log_accept,
        )
        normal_accept = (normal_candidate >= lower) & (normal_candidate <= upper)
        uniform_accept = log_u <= (-0.5 * jnp.square(uniform_candidate))
        proposal_accept = jnp.where(
            positive_tail | negative_tail,
            tail_accept,
            jnp.where(central_normal, normal_accept, uniform_accept),
        )
        newly_accepted = (~accepted) & proposal_accept & jnp.isfinite(candidate)
        samples = jnp.where(newly_accepted, candidate, samples)
        return samples, accepted | newly_accepted, iteration + 1

    standard_samples, accepted, _ = jax.lax.while_loop(
        condition,
        body,
        (
            jnp.zeros(sample_shape, dtype=jnp.float32),
            # Invalid lanes are excluded from the loop from the start.  They
            # are restored to NaN after sampling so downstream rows fail closed.
            jnp.broadcast_to((~lane_valid)[:, None, :], sample_shape),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    successful = jnp.broadcast_to(lane_valid[:, None, :], sample_shape) & accepted
    sampled_actions = mean[:, None, :] + std[:, None, :] * standard_samples
    actions = jnp.where(successful, sampled_actions, jnp.nan)

    sampled_noise = (
        safe_noisy_actions[:, None, :] - safe_alpha[:, None, :] * sampled_actions
    ) / safe_sigma[:, None, :]
    proposal_noise = jnp.where(successful, sampled_noise, jnp.nan)
    return actions, proposal_noise


def posterior_noise_target(
    reward_q: Array,
    safety_q: Array,
    proposal_noise: Array,
    feasible_state: Array,
    *,
    alpha_reward: float,
    beta_safety: float,
    safety_threshold: float = 0.0,
) -> PosteriorTarget:
    r"""Compute the hard-feasible/recovery posterior epsilon target.

    Feasible-state clean weights are
    ``exp(alpha_reward * Q_r) 1[Q_h <= threshold]``.  Recovery-state weights
    are ``exp(-beta_safety * Q_h)``.  Rows with no feasible candidate retain
    zero weights and are marked invalid; callers must skip them rather than
    silently falling back to unsafe uniform weights.
    """
    reward_q = jnp.asarray(reward_q, dtype=jnp.float32)
    safety_q = jnp.asarray(safety_q, dtype=jnp.float32)
    proposal_noise = jnp.asarray(proposal_noise, dtype=jnp.float32)
    feasible_state = jnp.asarray(feasible_state, dtype=jnp.bool_)
    if reward_q.ndim != 2 or reward_q.shape != safety_q.shape:
        raise ValueError(
            "reward_q and safety_q must have the same [B,M] shape; got "
            f"{reward_q.shape!r} and {safety_q.shape!r}."
        )
    if proposal_noise.ndim != 3 or proposal_noise.shape[:2] != reward_q.shape:
        raise ValueError(
            "proposal_noise must have shape [B,M,D] aligned with Q values; got "
            f"{proposal_noise.shape!r}."
        )
    if feasible_state.shape not in {(reward_q.shape[0],), (reward_q.shape[0], 1)}:
        raise ValueError(
            f"feasible_state must have shape [B] or [B,1], got {feasible_state.shape!r}."
        )
    feasible_state = feasible_state.reshape((-1, 1))

    feasible_support = safety_q <= jnp.asarray(safety_threshold, jnp.float32)
    feasible_log_weight = jnp.where(
        feasible_support,
        jnp.asarray(alpha_reward, jnp.float32) * reward_q,
        -jnp.inf,
    )
    recovery_log_weight = -jnp.asarray(beta_safety, jnp.float32) * safety_q
    log_weight = jnp.where(feasible_state, feasible_log_weight, recovery_log_weight)
    candidate_acceptance = jnp.where(
        feasible_state[:, 0],
        jnp.mean(feasible_support.astype(jnp.float32), axis=1),
        1.0,
    )
    return normalize_log_weighted_noise(
        log_weight,
        proposal_noise,
        candidate_acceptance=candidate_acceptance,
    )


def stable_log_tanh_jacobian(latent: Array) -> Array:
    r"""Return ``sum_j log(1 - tanh(z_j)^2)`` without saturation NaNs."""
    latent = jnp.asarray(latent, dtype=jnp.float32)
    per_dim = 2.0 * (
        jnp.log(jnp.asarray(2.0, dtype=jnp.float32))
        - latent
        - jax.nn.softplus(-2.0 * latent)
    )
    return jnp.sum(per_dim, axis=-1)


def actions_to_latent(actions: Array, boundary_epsilon: float = 1e-6) -> Array:
    """Map bounded actions to finite tanh latents for replay/outer sampling."""
    if not 0.0 < boundary_epsilon < 1.0:
        raise ValueError("boundary_epsilon must lie in (0, 1).")
    actions = jnp.asarray(actions, dtype=jnp.float32)
    clipped = jnp.clip(actions, -1.0 + boundary_epsilon, 1.0 - boundary_epsilon)
    return jnp.arctanh(clipped)


def latent_clean_log_weight(
    reward_q: Array,
    safety_q: Array,
    latent_actions: Array,
    feasible_state: Array,
    *,
    alpha_reward: float,
    beta_safety: float,
    safety_threshold: float = 0.0,
) -> Array:
    """Unnormalized clean log-density in tanh latent coordinates.

    The Jacobian is required because the paper target is defined relative to
    action-space Lebesgue measure, not relative to a latent-space base measure.
    """
    jacobian = stable_log_tanh_jacobian(latent_actions)
    feasible_state = jnp.asarray(feasible_state, dtype=jnp.bool_).reshape((-1, 1))
    feasible = (
        jnp.asarray(alpha_reward, jnp.float32) * reward_q
        + jacobian
    )
    feasible = jnp.where(safety_q <= safety_threshold, feasible, -jnp.inf)
    recovery = -jnp.asarray(beta_safety, jnp.float32) * safety_q + jacobian
    return jnp.where(feasible_state, feasible, recovery)

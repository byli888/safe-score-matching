"""Closed-form proposals for defensive reference-bridge MIS.

This module changes only the proposal used to estimate the existing LP-PS
posterior-noise expectation.  It does not define the clean Gibbs target, the
feasible/recovery branch, or the actor anchor.

For ``z = alpha * u + sigma * epsilon`` and a frozen-reference centre
``u_bar``, the normalized product

``q(z | u) N(u; u_bar, bridge_latent_std**2 I)``

is an isotropic Gaussian in ``u``.  The functions below expose both the
closed-form parameters and the *complete* defensive-mixture density, so every
sample can be corrected with the same balance-heuristic denominator.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array
LOG_TWO_PI = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32))

BRIDGE_COMPONENT_WEIGHTING_UNIFORM = "uniform"
BRIDGE_COMPONENT_WEIGHTING_EVIDENCE = "evidence"
BRIDGE_COMPONENT_WEIGHTINGS = (
    BRIDGE_COMPONENT_WEIGHTING_UNIFORM,
    BRIDGE_COMPONENT_WEIGHTING_EVIDENCE,
)


class DefensiveBridgeProposal(NamedTuple):
    """Samples and their full deterministic-mixture proposal density."""

    latents: Array
    log_density: Array
    source_is_likelihood: Array
    likelihood_fraction_actual: Array
    bridge_component_indices: Array
    bridge_means: Array
    bridge_std: Array
    bridge_log_component_weights: Array


def isotropic_normal_log_density(
    value: Array, mean: Array, std: Array
) -> Array:
    """Log density of an isotropic Normal along the final dimension."""

    value = jnp.asarray(value, dtype=jnp.float32)
    mean = jnp.asarray(mean, dtype=jnp.float32)
    std = jnp.asarray(std, dtype=jnp.float32)
    standardized = (value - mean) / std
    dimension = value.shape[-1]
    return (
        -0.5 * jnp.sum(jnp.square(standardized), axis=-1)
        - dimension * jnp.log(std[..., 0])
        - 0.5 * dimension * LOG_TWO_PI
    )


def reference_bridge_parameters(
    noisy_latent: Array,
    alpha: Array,
    sigma: Array,
    reference_centers: Array,
    *,
    bridge_latent_std: float,
    component_weighting: str = BRIDGE_COMPONENT_WEIGHTING_UNIFORM,
) -> tuple[Array, Array, Array]:
    r"""Return means, std, and normalized component log weights.

    ``alpha`` and ``sigma`` are the forward *marginal* coefficients
    ``sqrt(alpha_hat_t)`` and ``sqrt(1-alpha_hat_t)``.  ``sigma`` is not the
    per-step DDPM beta.

    ``uniform`` assigns equal weights to individually normalized product
    Gaussians, yielding a valid importance proposal.  ``evidence`` instead
    returns the exact conditional of an equal-weight reference Gaussian KDE;
    its responsibilities are proportional to

    ``N(z; alpha * u_bar_j, sigma**2 + alpha**2 * sigma_p**2)``.
    """

    if bridge_latent_std <= 0.0:
        raise ValueError("bridge_latent_std must be positive.")
    if component_weighting not in BRIDGE_COMPONENT_WEIGHTINGS:
        raise ValueError(
            "component_weighting must be one of "
            f"{BRIDGE_COMPONENT_WEIGHTINGS}, got {component_weighting!r}."
        )
    noisy_latent = jnp.asarray(noisy_latent, dtype=jnp.float32)
    reference_centers = jnp.asarray(reference_centers, dtype=jnp.float32)
    if noisy_latent.ndim != 2:
        raise ValueError("noisy_latent must have shape [B,A].")
    if reference_centers.ndim != 3:
        raise ValueError("reference_centers must have shape [B,J,A].")
    if (
        reference_centers.shape[0] != noisy_latent.shape[0]
        or reference_centers.shape[2] != noisy_latent.shape[1]
        or reference_centers.shape[1] <= 0
    ):
        raise ValueError(
            "reference_centers must match noisy_latent batch/action dimensions "
            "and contain at least one centre."
        )

    batch_size = noisy_latent.shape[0]
    alpha = jnp.asarray(alpha, dtype=jnp.float32).reshape((batch_size, 1))
    sigma = jnp.asarray(sigma, dtype=jnp.float32).reshape((batch_size, 1))
    sigma_p = jnp.asarray(bridge_latent_std, dtype=jnp.float32)
    variance = 1.0 / (
        jnp.square(alpha) / jnp.square(sigma) + 1.0 / jnp.square(sigma_p)
    )
    bridge_std = jnp.sqrt(variance)
    bridge_means = variance[:, None, :] * (
        alpha[:, None, :] * noisy_latent[:, None, :] / jnp.square(sigma[:, None, :])
        + reference_centers / jnp.square(sigma_p)
    )

    reference_count = reference_centers.shape[1]
    if component_weighting == BRIDGE_COMPONENT_WEIGHTING_UNIFORM:
        log_component_weights = jnp.full(
            (batch_size, reference_count),
            -jnp.log(jnp.asarray(reference_count, dtype=jnp.float32)),
            dtype=jnp.float32,
        )
    else:
        evidence_std = jnp.sqrt(
            jnp.square(sigma) + jnp.square(alpha) * jnp.square(sigma_p)
        )
        log_evidence = isotropic_normal_log_density(
            noisy_latent[:, None, :],
            alpha[:, None, :] * reference_centers,
            evidence_std[:, None, :],
        )
        log_component_weights = jax.nn.log_softmax(log_evidence, axis=1)
    return bridge_means, bridge_std, log_component_weights


def reference_bridge_log_density(
    latents: Array,
    bridge_means: Array,
    bridge_std: Array,
    bridge_log_component_weights: Array,
) -> Array:
    """Evaluate the complete normalized reference-bridge GMM density."""

    latents = jnp.asarray(latents, dtype=jnp.float32)
    if latents.ndim != 3 or bridge_means.ndim != 3:
        raise ValueError("latents/bridge_means must have shapes [B,M,A]/[B,J,A].")
    component_log_density = isotropic_normal_log_density(
        latents[:, :, None, :],
        bridge_means[:, None, :, :],
        bridge_std[:, None, None, :],
    )
    return jax.scipy.special.logsumexp(
        bridge_log_component_weights[:, None, :] + component_log_density,
        axis=2,
    )


def sample_defensive_bridge_proposal(
    rng: Array,
    noisy_latent: Array,
    alpha: Array,
    sigma: Array,
    reference_centers: Array,
    *,
    num_samples: int,
    likelihood_fraction: float,
    bridge_latent_std: float,
    component_weighting: str = BRIDGE_COMPONENT_WEIGHTING_UNIFORM,
) -> DefensiveBridgeProposal:
    r"""Sample fixed-count defensive MIS candidates and evaluate ``r_mix``.

    The number drawn from each top-level source is deterministic, with the
    actual fraction ``round(M * omega) / M`` used in the mixture denominator.
    Bridge component indices are i.i.d. from the declared component weights.
    Every returned sample is evaluated under the *full* mixture density, never
    only under the component that generated it.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not 0.0 <= likelihood_fraction <= 1.0:
        raise ValueError("likelihood_fraction must lie in [0,1].")
    noisy_latent = jnp.asarray(noisy_latent, dtype=jnp.float32)
    if noisy_latent.ndim != 2:
        raise ValueError("noisy_latent must have shape [B,A].")
    batch_size, action_dim = noisy_latent.shape
    alpha = jnp.asarray(alpha, dtype=jnp.float32).reshape((batch_size, 1))
    sigma = jnp.asarray(sigma, dtype=jnp.float32).reshape((batch_size, 1))
    likelihood_count = int(round(num_samples * likelihood_fraction))
    bridge_count = num_samples - likelihood_count
    actual_fraction = likelihood_count / num_samples

    bridge_means, bridge_std, bridge_log_component_weights = (
        reference_bridge_parameters(
            noisy_latent,
            alpha,
            sigma,
            reference_centers,
            bridge_latent_std=bridge_latent_std,
            component_weighting=component_weighting,
        )
    )
    likelihood_key, component_key, bridge_noise_key = jax.random.split(rng, 3)
    likelihood_mean = noisy_latent / alpha
    likelihood_std = sigma / alpha
    proposal_parts = []

    if likelihood_count:
        likelihood_noise = jax.random.normal(
            likelihood_key,
            (batch_size, likelihood_count, action_dim),
            dtype=jnp.float32,
        )
        proposal_parts.append(
            likelihood_mean[:, None, :]
            + likelihood_std[:, None, :] * likelihood_noise
        )

    if bridge_count:
        component_indices = jax.random.categorical(
            component_key,
            bridge_log_component_weights[:, None, :],
            axis=-1,
            shape=(batch_size, bridge_count),
        ).astype(jnp.int32)
        selected_means = bridge_means[
            jnp.arange(batch_size)[:, None], component_indices, :
        ]
        bridge_noise = jax.random.normal(
            bridge_noise_key,
            (batch_size, bridge_count, action_dim),
            dtype=jnp.float32,
        )
        proposal_parts.append(
            selected_means + bridge_std[:, None, :] * bridge_noise
        )
    else:
        component_indices = jnp.zeros((batch_size, 0), dtype=jnp.int32)

    proposal_latents = (
        proposal_parts[0]
        if len(proposal_parts) == 1
        else jnp.concatenate(proposal_parts, axis=1)
    )
    log_likelihood_proposal = isotropic_normal_log_density(
        proposal_latents,
        likelihood_mean[:, None, :],
        likelihood_std[:, None, :],
    )
    log_bridge_proposal = reference_bridge_log_density(
        proposal_latents,
        bridge_means,
        bridge_std,
        bridge_log_component_weights,
    )
    if actual_fraction == 1.0:
        log_proposal = log_likelihood_proposal
    elif actual_fraction == 0.0:
        log_proposal = log_bridge_proposal
    else:
        log_proposal = jnp.logaddexp(
            jnp.log(jnp.asarray(actual_fraction, dtype=jnp.float32))
            + log_likelihood_proposal,
            jnp.log1p(-jnp.asarray(actual_fraction, dtype=jnp.float32))
            + log_bridge_proposal,
        )
    source_is_likelihood = jnp.arange(num_samples) < likelihood_count
    return DefensiveBridgeProposal(
        latents=proposal_latents,
        log_density=log_proposal,
        source_is_likelihood=source_is_likelihood,
        likelihood_fraction_actual=jnp.asarray(
            actual_fraction, dtype=jnp.float32
        ),
        bridge_component_indices=component_indices,
        bridge_means=bridge_means,
        bridge_std=bridge_std,
        bridge_log_component_weights=bridge_log_component_weights,
    )


__all__ = [
    "BRIDGE_COMPONENT_WEIGHTING_EVIDENCE",
    "BRIDGE_COMPONENT_WEIGHTING_UNIFORM",
    "BRIDGE_COMPONENT_WEIGHTINGS",
    "DefensiveBridgeProposal",
    "isotropic_normal_log_density",
    "reference_bridge_log_density",
    "reference_bridge_parameters",
    "sample_defensive_bridge_proposal",
]

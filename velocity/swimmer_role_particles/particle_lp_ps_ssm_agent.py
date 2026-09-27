"""Latent Proximal Posterior-SNIS actor for online SSM.

LPPSAgent owns every actor-facing code path. SSMOnlineAgent supplies the
reward/safety critics and Bellman updates; its clipped DDPM action sampler and
direct-Q/dead-zone-fill actor update are not called here.

The policy is one epsilon network in an unconstrained latent coordinate u.
Environment and critic actions are tanh(u). A frozen parameter snapshot of
that network supplies the outer clean sample and proximal/reference epsilon
target during each policy-improvement block.

Posterior proposals can use the forward likelihood alone or balance-heuristic
mixtures with a uniform-action proposal or a Gaussian bridge around frozen
reference-policy samples. The released velocity recipes use the defensive
reference bridge. All proposal families target the same action-Lebesgue Gibbs
density and include the tanh Jacobian; the proposal mode selects how it is
estimated, not a different target measure.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import optax

from lp_ps_bridge_mis import (
    BRIDGE_COMPONENT_WEIGHTING_UNIFORM,
    BRIDGE_COMPONENT_WEIGHTINGS,
    sample_defensive_bridge_proposal,
)
from swimmer_role_particles.particle_selection import (
    ParticleSelection,
    sample_safe_particle,
)
from posterior_score_snis import (
    actions_to_latent,
    normalize_log_weighted_noise,
    open_unit_uniform,
    stable_log_tanh_jacobian,
)
from ssm_agent_v51_hard_dzfill_qhstage_v_2 import SSMOnlineAgent


Array = jax.Array
LOG_TWO_PI = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32))

LP_ACTOR_COORDINATE = "tanh_latent_v1"
LP_TARGET_ESTIMATOR = "reference_anchored_posterior_snis_v1"
LP_PROPOSAL_MODE = "forward_likelihood_only"
LP_BALANCE_MIS_PROPOSAL_MODE = "likelihood_uniform_action_balance_mis"
LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE = (
    "defensive_reference_bridge_balance_mis_v1"
)
LP_LEGACY_AUTO_PROPOSAL_MODE = "legacy_auto"
LP_LEGACY_ACTOR_USED = False
LP_ACTOR_SAFETY_QH_ROUTED = "qh_routed"
LP_ACTOR_SAFETY_REWARD_ONLY = "reward_only"
LP_ACTOR_SAFETY_MODES = (
    LP_ACTOR_SAFETY_QH_ROUTED,
    LP_ACTOR_SAFETY_REWARD_ONLY,
)
LP_PARTICLE_MODE_S1_MIN = "s1_min"
LP_PARTICLE_HJ_ROLE_SEPARATED = "role_separated"
LP_PARTICLE_HJ_FULL_CONSISTENT = "full_consistent"
LP_PARTICLE_MODES = (
    LP_PARTICLE_MODE_S1_MIN,
    LP_PARTICLE_HJ_ROLE_SEPARATED,
    LP_PARTICLE_HJ_FULL_CONSISTENT,
)
LP_REWARD_BACKUP_ONLINE_K1 = "online_policy_k1"
LP_REWARD_BACKUP_REFERENCE_MEAN = "reference_policy_mean"
LP_REWARD_BACKUP_MODES = (
    LP_REWARD_BACKUP_ONLINE_K1,
    LP_REWARD_BACKUP_REFERENCE_MEAN,
)
LP_REWARD_TILT_FIXED_ALPHA = "fixed_alpha"
LP_REWARD_TILT_KL_BUDGET = "kl_budget"
LP_REWARD_TILT_BATCH_KL = "batch_kl"
LP_REWARD_TILT_CAPPED_BATCH_KL = "capped_batch_kl"
LP_REWARD_TILT_MODES = (
    LP_REWARD_TILT_FIXED_ALPHA,
    LP_REWARD_TILT_KL_BUDGET,
    LP_REWARD_TILT_BATCH_KL,
    LP_REWARD_TILT_CAPPED_BATCH_KL,
)
PARTICLE_BEHAVIOR_INFO_KEYS = (
    "particle_data_active",
    "particle_data_n",
    "particle_data_temperature",
    "particle_data_ess",
    "particle_data_target_ess",
    "particle_data_feasible_fraction",
    "particle_data_no_feasible",
    "particle_data_eligible_count",
    "particle_data_invalid_candidate_fraction",
    "particle_data_selected_index",
    "particle_data_selected_probability",
    "particle_data_selected_reward_rank",
    "particle_data_selected_safety_rank",
    "particle_data_selected_qr",
    "particle_data_selected_qh",
    "particle_data_selected_vs_particle0_l2",
    "particle_data_row_valid",
    "particle_data_rng_routing_valid",
    "particle_data_key_collision_rate",
)


class LPPolicyParticleBatch(NamedTuple):
    """Independent full-policy candidates and one safe-softmax selection."""

    latents: Array
    actions: Array
    reward_q: Array
    safety_q: Array
    selection: ParticleSelection
    selected_latents: Array
    selected_actions: Array
    rng_routing_valid: Array
    key_collision_rate: Array


def lp_proposal_mode(
    likelihood_fraction: float, proposal_mode: str | None = None
) -> str:
    """Resolve a proposal mode without changing legacy fraction semantics."""

    if not 0.0 <= likelihood_fraction <= 1.0:
        raise ValueError("likelihood_fraction must lie in [0,1].")
    legacy_mode = (
        LP_PROPOSAL_MODE
        if likelihood_fraction == 1.0
        else LP_BALANCE_MIS_PROPOSAL_MODE
    )
    if proposal_mode in (None, LP_LEGACY_AUTO_PROPOSAL_MODE):
        return legacy_mode
    if proposal_mode == LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE:
        return proposal_mode
    if proposal_mode in (LP_PROPOSAL_MODE, LP_BALANCE_MIS_PROPOSAL_MODE):
        if proposal_mode != legacy_mode:
            raise ValueError(
                "Explicit legacy proposal_mode conflicts with "
                f"likelihood_fraction={likelihood_fraction}."
            )
        return proposal_mode
    raise ValueError(f"Unsupported proposal_mode {proposal_mode!r}.")


class LPPosteriorTarget(NamedTuple):
    """Posterior epsilon mean plus testable per-row diagnostics.

    ``*_component_ess`` is conditional ESS for the two top-level proposal
    sources (likelihood versus the complete reference-bridge GMM).  It is not
    a separate ESS for every one of the ``J`` reference centres.
    """

    epsilon: Array
    weights: Array
    valid: Array
    ess: Array
    max_weight: Array
    feasible_acceptance: Array
    critic_nonfinite_fraction: Array
    likelihood_weight_mass: Array
    likelihood_fraction_actual: Array
    likelihood_component_ess: Array
    likelihood_feasible_acceptance: Array
    reference_bridge_weight_mass: Array
    reference_bridge_fraction_actual: Array
    reference_bridge_component_ess: Array
    reference_bridge_feasible_acceptance: Array
    reference_center_latent_norm_mean: Array
    likelihood_sample_count: Array
    reference_bridge_sample_count: Array
    proposal_latents: Array
    proposal_actions: Array
    proposal_noise: Array
    reward_q_min: Array
    safety_q_max: Array
    reward_tilt_achieved_kl: Array
    reward_tilt_target_hit: Array
    reward_tilt_attainable: Array
    reward_tilt_ess_over_m: Array
    reward_tilt_max_weight: Array
    reward_tilt_q_gain: Array
    reward_tilt_flat: Array
    reward_tilt_active: Array
    reward_tilt_row_cap_active: Array
    reward_tilt_effective_kappa: Array
    reward_tilt_tau_proxy: Array


class KLBudgetRewardTilt(NamedTuple):
    """Per-row reward tilt under a KL trust-region around an MIS base."""

    log_weight: Array
    weights: Array
    valid: Array
    achieved_kl: Array
    target_hit: Array
    attainable: Array
    ess_over_m: Array
    max_weight: Array
    q_gain: Array
    flat: Array
    active: Array
    row_cap_active: Array
    effective_kappa: Array
    tau_proxy: Array


class LPBlendTarget(NamedTuple):
    """Reference blend and explicit fallback masks."""

    epsilon: Array
    effective_eta: Array
    invalid_feasible: Array
    invalid_any: Array


def tree_global_norm(tree: Any) -> Array:
    """Return an L2 norm over every array leaf of a PyTree."""

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(jnp.asarray(leaf))) for leaf in leaves)
    )


def copy_params(params: Any) -> Any:
    """Make an explicit immutable-array copy for a frozen reference actor."""

    return jax.tree_util.tree_map(lambda value: jnp.array(value), params)


def bounded_latent_to_action(
    latent: Array, boundary_epsilon: float = 1e-5
) -> Array:
    """Map latent values to the strict interior of the normalized action box.

    The mathematical map is exactly ``tanh``.  ``nextafter`` changes only a
    float32 endpoint produced by numerical saturation; it neither clips a
    reverse state nor shrinks the target action domain.
    """

    if not 0.0 < boundary_epsilon < 1.0:
        raise ValueError("boundary_epsilon must lie in (0,1).")
    del boundary_epsilon  # reserved for the inverse-map numerical guard
    action = jnp.tanh(latent)
    one = jnp.asarray(1.0, dtype=action.dtype)
    zero = jnp.asarray(0.0, dtype=action.dtype)
    upper = jnp.nextafter(one, zero)
    lower = jnp.nextafter(-one, zero)
    return jnp.where(action >= one, upper, jnp.where(action <= -one, lower, action))


def stable_log_bounded_tanh_jacobian(
    latent: Array, boundary_epsilon: float = 1e-5
) -> Array:
    """Log absolute Jacobian of the exact ``tanh`` map."""

    if not 0.0 < boundary_epsilon < 1.0:
        raise ValueError("boundary_epsilon must lie in (0,1).")
    del boundary_epsilon  # inverse-map guard; not part of the target density
    return stable_log_tanh_jacobian(jnp.asarray(latent, dtype=jnp.float32))


def bounded_actions_to_latent(
    actions: Array, boundary_epsilon: float = 1e-5
) -> tuple[Array, Array]:
    """Invert ``tanh`` with the configured epsilon guard and clip mask."""

    if not 0.0 < boundary_epsilon < 1.0:
        raise ValueError("boundary_epsilon must lie in (0,1).")
    actions = jnp.asarray(actions, dtype=jnp.float32)
    clip_mask = (actions <= -1.0 + boundary_epsilon) | (
        actions >= 1.0 - boundary_epsilon
    )
    clipped = jnp.clip(
        actions, -1.0 + boundary_epsilon, 1.0 - boundary_epsilon
    )
    return jnp.arctanh(clipped), clip_mask


def reference_anchored_epsilon_target(
    reference_epsilon: Array,
    posterior: LPPosteriorTarget,
    feasible_state: Array,
    *,
    eta: float,
) -> LPBlendTarget:
    """Blend a valid posterior target, or fall back exactly to reference.

    This is a small public helper so tests can assert that an all-zero feasible
    support never becomes unsafe uniform weighting.  Invalid recovery rows
    (possible only through non-finite teacher/proposal values) also fail closed
    to the reference actor.
    """

    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0,1].")
    reference_epsilon = jnp.asarray(reference_epsilon, dtype=jnp.float32)
    feasible_state = jnp.asarray(feasible_state, dtype=jnp.bool_).reshape((-1,))
    if reference_epsilon.shape != posterior.epsilon.shape:
        raise ValueError("reference and posterior epsilon shapes differ.")
    if feasible_state.shape != posterior.valid.shape:
        raise ValueError("feasible_state and posterior valid shapes differ.")
    valid = posterior.valid
    effective_eta = jnp.asarray(eta, jnp.float32) * valid.astype(jnp.float32)
    epsilon = reference_epsilon + effective_eta[:, None] * (
        posterior.epsilon - reference_epsilon
    )
    return LPBlendTarget(
        epsilon=jax.lax.stop_gradient(epsilon),
        effective_eta=effective_eta,
        invalid_feasible=feasible_state & (~valid),
        invalid_any=~valid,
    )


def make_lp_actor_train_state(
    *,
    apply_fn,
    params: Any,
    actor_lr: float = 1e-4,
    grad_clip_norm: float = 1.0,
) -> TrainState:
    """Create the prescribed Adam actor state with global-norm clipping.

    This helper intentionally creates a fresh optimizer state.  It is for the
    from-scratch LP-PS-SSM coordinate system, not for silently resuming an old
    direct-Q actor optimizer.
    """

    if actor_lr <= 0.0:
        raise ValueError(f"actor_lr must be positive, got {actor_lr}.")
    if grad_clip_norm <= 0.0:
        raise ValueError(
            f"grad_clip_norm must be positive, got {grad_clip_norm}."
        )
    tx = optax.chain(
        optax.clip_by_global_norm(float(grad_clip_norm)),
        optax.adam(float(actor_lr)),
    )
    return TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def rebuild_actor_optimizer(
    core: SSMOnlineAgent,
    *,
    actor_lr: float = 1e-4,
    grad_clip_norm: float = 1.0,
) -> SSMOnlineAgent:
    """Replace only the score-model optimizer and disable legacy clipping.

    The network parameters and apply function are preserved.  Optimizer
    moments/step are reset by design because LP-PS-SSM is a from-scratch run.
    ``clip_sampler=False`` is metadata-level defense in depth; all sampling in
    this module uses :func:`latent_ddpm_sampler` regardless of that field.
    """

    score_model = make_lp_actor_train_state(
        apply_fn=core.score_model.apply_fn,
        params=core.score_model.params,
        actor_lr=actor_lr,
        grad_clip_norm=grad_clip_norm,
    )
    return core.replace(score_model=score_model, clip_sampler=False)


@partial(
    jax.jit,
    static_argnames=(
        "actor_apply_fn",
        "num_steps",
        "action_dim",
        "repeat_last_step",
    ),
)
def latent_ddpm_sampler(
    actor_apply_fn,
    actor_params: Any,
    *,
    num_steps: int,
    rng: Array,
    action_dim: int,
    observations: Array,
    alphas: Array,
    alpha_hats: Array,
    betas: Array,
    sample_temperature: float,
    repeat_last_step: int = 0,
) -> tuple[Array, Array]:
    """Sample unconstrained DDPM latents without intermediate/final clipping."""

    batch_size = observations.shape[0]

    def reverse_step(carry, time):
        latent, step_rng = carry
        time_input = jnp.full((batch_size, 1), time, dtype=jnp.int32)
        epsilon = actor_apply_fn(
            {"params": actor_params},
            observations,
            latent,
            time_input,
            training=False,
        )
        inverse_sqrt_alpha = jax.lax.rsqrt(alphas[time])
        epsilon_scale = (1.0 - alphas[time]) / jnp.sqrt(
            1.0 - alpha_hats[time]
        )
        mean = inverse_sqrt_alpha * (latent - epsilon_scale * epsilon)
        step_rng, noise_key = jax.random.split(step_rng)
        reverse_noise = jax.random.normal(noise_key, mean.shape, dtype=mean.dtype)
        latent = mean + (time > 0) * (
            jnp.sqrt(betas[time]) * sample_temperature * reverse_noise
        )
        # Deliberately no jnp.clip here: action bounds are imposed only by tanh.
        return (latent, step_rng), None

    initial_key, rng = jax.random.split(rng)
    initial_latent = jax.random.normal(
        initial_key, (batch_size, action_dim), dtype=jnp.float32
    )
    (latent, rng), _ = jax.lax.scan(
        reverse_step,
        (initial_latent, rng),
        jnp.arange(num_steps - 1, -1, -1),
        unroll=num_steps,
    )
    for _ in range(repeat_last_step):
        (latent, rng), _ = reverse_step((latent, rng), jnp.asarray(0))
    return latent, rng


def _normal_log_density(value: Array, mean: Array, std: Array) -> Array:
    """Diagonal isotropic Normal log-density along the final dimension."""

    standardized = (value - mean) / std
    action_dim = value.shape[-1]
    return (
        -0.5 * jnp.sum(jnp.square(standardized), axis=-1)
        - action_dim * jnp.log(std[..., 0])
        - 0.5 * action_dim * LOG_TWO_PI
    )


def _source_weight_statistics(
    weights: Array, source_mask: Array
) -> tuple[Array, Array]:
    """Return posterior weight mass and conditional ESS for one source."""

    weights = jnp.asarray(weights, dtype=jnp.float32)
    source_mask = jnp.asarray(source_mask, dtype=jnp.bool_)
    if source_mask.ndim == 1:
        source_mask = source_mask[None, :]
    masked_weights = jnp.where(source_mask, weights, 0.0)
    mass = jnp.sum(masked_weights, axis=1)
    squared_mass = jnp.sum(jnp.square(masked_weights), axis=1)
    ess = jnp.where(
        mass > 0.0,
        jnp.square(mass) / jnp.maximum(squared_mass, jnp.finfo(jnp.float32).tiny),
        0.0,
    )
    return mass, ess


def _source_feasible_acceptance(
    feasible_support: Array, source_mask: Array
) -> Array:
    """Fraction of safety-feasible candidates drawn from one source."""

    feasible_support = jnp.asarray(feasible_support, dtype=jnp.bool_)
    source_mask = jnp.asarray(source_mask, dtype=jnp.bool_)
    count = jnp.sum(source_mask.astype(jnp.float32))
    accepted = jnp.sum(
        feasible_support.astype(jnp.float32) * source_mask[None, :], axis=1
    )
    return jnp.where(count > 0.0, accepted / count, 0.0)


def kl_budget_reward_tilt(
    base_log_weight: Array,
    reward_q: Array,
    support: Array,
    *,
    delta: float,
    bisection_steps: int = 32,
    max_kappa: float = 64.0,
) -> KLBudgetRewardTilt:
    r"""Tilt a finite MIS base by reward under a per-row KL budget.

    Let ``b`` be the normalized, reward-free MIS weights on the supplied safe
    support and let ``z=(Q_r-E_b Q_r)/std_b(Q_r)``.  This function returns

    ``w_kappa(i) = b(i) exp(kappa z(i)) / Z(kappa)``,

    choosing ``kappa`` by adaptive bracketing followed by fixed-count
    bisection so that
    ``KL(w_kappa || b)=delta`` whenever the target is attainable.  Centering
    and scaling make the update invariant to positive affine transformations
    of ``Q_r``.  Invalid and reward-flat rows retain ``b`` exactly.
    """

    if delta < 0.0 or not float(delta) < float("inf"):
        raise ValueError("delta must be finite and non-negative.")
    if bisection_steps != 32:
        raise ValueError("The registered KL solver uses exactly 32 bisection steps.")
    if (
        max_kappa <= 0.0
        or not float(max_kappa) < float("inf")
        or float(max_kappa) != 64.0
    ):
        raise ValueError("The registered KL solver requires finite max_kappa=64.")

    base_log_weight = jnp.asarray(base_log_weight, dtype=jnp.float32)
    reward_q = jnp.asarray(reward_q, dtype=jnp.float32)
    support = jnp.asarray(support, dtype=jnp.bool_)
    if (
        base_log_weight.ndim != 2
        or reward_q.shape != base_log_weight.shape
        or support.shape != base_log_weight.shape
    ):
        raise ValueError(
            "base_log_weight, reward_q, and support must share shape [B,M]."
        )

    finite = support & jnp.isfinite(base_log_weight) & jnp.isfinite(reward_q)
    valid = jnp.any(finite, axis=1)
    invalid_fallback_log_probability = jnp.where(
        jnp.arange(base_log_weight.shape[1]) == 0, 0.0, -jnp.inf
    )
    safe_base_log_weight = jnp.where(finite, base_log_weight, -jnp.inf)
    normalizer_input = jnp.where(
        valid[:, None],
        safe_base_log_weight,
        invalid_fallback_log_probability[None, :],
    )
    base_log_normalizer = logsumexp(
        normalizer_input, axis=1, keepdims=True
    )
    base_log_probability = normalizer_input - base_log_normalizer
    base_weights = jnp.where(
        finite, jnp.exp(base_log_probability), 0.0
    )

    safe_reward = jnp.where(finite, reward_q, 0.0)
    base_q_mean = jnp.sum(base_weights * safe_reward, axis=1)
    q_min = jnp.min(jnp.where(finite, reward_q, jnp.inf), axis=1)
    q_max = jnp.max(jnp.where(finite, reward_q, -jnp.inf), axis=1)
    q_min = jnp.where(valid, q_min, 0.0)
    q_span = jnp.where(valid, jnp.maximum(q_max - q_min, 0.0), 0.0)
    # Normalize by the row span before measuring flatness.  This makes the
    # criterion and standardized reward invariant to positive affine changes
    # over the representable float32 range.
    unit_reward = jnp.where(
        finite,
        (safe_reward - q_min[:, None])
        / jnp.maximum(q_span[:, None], jnp.finfo(jnp.float32).tiny),
        0.0,
    )
    unit_q_mean = jnp.sum(base_weights * unit_reward, axis=1)
    centered_unit_q = jnp.where(
        finite, unit_reward - unit_q_mean[:, None], 0.0
    )
    unit_q_variance = jnp.sum(
        base_weights * jnp.square(centered_unit_q), axis=1
    )
    unit_q_std = jnp.sqrt(jnp.maximum(unit_q_variance, 0.0))
    q_std = q_span * unit_q_std
    nonflat = (
        valid
        & (q_span > 0.0)
        & (unit_q_variance > jnp.asarray(1e-12, jnp.float32))
    )
    standardized_q = jnp.where(
        nonflat[:, None],
        centered_unit_q / jnp.maximum(unit_q_std[:, None], 1e-12),
        0.0,
    )

    def weights_and_kl(kappa: Array) -> tuple[Array, Array]:
        logits = base_log_probability + kappa[:, None] * standardized_q
        log_z = logsumexp(logits, axis=1)
        weights = jnp.where(
            finite,
            jnp.exp(logits - log_z[:, None]),
            0.0,
        )
        expected_z = jnp.sum(weights * standardized_q, axis=1)
        kl = jnp.where(
            valid,
            jnp.maximum(kappa * expected_z - log_z, 0.0),
            0.0,
        )
        return weights, kl

    delta_array = jnp.full(
        (base_log_weight.shape[0],), float(delta), dtype=jnp.float32
    )
    requested = delta_array > 0.0
    active = valid & nonflat & requested
    # A universal [0, max_kappa] interval has absolute resolution
    # max_kappa / 2**N.  That is insufficient when a concentrated MIS base
    # requires a very small kappa.  Bracket each row from one upward instead;
    # the subsequent [0, high] interval has at most unit width for small roots.
    high = jnp.ones_like(delta_array)

    def bracket_body(_: int, current_high: Array) -> Array:
        _, current_kl = weights_and_kl(current_high)
        needs_growth = active & (current_kl < delta_array)
        return jnp.where(
            needs_growth,
            jnp.minimum(2.0 * current_high, float(max_kappa)),
            current_high,
        )

    high = jax.lax.fori_loop(0, 6, bracket_body, high)
    _, high_kl = weights_and_kl(high)
    attainable = valid & (
        (~requested)
        | (nonflat & (high_kl + jnp.asarray(2e-5, jnp.float32) >= delta_array))
    )
    target = jnp.minimum(delta_array, high_kl)
    # Search in log(kappa), retaining an explicit kappa=0 safe endpoint.
    # This resolves roots many orders of magnitude below one without ever
    # returning a midpoint whose evaluated KL exceeds the requested budget.
    log_low = jnp.full_like(high, -80.0)
    log_high = jnp.log(high)
    safe_kappa = jnp.zeros_like(high)

    def bisection_body(
        _: int, bracket: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        current_log_low, current_log_high, current_safe_kappa = bracket
        midpoint_log = 0.5 * (current_log_low + current_log_high)
        midpoint = jnp.exp(midpoint_log)
        _, midpoint_kl = weights_and_kl(midpoint)
        midpoint_is_safe = midpoint_kl <= target
        next_log_low = jnp.where(
            active & midpoint_is_safe, midpoint_log, current_log_low
        )
        next_log_high = jnp.where(
            active & (~midpoint_is_safe), midpoint_log, current_log_high
        )
        next_safe_kappa = jnp.where(
            active & midpoint_is_safe, midpoint, current_safe_kappa
        )
        return next_log_low, next_log_high, next_safe_kappa

    _, _, safe_kappa = jax.lax.fori_loop(
        0,
        bisection_steps,
        bisection_body,
        (log_low, log_high, safe_kappa),
    )
    kappa = jnp.where(active, safe_kappa, 0.0)
    tilted_weights, achieved_kl = weights_and_kl(kappa)
    # Explicit no-op routing makes delta=0 and flat rows exactly equal to b.
    weights = jnp.where(active[:, None], tilted_weights, base_weights)
    achieved_kl = jnp.where(active, achieved_kl, 0.0)
    log_weight = jnp.where(
        weights > 0.0, jnp.log(weights), -jnp.inf
    )
    squared_weight_sum = jnp.sum(jnp.square(weights), axis=1)
    ess = jnp.where(
        valid,
        1.0 / jnp.maximum(squared_weight_sum, jnp.finfo(jnp.float32).tiny),
        0.0,
    )
    max_weight = jnp.max(weights, axis=1)
    tilted_q_mean = jnp.sum(weights * safe_reward, axis=1)
    q_gain = jnp.where(valid, tilted_q_mean - base_q_mean, 0.0)
    tolerance = jnp.maximum(
        jnp.asarray(2e-4, jnp.float32),
        jnp.asarray(2e-3, jnp.float32) * delta_array,
    )
    target_hit = valid & (
        (~requested)
        | (
            nonflat
            & attainable
            & (jnp.abs(achieved_kl - delta_array) <= tolerance)
        )
    )
    tau_proxy = jnp.where(
        active,
        q_std / jnp.maximum(kappa, jnp.finfo(jnp.float32).tiny),
        0.0,
    )
    return KLBudgetRewardTilt(
        log_weight=jax.lax.stop_gradient(log_weight),
        weights=jax.lax.stop_gradient(weights),
        valid=jax.lax.stop_gradient(valid),
        achieved_kl=jax.lax.stop_gradient(achieved_kl),
        target_hit=jax.lax.stop_gradient(target_hit),
        attainable=jax.lax.stop_gradient(attainable),
        ess_over_m=jax.lax.stop_gradient(
            ess / float(base_log_weight.shape[1])
        ),
        max_weight=jax.lax.stop_gradient(max_weight),
        q_gain=jax.lax.stop_gradient(q_gain),
        flat=jax.lax.stop_gradient(valid & (~nonflat)),
        active=jax.lax.stop_gradient(active),
        row_cap_active=jax.lax.stop_gradient(jnp.zeros_like(active)),
        effective_kappa=jax.lax.stop_gradient(kappa),
        tau_proxy=jax.lax.stop_gradient(tau_proxy),
    )


def batch_kl_reward_tilt(
    base_log_weight: Array,
    reward_q: Array,
    support: Array,
    eligible_row: Array,
    *,
    delta: float,
    row_kl_cap: float | None = None,
    bisection_steps: int = 32,
    max_kappa: float = 64.0,
) -> KLBudgetRewardTilt:
    r"""Solve one batch-relative-entropy reward tilt on raw reward advantages.

    For the finite, non-flat rows selected by ``eligible_row``, let ``b_s`` be
    the normalized reward-free MIS base and

    ``A_si = Q_r(s,a_i) - E_b[Q_r(s,a)]``.

    The uncapped mode solves

    .. math::

       \max_{\{w_s\}}\frac1{|\mathcal F|}\sum_s E_{w_s}[A_s]
       \quad\text{s.t.}\quad
       \frac1{|\mathcal F|}\sum_s KL(w_s\Vert b_s)\le\bar\delta.

    Its KKT solution has one shared dual coefficient:
    ``w_s ∝ b_s exp(kappa A_s)``.  If ``row_kl_cap`` is supplied, the same
    problem additionally imposes ``KL(w_s||b_s) <= row_kl_cap`` for every
    active row.  The corresponding row coefficient is the smaller of the
    shared coefficient and the coefficient that reaches the row cap.

    A single batch-wide RMS rescales all advantages before root finding.  This
    changes only the numerical parameterization of the shared dual, not the
    optimizer, and gives positive global affine invariance without erasing
    relative contrast between rows.  Recovery, invalid, and reward-flat rows
    are excluded from both the objective and the KL denominator.
    """

    if delta < 0.0 or not float(delta) < float("inf"):
        raise ValueError("delta must be finite and non-negative.")
    if row_kl_cap is not None:
        if row_kl_cap < 0.0 or not float(row_kl_cap) < float("inf"):
            raise ValueError("row_kl_cap must be finite and non-negative.")
        if float(row_kl_cap) < float(delta):
            raise ValueError("row_kl_cap must be at least the batch KL delta.")
    if bisection_steps != 32:
        raise ValueError(
            "The registered batch KL solver uses exactly 32 bisection steps."
        )
    if (
        max_kappa <= 0.0
        or not float(max_kappa) < float("inf")
        or float(max_kappa) != 64.0
    ):
        raise ValueError("The registered batch KL solver requires max_kappa=64.")

    base_log_weight = jnp.asarray(base_log_weight, dtype=jnp.float32)
    reward_q = jnp.asarray(reward_q, dtype=jnp.float32)
    support = jnp.asarray(support, dtype=jnp.bool_)
    eligible_row = jnp.asarray(eligible_row, dtype=jnp.bool_).reshape((-1,))
    if (
        base_log_weight.ndim != 2
        or reward_q.shape != base_log_weight.shape
        or support.shape != base_log_weight.shape
        or eligible_row.shape != (base_log_weight.shape[0],)
    ):
        raise ValueError(
            "base_log_weight, reward_q, support, and eligible_row must have "
            "shapes [B,M], [B,M], [B,M], and [B]."
        )

    finite = support & jnp.isfinite(base_log_weight) & jnp.isfinite(reward_q)
    valid = jnp.any(finite, axis=1)
    fallback = jnp.where(
        jnp.arange(base_log_weight.shape[1]) == 0, 0.0, -jnp.inf
    )
    masked_base = jnp.where(finite, base_log_weight, -jnp.inf)
    normalization_input = jnp.where(
        valid[:, None], masked_base, fallback[None, :]
    )
    base_log_probability = normalization_input - logsumexp(
        normalization_input, axis=1, keepdims=True
    )
    base_weights = jnp.where(finite, jnp.exp(base_log_probability), 0.0)

    safe_reward = jnp.where(finite, reward_q, 0.0)
    base_q_mean = jnp.sum(base_weights * safe_reward, axis=1)
    centered_q = jnp.where(
        finite, safe_reward - base_q_mean[:, None], 0.0
    )
    q_variance = jnp.sum(base_weights * jnp.square(centered_q), axis=1)
    q_min = jnp.min(jnp.where(finite, reward_q, jnp.inf), axis=1)
    q_max = jnp.max(jnp.where(finite, reward_q, -jnp.inf), axis=1)
    q_min = jnp.where(valid, q_min, 0.0)
    q_max = jnp.where(valid, q_max, 0.0)
    q_span = jnp.maximum(q_max - q_min, 0.0)
    unit_q = jnp.where(
        finite,
        (safe_reward - q_min[:, None])
        / jnp.maximum(q_span[:, None], jnp.finfo(jnp.float32).tiny),
        0.0,
    )
    unit_q_mean = jnp.sum(base_weights * unit_q, axis=1)
    unit_q_variance = jnp.sum(
        base_weights
        * jnp.square(
            jnp.where(finite, unit_q - unit_q_mean[:, None], 0.0)
        ),
        axis=1,
    )
    nonflat = (
        valid
        & (q_span > 0.0)
        & (unit_q_variance > jnp.asarray(1e-12, dtype=jnp.float32))
    )
    eligible = eligible_row & nonflat
    eligible_count = jnp.sum(eligible.astype(jnp.float32))
    global_variance = jnp.sum(
        jnp.where(eligible, q_variance, 0.0)
    ) / jnp.maximum(eligible_count, 1.0)
    global_scale = jnp.sqrt(
        jnp.maximum(global_variance, jnp.finfo(jnp.float32).tiny)
    )
    scaled_advantage = centered_q / global_scale

    def weights_and_kl(row_kappa: Array) -> tuple[Array, Array]:
        row_kappa = jnp.asarray(row_kappa, dtype=jnp.float32).reshape((-1,))
        logits = (
            base_log_probability
            + row_kappa[:, None] * scaled_advantage
        )
        log_z = logsumexp(logits, axis=1, keepdims=True)
        weights = jnp.where(
            finite, jnp.exp(logits - log_z), 0.0
        )
        # Since base_log_probability is normalized, the finite-support
        # log-ratio is kappa*A-log(Z).  Using this reduced identity is also
        # important numerically: directly subtracting the two masked ``-inf``
        # logits would create NaNs before ``where`` can discard them.
        log_ratio = jnp.where(
            finite,
            row_kappa[:, None] * scaled_advantage - log_z,
            0.0,
        )
        kl = jnp.where(
            valid,
            jnp.maximum(jnp.sum(weights * log_ratio, axis=1), 0.0),
            0.0,
        )
        return weights, kl

    requested = jnp.asarray(float(delta) > 0.0)
    active = eligible & requested
    active_count = jnp.sum(active.astype(jnp.float32))
    any_active = active_count > 0.0

    if row_kl_cap is None:
        cap_kappa = jnp.full(
            (base_log_weight.shape[0],),
            float(max_kappa),
            dtype=jnp.float32,
        )
        cap_requested = jnp.asarray(False)
    else:
        cap_requested = jnp.asarray(True)
        cap_target = jnp.full(
            (base_log_weight.shape[0],),
            float(row_kl_cap),
            dtype=jnp.float32,
        )
        cap_search_active = eligible & (cap_target > 0.0)
        cap_high = jnp.ones_like(cap_target)

        def cap_bracket_body(_: int, current_high: Array) -> Array:
            _, current_kl = weights_and_kl(current_high)
            grow = cap_search_active & (current_kl < cap_target)
            return jnp.where(
                grow,
                jnp.minimum(2.0 * current_high, float(max_kappa)),
                current_high,
            )

        cap_high = jax.lax.fori_loop(0, 6, cap_bracket_body, cap_high)
        _, cap_high_kl = weights_and_kl(cap_high)
        effective_cap_target = jnp.minimum(cap_target, cap_high_kl)
        cap_log_low = jnp.full_like(cap_high, -80.0)
        cap_log_high = jnp.log(cap_high)
        cap_safe_kappa = jnp.zeros_like(cap_high)

        def cap_bisection_body(
            _: int, bracket: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            low, high, safe = bracket
            midpoint_log = 0.5 * (low + high)
            midpoint = jnp.exp(midpoint_log)
            _, midpoint_kl = weights_and_kl(midpoint)
            midpoint_safe = midpoint_kl <= effective_cap_target
            return (
                jnp.where(
                    cap_search_active & midpoint_safe, midpoint_log, low
                ),
                jnp.where(
                    cap_search_active & (~midpoint_safe), midpoint_log, high
                ),
                jnp.where(
                    cap_search_active & midpoint_safe, midpoint, safe
                ),
            )

        _, _, cap_safe_kappa = jax.lax.fori_loop(
            0,
            bisection_steps,
            cap_bisection_body,
            (cap_log_low, cap_log_high, cap_safe_kappa),
        )
        cap_kappa = jnp.where(
            eligible,
            jnp.where(cap_target > 0.0, cap_safe_kappa, 0.0),
            float(max_kappa),
        )

    def batch_weights_and_kl(
        shared_kappa: Array,
    ) -> tuple[Array, Array, Array]:
        shared_vector = jnp.full(
            (base_log_weight.shape[0],),
            shared_kappa,
            dtype=jnp.float32,
        )
        row_kappa = jnp.minimum(shared_vector, cap_kappa)
        weights, row_kl = weights_and_kl(row_kappa)
        mean_kl = jnp.sum(jnp.where(active, row_kl, 0.0)) / jnp.maximum(
            active_count, 1.0
        )
        return weights, row_kl, mean_kl

    high = jnp.asarray(1.0, dtype=jnp.float32)

    def batch_bracket_body(_: int, current_high: Array) -> Array:
        _, _, current_mean_kl = batch_weights_and_kl(current_high)
        grow = any_active & (current_mean_kl < float(delta))
        return jnp.where(
            grow,
            jnp.minimum(2.0 * current_high, float(max_kappa)),
            current_high,
        )

    high = jax.lax.fori_loop(0, 6, batch_bracket_body, high)
    _, _, high_mean_kl = batch_weights_and_kl(high)
    target = jnp.minimum(
        jnp.asarray(float(delta), dtype=jnp.float32), high_mean_kl
    )
    log_low = jnp.asarray(-80.0, dtype=jnp.float32)
    log_high = jnp.log(high)
    safe_kappa = jnp.asarray(0.0, dtype=jnp.float32)

    def batch_bisection_body(
        _: int, bracket: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        low, current_high, safe = bracket
        midpoint_log = 0.5 * (low + current_high)
        midpoint = jnp.exp(midpoint_log)
        _, _, midpoint_mean_kl = batch_weights_and_kl(midpoint)
        midpoint_safe = midpoint_mean_kl <= target
        search_active = any_active & requested
        return (
            jnp.where(search_active & midpoint_safe, midpoint_log, low),
            jnp.where(
                search_active & (~midpoint_safe),
                midpoint_log,
                current_high,
            ),
            jnp.where(search_active & midpoint_safe, midpoint, safe),
        )

    _, _, safe_kappa = jax.lax.fori_loop(
        0,
        bisection_steps,
        batch_bisection_body,
        (log_low, log_high, safe_kappa),
    )
    shared_kappa = jnp.where(any_active & requested, safe_kappa, 0.0)
    shared_vector = jnp.full(
        (base_log_weight.shape[0],),
        shared_kappa,
        dtype=jnp.float32,
    )
    row_kappa = jnp.minimum(shared_vector, cap_kappa)
    tilted_weights, row_kl = weights_and_kl(row_kappa)
    weights = jnp.where(active[:, None], tilted_weights, base_weights)
    achieved_kl = jnp.where(active, row_kl, 0.0)
    achieved_mean_kl = jnp.sum(achieved_kl) / jnp.maximum(active_count, 1.0)
    tolerance = jnp.maximum(
        jnp.asarray(2e-4, dtype=jnp.float32),
        jnp.asarray(2e-3 * float(delta), dtype=jnp.float32),
    )
    globally_attainable = (~requested) | (
        any_active & (high_mean_kl + tolerance >= float(delta))
    )
    global_target_hit = (~requested) | (
        globally_attainable
        & (jnp.abs(achieved_mean_kl - float(delta)) <= tolerance)
    )
    row_cap_active = (
        active
        & cap_requested
        & (row_kappa + jnp.asarray(2e-5, jnp.float32) < shared_kappa)
    )

    squared_weight_sum = jnp.sum(jnp.square(weights), axis=1)
    ess = jnp.where(
        valid,
        1.0
        / jnp.maximum(
            squared_weight_sum, jnp.finfo(jnp.float32).tiny
        ),
        0.0,
    )
    max_weight = jnp.max(weights, axis=1)
    tilted_q_mean = jnp.sum(weights * safe_reward, axis=1)
    q_gain = jnp.where(valid, tilted_q_mean - base_q_mean, 0.0)
    raw_kappa = jnp.where(
        active, row_kappa / global_scale, 0.0
    )
    tau_proxy = jnp.where(
        raw_kappa > 0.0,
        1.0 / jnp.maximum(raw_kappa, jnp.finfo(jnp.float32).tiny),
        0.0,
    )
    log_weight = jnp.where(weights > 0.0, jnp.log(weights), -jnp.inf)
    return KLBudgetRewardTilt(
        log_weight=jax.lax.stop_gradient(log_weight),
        weights=jax.lax.stop_gradient(weights),
        valid=jax.lax.stop_gradient(valid),
        achieved_kl=jax.lax.stop_gradient(achieved_kl),
        target_hit=jax.lax.stop_gradient(active & global_target_hit),
        attainable=jax.lax.stop_gradient(active & globally_attainable),
        ess_over_m=jax.lax.stop_gradient(
            ess / float(base_log_weight.shape[1])
        ),
        max_weight=jax.lax.stop_gradient(max_weight),
        q_gain=jax.lax.stop_gradient(q_gain),
        flat=jax.lax.stop_gradient(valid & (~nonflat)),
        active=jax.lax.stop_gradient(active),
        row_cap_active=jax.lax.stop_gradient(row_cap_active),
        effective_kappa=jax.lax.stop_gradient(raw_kappa),
        tau_proxy=jax.lax.stop_gradient(tau_proxy),
    )


def apply_reward_tilt_to_legacy_weights(
    legacy_log_weight: Array,
    reward_free_base_log_weight: Array,
    reward_q: Array,
    feasible_state: Array,
    feasible_support: Array,
    candidate_valid: Array,
    *,
    mode: str,
    kl_delta: float,
    row_kl_cap: float = 1.0,
) -> tuple[Array, KLBudgetRewardTilt]:
    """Replace only feasible-row reward weights; recovery remains untouched."""

    legacy_log_weight = jnp.asarray(legacy_log_weight, dtype=jnp.float32)
    feasible_state = jnp.asarray(feasible_state, dtype=jnp.bool_).reshape((-1,))
    if mode == LP_REWARD_TILT_FIXED_ALPHA:
        batch_size = legacy_log_weight.shape[0]
        zero = jnp.zeros((batch_size,), dtype=jnp.float32)
        false = jnp.zeros((batch_size,), dtype=jnp.bool_)
        diagnostics = KLBudgetRewardTilt(
            log_weight=legacy_log_weight,
            weights=jnp.zeros_like(legacy_log_weight),
            valid=false,
            achieved_kl=zero,
            target_hit=false,
            attainable=false,
            ess_over_m=zero,
            max_weight=zero,
            q_gain=zero,
            flat=false,
            active=false,
            row_cap_active=false,
            effective_kappa=zero,
            tau_proxy=zero,
        )
        return legacy_log_weight, diagnostics
    if mode in (
        LP_REWARD_TILT_BATCH_KL,
        LP_REWARD_TILT_CAPPED_BATCH_KL,
    ):
        tilt = batch_kl_reward_tilt(
            reward_free_base_log_weight,
            reward_q,
            jnp.asarray(feasible_support, dtype=jnp.bool_)
            & jnp.asarray(candidate_valid, dtype=jnp.bool_),
            feasible_state,
            delta=kl_delta,
            row_kl_cap=(
                row_kl_cap
                if mode == LP_REWARD_TILT_CAPPED_BATCH_KL
                else None
            ),
        )
    elif mode == LP_REWARD_TILT_KL_BUDGET:
        tilt = kl_budget_reward_tilt(
            reward_free_base_log_weight,
            reward_q,
            jnp.asarray(feasible_support, dtype=jnp.bool_)
            & jnp.asarray(candidate_valid, dtype=jnp.bool_),
            delta=kl_delta,
        )
    else:
        raise ValueError(f"Unsupported reward tilt mode {mode!r}.")
    combined = jnp.where(
        feasible_state[:, None], tilt.log_weight, legacy_log_weight
    )
    feasible_float = feasible_state.astype(jnp.float32)
    feasible_bool = feasible_state
    return combined, tilt._replace(
        log_weight=combined,
        weights=tilt.weights * feasible_float[:, None],
        valid=tilt.valid & feasible_bool,
        achieved_kl=tilt.achieved_kl * feasible_float,
        target_hit=tilt.target_hit & feasible_bool,
        attainable=tilt.attainable & feasible_bool,
        ess_over_m=tilt.ess_over_m * feasible_float,
        max_weight=tilt.max_weight * feasible_float,
        q_gain=tilt.q_gain * feasible_float,
        flat=tilt.flat & feasible_bool,
        active=tilt.active & feasible_bool,
        row_cap_active=tilt.row_cap_active & feasible_bool,
        effective_kappa=tilt.effective_kappa * feasible_float,
        tau_proxy=tilt.tau_proxy * feasible_float,
    )


def _evaluate_target_critics_chunk(
    core: SSMOnlineAgent,
    observations: Array,
    bounded_actions: Array,
    *,
    actor_safety_mode: str = LP_ACTOR_SAFETY_QH_ROUTED,
) -> tuple[Array, Array]:
    """Evaluate target critics using the actor's registered safety semantics."""

    batch_size, num_samples, action_dim = bounded_actions.shape
    flat_observations = jnp.repeat(observations, num_samples, axis=0)
    flat_actions = bounded_actions.reshape((batch_size * num_samples, action_dim))
    reward_1 = core.target_critic_1.apply_fn(
        {"params": core.target_critic_1.params}, flat_observations, flat_actions
    )
    reward_2 = core.target_critic_2.apply_fn(
        {"params": core.target_critic_2.params}, flat_observations, flat_actions
    )
    reward = jnp.minimum(reward_1, reward_2)
    if actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY:
        # Q_h can still be trained by the critic path; the reward-only actor
        # does not read its parameters.
        safety = jnp.zeros_like(reward)
    elif actor_safety_mode == LP_ACTOR_SAFETY_QH_ROUTED:
        if hasattr(core, "qh_critic_mode"):
            safety, _, _ = SSMOnlineAgent._evaluate_safe_critics(
                core,
                flat_observations,
                flat_actions,
                use_target=True,
            )
        else:
            # Lightweight analytic unit-test cores predate explicit reducer
            # metadata and model the historical twin-max endpoint.
            safety_1 = core.safe_target_critic.apply_fn(
                {"params": core.safe_target_critic.params},
                flat_observations,
                flat_actions,
            )
            safety_2 = core.safe_target_critic_2.apply_fn(
                {"params": core.safe_target_critic_2.params},
                flat_observations,
                flat_actions,
            )
            safety = jnp.maximum(safety_1, safety_2)
    else:
        raise ValueError(f"Unsupported actor_safety_mode={actor_safety_mode!r}.")
    return (
        reward.reshape((batch_size, num_samples)),
        safety.reshape((batch_size, num_samples)),
    )


def _evaluate_target_critics_chunked(
    core: SSMOnlineAgent,
    observations: Array,
    bounded_actions: Array,
    *,
    proposal_chunk_size: int,
    actor_safety_mode: str = LP_ACTOR_SAFETY_QH_ROUTED,
) -> tuple[Array, Array]:
    """Chunk only the proposal axis while preserving exact SNIS normalization."""

    if proposal_chunk_size <= 0:
        raise ValueError(
            f"proposal_chunk_size must be positive, got {proposal_chunk_size}."
        )
    num_samples = bounded_actions.shape[1]
    reward_chunks = []
    safety_chunks = []
    for start in range(0, num_samples, proposal_chunk_size):
        stop = min(start + proposal_chunk_size, num_samples)
        reward, safety = _evaluate_target_critics_chunk(
            core,
            observations,
            bounded_actions[:, start:stop, :],
            actor_safety_mode=actor_safety_mode,
        )
        reward_chunks.append(reward)
        safety_chunks.append(safety)
    return jnp.concatenate(reward_chunks, axis=1), jnp.concatenate(
        safety_chunks, axis=1
    )


def latent_balance_mis_posterior_target(
    core: SSMOnlineAgent,
    observations: Array,
    noisy_latent: Array,
    alpha: Array,
    sigma: Array,
    feasible_state: Array,
    rng: Array,
    *,
    num_samples: int = 64,
    likelihood_fraction: float = 0.5,
    proposal_chunk_size: int = 32,
    alpha_reward: float = 1.0,
    beta_safety: float = 3.0,
    reward_tilt_mode: str = LP_REWARD_TILT_FIXED_ALPHA,
    reward_kl_delta: float = 0.25,
    reward_row_kl_cap: float = 1.0,
    safety_threshold: float = 0.0,
    boundary_epsilon: float = 1e-5,
    actor_safety_mode: str = LP_ACTOR_SAFETY_QH_ROUTED,
) -> LPPosteriorTarget:
    r"""Estimate ``E[epsilon | u_t,s]`` with balance-heuristic MIS.

    The clean target density is specified relative to action-space Lebesgue
    measure.  Consequently its latent density contains
    ``log|det J_tanh|``.  The full unnormalized importance log weight is

    ``log p(u_t|u_0) + log target_u(u_0|s) - log q_mix(u_0|u_t)``.

    With ``likelihood_fraction=1`` the likelihood/proposal terms differ only
    by a row constant, recovering the likelihood-only ``energy + log_jac``
    weights. Smaller likelihood fractions mix in uniform-action proposals.
    """

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if not 0.0 <= likelihood_fraction <= 1.0:
        raise ValueError("likelihood_fraction must lie in [0,1].")
    if proposal_chunk_size <= 0:
        raise ValueError("proposal_chunk_size must be positive.")
    if not 0.0 < boundary_epsilon < 1.0:
        raise ValueError("boundary_epsilon must lie in (0,1).")
    observations = jnp.asarray(observations, dtype=jnp.float32)
    noisy_latent = jnp.asarray(noisy_latent, dtype=jnp.float32)
    if observations.ndim != 2 or noisy_latent.ndim != 2:
        raise ValueError("observations and noisy_latent must have shapes [B,D].")
    batch_size, action_dim = noisy_latent.shape
    if observations.shape[0] != batch_size:
        raise ValueError("observations and noisy_latent batch sizes differ.")

    likelihood_count = int(round(num_samples * likelihood_fraction))
    base_count = num_samples - likelihood_count
    actual_fraction = likelihood_count / num_samples
    alpha = jnp.asarray(alpha, dtype=jnp.float32).reshape((batch_size, 1))
    sigma = jnp.asarray(sigma, dtype=jnp.float32).reshape((batch_size, 1))
    mean = noisy_latent / alpha
    std = sigma / alpha
    likelihood_key, base_key = jax.random.split(rng)

    proposal_parts = []
    if likelihood_count:
        likelihood_noise = jax.random.normal(
            likelihood_key,
            (batch_size, likelihood_count, action_dim),
            dtype=jnp.float32,
        )
        proposal_parts.append(
            mean[:, None, :] + std[:, None, :] * likelihood_noise
        )
    if base_count:
        uniform_actions = 2.0 * open_unit_uniform(
            base_key, (batch_size, base_count, action_dim)
        ) - 1.0
        proposal_parts.append(jnp.arctanh(uniform_actions))
    proposal_latents = (
        proposal_parts[0]
        if len(proposal_parts) == 1
        else jnp.concatenate(proposal_parts, axis=1)
    )
    proposal_actions = bounded_latent_to_action(
        proposal_latents, boundary_epsilon
    )

    reward_q, safety_q = _evaluate_target_critics_chunked(
        core,
        observations,
        proposal_actions,
        proposal_chunk_size=proposal_chunk_size,
        actor_safety_mode=actor_safety_mode,
    )
    # Critics and every quantity derived from them are teacher targets only.
    reward_q = jax.lax.stop_gradient(reward_q)
    safety_q = jax.lax.stop_gradient(safety_q)

    alpha_3d = alpha[:, None, :]
    sigma_3d = sigma[:, None, :]
    proposal_noise = (
        noisy_latent[:, None, :] - alpha_3d * proposal_latents
    ) / sigma_3d
    log_likelihood = _normal_log_density(
        noisy_latent[:, None, :], alpha_3d * proposal_latents, sigma_3d
    )
    log_likelihood_proposal = _normal_log_density(
        proposal_latents, mean[:, None, :], std[:, None, :]
    )
    # The target is defined in action Lebesgue measure and a=tanh(u), so the
    # exact latent density contains |J_tanh|.  The uniform-action proposal has
    # latent density |J_tanh|/2^D.
    log_tanh_jacobian = stable_log_tanh_jacobian(proposal_latents)
    log_jacobian = stable_log_bounded_tanh_jacobian(
        proposal_latents, boundary_epsilon
    )
    log_uniform_action_proposal = log_tanh_jacobian - action_dim * jnp.log(2.0)

    if actual_fraction == 1.0:
        log_proposal = log_likelihood_proposal
    elif actual_fraction == 0.0:
        log_proposal = log_uniform_action_proposal
    else:
        log_proposal = jnp.logaddexp(
            jnp.log(actual_fraction) + log_likelihood_proposal,
            jnp.log1p(-actual_fraction) + log_uniform_action_proposal,
        )

    feasible_state = jnp.asarray(feasible_state, dtype=jnp.bool_)
    if feasible_state.shape == (batch_size, 1):
        feasible_state = feasible_state[:, 0]
    if feasible_state.shape != (batch_size,):
        raise ValueError(
            f"feasible_state must have shape [B], got {feasible_state.shape!r}."
        )
    feasible_row = feasible_state[:, None]
    if actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY:
        feasible_support = jnp.isfinite(reward_q)
        selected_energy = jnp.where(
            feasible_support,
            jnp.asarray(alpha_reward, jnp.float32) * reward_q,
            -jnp.inf,
        )
        selected_finite = jnp.isfinite(reward_q)
    elif actor_safety_mode == LP_ACTOR_SAFETY_QH_ROUTED:
        critic_finite = jnp.isfinite(reward_q) & jnp.isfinite(safety_q)
        feasible_support = critic_finite & (
            safety_q <= jnp.asarray(safety_threshold, jnp.float32)
        )
        feasible_energy = jnp.where(
            feasible_support,
            jnp.asarray(alpha_reward, jnp.float32) * reward_q,
            -jnp.inf,
        )
        recovery_energy = jnp.where(
            jnp.isfinite(safety_q),
            -jnp.asarray(beta_safety, jnp.float32) * safety_q,
            -jnp.inf,
        )
        selected_energy = jnp.where(
            feasible_row, feasible_energy, recovery_energy
        )
        selected_finite = jnp.where(
            feasible_row,
            jnp.isfinite(reward_q) & jnp.isfinite(safety_q),
            jnp.isfinite(safety_q),
        )
    else:
        raise ValueError(f"Unsupported actor_safety_mode={actor_safety_mode!r}.")
    if actual_fraction == 1.0:
        # With the likelihood proposal
        #   q(u_0 | u_t) = N(u_t / alpha_t, (sigma_t / alpha_t)^2 I),
        # Bayes' rule leaves only the clean Gibbs energy and tanh Jacobian in
        # the self-normalized weights.  Computing this reduced expression is
        # not merely an optimization: it avoids subtracting two large,
        # algebraically identical Gaussian log densities in float32.
        log_weight = selected_energy + log_jacobian
        reward_free_base_log_weight = log_jacobian
    else:
        log_weight = (
            log_likelihood + selected_energy + log_jacobian - log_proposal
        )
        reward_free_base_log_weight = (
            log_likelihood + log_jacobian - log_proposal
        )
    log_weight, reward_tilt = apply_reward_tilt_to_legacy_weights(
        log_weight,
        reward_free_base_log_weight,
        reward_q,
        feasible_state,
        feasible_support,
        jnp.all(jnp.isfinite(proposal_noise), axis=-1),
        mode=reward_tilt_mode,
        kl_delta=reward_kl_delta,
        row_kl_cap=reward_row_kl_cap,
    )
    if actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY:
        candidate_acceptance = jnp.mean(
            feasible_support.astype(jnp.float32), axis=1
        )
    else:
        candidate_acceptance = jnp.where(
            feasible_state,
            jnp.mean(feasible_support.astype(jnp.float32), axis=1),
            1.0,
        )
    normalized = normalize_log_weighted_noise(
        jax.lax.stop_gradient(log_weight),
        jax.lax.stop_gradient(proposal_noise),
        candidate_acceptance=candidate_acceptance,
    )
    likelihood_source_mask = jnp.arange(num_samples) < likelihood_count
    likelihood_weight_mass, likelihood_component_ess = (
        _source_weight_statistics(
            normalized.weights,
            likelihood_source_mask,
        )
    )
    likelihood_feasible_acceptance = _source_feasible_acceptance(
        feasible_support, likelihood_source_mask
    )
    zero_row = jnp.zeros((batch_size,), dtype=jnp.float32)
    return LPPosteriorTarget(
        epsilon=jax.lax.stop_gradient(normalized.epsilon),
        weights=jax.lax.stop_gradient(normalized.weights),
        valid=jax.lax.stop_gradient(normalized.valid),
        ess=jax.lax.stop_gradient(normalized.ess),
        max_weight=jax.lax.stop_gradient(normalized.max_weight),
        feasible_acceptance=jax.lax.stop_gradient(
            normalized.candidate_acceptance
        ),
        critic_nonfinite_fraction=jnp.mean(
            (~selected_finite).astype(jnp.float32), axis=1
        ),
        likelihood_weight_mass=jax.lax.stop_gradient(likelihood_weight_mass),
        likelihood_fraction_actual=jnp.asarray(actual_fraction, jnp.float32),
        likelihood_component_ess=jax.lax.stop_gradient(
            likelihood_component_ess
        ),
        likelihood_feasible_acceptance=jax.lax.stop_gradient(
            likelihood_feasible_acceptance
        ),
        reference_bridge_weight_mass=zero_row,
        reference_bridge_fraction_actual=jnp.asarray(0.0, jnp.float32),
        reference_bridge_component_ess=zero_row,
        reference_bridge_feasible_acceptance=zero_row,
        reference_center_latent_norm_mean=zero_row,
        likelihood_sample_count=jnp.asarray(
            likelihood_count, dtype=jnp.float32
        ),
        reference_bridge_sample_count=jnp.asarray(0.0, dtype=jnp.float32),
        proposal_latents=jax.lax.stop_gradient(proposal_latents),
        proposal_actions=jax.lax.stop_gradient(proposal_actions),
        proposal_noise=jax.lax.stop_gradient(proposal_noise),
        reward_q_min=reward_q,
        safety_q_max=safety_q,
        reward_tilt_achieved_kl=reward_tilt.achieved_kl,
        reward_tilt_target_hit=reward_tilt.target_hit,
        reward_tilt_attainable=reward_tilt.attainable,
        reward_tilt_ess_over_m=reward_tilt.ess_over_m,
        reward_tilt_max_weight=reward_tilt.max_weight,
        reward_tilt_q_gain=reward_tilt.q_gain,
        reward_tilt_flat=reward_tilt.flat,
        reward_tilt_active=reward_tilt.active,
        reward_tilt_row_cap_active=reward_tilt.row_cap_active,
        reward_tilt_effective_kappa=reward_tilt.effective_kappa,
        reward_tilt_tau_proxy=reward_tilt.tau_proxy,
    )


def latent_defensive_bridge_mis_posterior_target(
    core: SSMOnlineAgent,
    observations: Array,
    noisy_latent: Array,
    alpha: Array,
    sigma: Array,
    feasible_state: Array,
    rng: Array,
    reference_centers: Array,
    *,
    num_samples: int = 64,
    bridge_likelihood_fraction: float = 0.5,
    bridge_latent_std: float,
    bridge_component_weighting: str = BRIDGE_COMPONENT_WEIGHTING_UNIFORM,
    proposal_chunk_size: int = 32,
    alpha_reward: float = 1.0,
    beta_safety: float = 3.0,
    reward_tilt_mode: str = LP_REWARD_TILT_FIXED_ALPHA,
    reward_kl_delta: float = 0.25,
    reward_row_kl_cap: float = 1.0,
    safety_threshold: float = 0.0,
    boundary_epsilon: float = 1e-5,
    actor_safety_mode: str = LP_ACTOR_SAFETY_QH_ROUTED,
) -> LPPosteriorTarget:
    r"""Estimate the same posterior-noise mean with defensive bridge MIS.

    The target and branch energies are identical to
    :func:`latent_balance_mis_posterior_target`.  Only the proposal changes:

    ``r_mix = omega * r_L + (1-omega) * r_bridge``.

    All candidates use the complete mixture denominator.  In particular, the
    reference actor is not multiplied into the clean target.  At
    ``bridge_likelihood_fraction=1`` this function statically dispatches to the
    historical reduced likelihood-only path with the unchanged key, providing
    an exact endpoint regression.
    """

    if bridge_likelihood_fraction == 1.0:
        return latent_balance_mis_posterior_target(
            core,
            observations,
            noisy_latent,
            alpha,
            sigma,
            feasible_state,
            rng,
            num_samples=num_samples,
            likelihood_fraction=1.0,
            proposal_chunk_size=proposal_chunk_size,
            alpha_reward=alpha_reward,
            beta_safety=beta_safety,
            reward_tilt_mode=reward_tilt_mode,
            reward_kl_delta=reward_kl_delta,
            reward_row_kl_cap=reward_row_kl_cap,
            safety_threshold=safety_threshold,
            boundary_epsilon=boundary_epsilon,
            actor_safety_mode=actor_safety_mode,
        )
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if not 0.0 <= bridge_likelihood_fraction <= 1.0:
        raise ValueError("bridge_likelihood_fraction must lie in [0,1].")
    if bridge_latent_std <= 0.0:
        raise ValueError("bridge_latent_std must be positive.")
    if bridge_component_weighting not in BRIDGE_COMPONENT_WEIGHTINGS:
        raise ValueError(
            "bridge_component_weighting must be one of "
            f"{BRIDGE_COMPONENT_WEIGHTINGS}."
        )
    if proposal_chunk_size <= 0:
        raise ValueError("proposal_chunk_size must be positive.")
    if not 0.0 < boundary_epsilon < 1.0:
        raise ValueError("boundary_epsilon must lie in (0,1).")

    observations = jnp.asarray(observations, dtype=jnp.float32)
    noisy_latent = jnp.asarray(noisy_latent, dtype=jnp.float32)
    reference_centers = jnp.asarray(reference_centers, dtype=jnp.float32)
    if observations.ndim != 2 or noisy_latent.ndim != 2:
        raise ValueError("observations and noisy_latent must have shapes [B,D].")
    batch_size, _ = noisy_latent.shape
    if observations.shape[0] != batch_size:
        raise ValueError("observations and noisy_latent batch sizes differ.")
    if (
        reference_centers.ndim != 3
        or reference_centers.shape[0] != batch_size
        or reference_centers.shape[2] != noisy_latent.shape[1]
        or reference_centers.shape[1] <= 0
    ):
        raise ValueError(
            "reference_centers must have shape [B,J,A] with J positive."
        )

    alpha = jnp.asarray(alpha, dtype=jnp.float32).reshape((batch_size, 1))
    sigma = jnp.asarray(sigma, dtype=jnp.float32).reshape((batch_size, 1))
    proposal = sample_defensive_bridge_proposal(
        rng,
        noisy_latent,
        alpha,
        sigma,
        jax.lax.stop_gradient(reference_centers),
        num_samples=num_samples,
        likelihood_fraction=bridge_likelihood_fraction,
        bridge_latent_std=bridge_latent_std,
        component_weighting=bridge_component_weighting,
    )
    proposal_latents = proposal.latents
    proposal_actions = bounded_latent_to_action(
        proposal_latents, boundary_epsilon
    )
    reward_q, safety_q = _evaluate_target_critics_chunked(
        core,
        observations,
        proposal_actions,
        proposal_chunk_size=proposal_chunk_size,
        actor_safety_mode=actor_safety_mode,
    )
    reward_q = jax.lax.stop_gradient(reward_q)
    safety_q = jax.lax.stop_gradient(safety_q)

    alpha_3d = alpha[:, None, :]
    sigma_3d = sigma[:, None, :]
    proposal_noise = (
        noisy_latent[:, None, :] - alpha_3d * proposal_latents
    ) / sigma_3d
    log_likelihood = _normal_log_density(
        noisy_latent[:, None, :], alpha_3d * proposal_latents, sigma_3d
    )
    log_jacobian = stable_log_bounded_tanh_jacobian(
        proposal_latents, boundary_epsilon
    )

    feasible_state = jnp.asarray(feasible_state, dtype=jnp.bool_)
    if feasible_state.shape == (batch_size, 1):
        feasible_state = feasible_state[:, 0]
    if feasible_state.shape != (batch_size,):
        raise ValueError(
            f"feasible_state must have shape [B], got {feasible_state.shape!r}."
        )
    feasible_row = feasible_state[:, None]
    if actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY:
        feasible_support = jnp.isfinite(reward_q)
        selected_energy = jnp.where(
            feasible_support,
            jnp.asarray(alpha_reward, jnp.float32) * reward_q,
            -jnp.inf,
        )
        selected_finite = jnp.isfinite(reward_q)
    elif actor_safety_mode == LP_ACTOR_SAFETY_QH_ROUTED:
        critic_finite = jnp.isfinite(reward_q) & jnp.isfinite(safety_q)
        feasible_support = critic_finite & (
            safety_q <= jnp.asarray(safety_threshold, jnp.float32)
        )
        feasible_energy = jnp.where(
            feasible_support,
            jnp.asarray(alpha_reward, jnp.float32) * reward_q,
            -jnp.inf,
        )
        recovery_energy = jnp.where(
            jnp.isfinite(safety_q),
            -jnp.asarray(beta_safety, jnp.float32) * safety_q,
            -jnp.inf,
        )
        selected_energy = jnp.where(
            feasible_row, feasible_energy, recovery_energy
        )
        selected_finite = jnp.where(
            feasible_row,
            jnp.isfinite(reward_q) & jnp.isfinite(safety_q),
            jnp.isfinite(safety_q),
        )
    else:
        raise ValueError(f"Unsupported actor_safety_mode={actor_safety_mode!r}.")
    log_weight = (
        log_likelihood
        + selected_energy
        + log_jacobian
        - proposal.log_density
    )
    reward_free_base_log_weight = (
        log_likelihood + log_jacobian - proposal.log_density
    )
    log_weight, reward_tilt = apply_reward_tilt_to_legacy_weights(
        log_weight,
        reward_free_base_log_weight,
        reward_q,
        feasible_state,
        feasible_support,
        jnp.all(jnp.isfinite(proposal_noise), axis=-1),
        mode=reward_tilt_mode,
        kl_delta=reward_kl_delta,
        row_kl_cap=reward_row_kl_cap,
    )
    if actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY:
        candidate_acceptance = jnp.mean(
            feasible_support.astype(jnp.float32), axis=1
        )
    else:
        candidate_acceptance = jnp.where(
            feasible_state,
            jnp.mean(feasible_support.astype(jnp.float32), axis=1),
            1.0,
        )
    normalized = normalize_log_weighted_noise(
        jax.lax.stop_gradient(log_weight),
        jax.lax.stop_gradient(proposal_noise),
        candidate_acceptance=candidate_acceptance,
    )
    likelihood_mask = proposal.source_is_likelihood
    bridge_mask = ~likelihood_mask
    likelihood_weight_mass, likelihood_component_ess = (
        _source_weight_statistics(normalized.weights, likelihood_mask)
    )
    bridge_weight_mass, bridge_component_ess = _source_weight_statistics(
        normalized.weights, bridge_mask
    )
    likelihood_feasible_acceptance = _source_feasible_acceptance(
        feasible_support, likelihood_mask
    )
    bridge_feasible_acceptance = _source_feasible_acceptance(
        feasible_support, bridge_mask
    )
    reference_center_norm = jnp.mean(
        jnp.linalg.norm(reference_centers, axis=-1), axis=1
    )
    likelihood_count = int(
        round(num_samples * bridge_likelihood_fraction)
    )
    bridge_count = num_samples - likelihood_count
    return LPPosteriorTarget(
        epsilon=jax.lax.stop_gradient(normalized.epsilon),
        weights=jax.lax.stop_gradient(normalized.weights),
        valid=jax.lax.stop_gradient(normalized.valid),
        ess=jax.lax.stop_gradient(normalized.ess),
        max_weight=jax.lax.stop_gradient(normalized.max_weight),
        feasible_acceptance=jax.lax.stop_gradient(
            normalized.candidate_acceptance
        ),
        critic_nonfinite_fraction=jnp.mean(
            (~selected_finite).astype(jnp.float32), axis=1
        ),
        likelihood_weight_mass=jax.lax.stop_gradient(
            likelihood_weight_mass
        ),
        likelihood_fraction_actual=proposal.likelihood_fraction_actual,
        likelihood_component_ess=jax.lax.stop_gradient(
            likelihood_component_ess
        ),
        likelihood_feasible_acceptance=jax.lax.stop_gradient(
            likelihood_feasible_acceptance
        ),
        reference_bridge_weight_mass=jax.lax.stop_gradient(
            bridge_weight_mass
        ),
        reference_bridge_fraction_actual=jnp.asarray(
            bridge_count / num_samples, dtype=jnp.float32
        ),
        reference_bridge_component_ess=jax.lax.stop_gradient(
            bridge_component_ess
        ),
        reference_bridge_feasible_acceptance=jax.lax.stop_gradient(
            bridge_feasible_acceptance
        ),
        reference_center_latent_norm_mean=jax.lax.stop_gradient(
            reference_center_norm
        ),
        likelihood_sample_count=jnp.asarray(
            likelihood_count, dtype=jnp.float32
        ),
        reference_bridge_sample_count=jnp.asarray(
            bridge_count, dtype=jnp.float32
        ),
        proposal_latents=jax.lax.stop_gradient(proposal_latents),
        proposal_actions=jax.lax.stop_gradient(proposal_actions),
        proposal_noise=jax.lax.stop_gradient(proposal_noise),
        reward_q_min=reward_q,
        safety_q_max=safety_q,
        reward_tilt_achieved_kl=reward_tilt.achieved_kl,
        reward_tilt_target_hit=reward_tilt.target_hit,
        reward_tilt_attainable=reward_tilt.attainable,
        reward_tilt_ess_over_m=reward_tilt.ess_over_m,
        reward_tilt_max_weight=reward_tilt.max_weight,
        reward_tilt_q_gain=reward_tilt.q_gain,
        reward_tilt_flat=reward_tilt.flat,
        reward_tilt_active=reward_tilt.active,
        reward_tilt_row_cap_active=reward_tilt.row_cap_active,
        reward_tilt_effective_kappa=reward_tilt.effective_kappa,
        reward_tilt_tau_proxy=reward_tilt.tau_proxy,
    )


def _evaluate_online_safety_candidates(
    core: SSMOnlineAgent,
    observations: Array,
    bounded_actions: Array,
) -> Array:
    """Evaluate the registered online Qh reducer for Stage-B routing.

    In ``twin_head0`` mode the second head remains diagnostic-only.  Calling
    the core reducer here keeps the LP actor's state gate consistent with the
    Bellman bootstrap, candidate minimum, posterior mask, and recovery energy.
    """

    batch_size, candidate_count, action_dim = bounded_actions.shape
    flat_observations = jnp.repeat(observations, candidate_count, axis=0)
    flat_actions = bounded_actions.reshape(
        (batch_size * candidate_count, action_dim)
    )
    qh, _, _ = SSMOnlineAgent._evaluate_safe_critics(
        core,
        flat_observations,
        flat_actions,
        use_target=False,
    )
    return qh.reshape((batch_size, candidate_count))


def _evaluate_particle_safety_candidates(
    core: SSMOnlineAgent,
    observations: Array,
    bounded_actions: Array,
    *,
    use_target: bool,
) -> Array:
    """Evaluate the configured single/head0/max Q_h reducer for particles."""

    batch_size, candidate_count, action_dim = bounded_actions.shape
    flat_observations = jnp.repeat(observations, candidate_count, axis=0)
    flat_actions = bounded_actions.reshape(
        (batch_size * candidate_count, action_dim)
    )
    qh, _, _ = SSMOnlineAgent._evaluate_safe_critics(
        core,
        flat_observations,
        flat_actions,
        use_target=use_target,
    )
    return qh.reshape((batch_size, candidate_count))


def _evaluate_particle_reward_candidates(
    core: SSMOnlineAgent,
    observations: Array,
    bounded_actions: Array,
    *,
    use_target: bool,
) -> Array:
    """Evaluate conservative twin-min reward Q for particle candidates."""

    batch_size, candidate_count, action_dim = bounded_actions.shape
    flat_observations = jnp.repeat(observations, candidate_count, axis=0)
    flat_actions = bounded_actions.reshape(
        (batch_size * candidate_count, action_dim)
    )
    critic_1 = core.target_critic_1 if use_target else core.critic_1
    critic_2 = core.target_critic_2 if use_target else core.critic_2
    q_1 = critic_1.apply_fn(
        {"params": critic_1.params}, flat_observations, flat_actions
    )
    q_2 = critic_2.apply_fn(
        {"params": critic_2.params}, flat_observations, flat_actions
    )
    return jnp.minimum(q_1, q_2).reshape((batch_size, candidate_count))


def _masked_finite_mean(values: Array, mask: Array) -> tuple[Array, Array]:
    """Finite branch mean and count; empty branches use an explicit zero."""

    values = jnp.asarray(values, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.bool_) & jnp.isfinite(values)
    count = jnp.sum(mask.astype(jnp.float32))
    mean = jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(count, 1.0)
    return mean, count


def _proposal_span_diagnostics(
    reward_q: Array,
    safety_q: Array,
    feasible: Array,
    valid: Array,
    *,
    alpha_reward: float,
    beta_safety: float,
) -> dict[str, Array]:
    """Return per-branch p90-p10 critic spans and scaled equivalents."""

    qr_low, qr_high = jnp.quantile(
        reward_q, jnp.asarray([0.1, 0.9], dtype=jnp.float32), axis=1
    )
    qh_low, qh_high = jnp.quantile(
        safety_q, jnp.asarray([0.1, 0.9], dtype=jnp.float32), axis=1
    )
    qr_span = qr_high - qr_low
    qh_span = qh_high - qh_low
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    feasible_mask = valid & jnp.asarray(feasible, dtype=jnp.bool_)
    recovery_mask = valid & (~jnp.asarray(feasible, dtype=jnp.bool_))
    result: dict[str, Array] = {}
    for branch, mask in (
        ("feasible", feasible_mask),
        ("recovery", recovery_mask),
    ):
        qr_mean, branch_count = _masked_finite_mean(qr_span, mask)
        qh_mean, _ = _masked_finite_mean(qh_span, mask)
        result[f"posterior_{branch}_row_count"] = branch_count
        result[f"posterior_{branch}_qr_span_p90_p10"] = qr_mean
        result[f"posterior_{branch}_qh_span_p90_p10"] = qh_mean
        result[f"posterior_{branch}_alpha_qr_span_p90_p10"] = (
            jnp.asarray(alpha_reward, dtype=jnp.float32) * qr_mean
        )
        result[f"posterior_{branch}_beta_qh_span_p90_p10"] = (
            jnp.asarray(beta_safety, dtype=jnp.float32) * qh_mean
        )
    return result


def _update_qh_with_particle_candidates(
    core: SSMOnlineAgent,
    batch,
    next_candidate_actions: Array,
) -> tuple[SSMOnlineAgent, dict[str, Array]]:
    """Q_h update whose HJ minimum is exactly the supplied policy particles."""

    if (
        next_candidate_actions.ndim != 3
        or next_candidate_actions.shape[0]
        != batch["next_observations"].shape[0]
        or next_candidate_actions.shape[2] != core.act_dim
        or next_candidate_actions.shape[1] <= 0
    ):
        raise ValueError(
            "next_candidate_actions must have shape [B,N,act_dim], N>0."
        )
    next_q_candidates = _evaluate_particle_safety_candidates(
        core,
        batch["next_observations"],
        next_candidate_actions,
        use_target=True,
    )
    next_qh = jnp.min(next_q_candidates, axis=1)
    candidate_count = jnp.asarray(
        next_candidate_actions.shape[1], dtype=jnp.float32
    )
    gap = jnp.asarray(core.h_hardgap, dtype=batch["costs"].dtype)
    h_qh = jnp.where(
        (batch["costs"] > 0.0) & (gap > 0.0), gap, batch["costs"]
    )
    qh_nonterminal = (
        (1.0 - core.discount_h) * h_qh
        + core.discount_h * jnp.maximum(h_qh, next_qh)
    )
    target_qh = jax.lax.stop_gradient(
        qh_nonterminal * batch["masks"] + h_qh * (1 - batch["masks"])
    )

    def loss_1(params):
        prediction = core.safe_critic.apply_fn(
            {"params": params}, batch["observations"], batch["actions"]
        )
        return jnp.mean(jnp.square(prediction - target_qh))

    safe_critic_loss_1, grads_1 = jax.value_and_grad(loss_1)(
        core.safe_critic.params
    )
    safe_critic = core.safe_critic.apply_gradients(grads=grads_1)
    if core.qh_critic_mode != "single":
        def loss_2(params):
            prediction = core.safe_critic_2.apply_fn(
                {"params": params}, batch["observations"], batch["actions"]
            )
            return jnp.mean(jnp.square(prediction - target_qh))

        safe_critic_loss_2, grads_2 = jax.value_and_grad(loss_2)(
            core.safe_critic_2.params
        )
        safe_critic_2 = core.safe_critic_2.apply_gradients(grads=grads_2)
        safe_critic_loss = 0.5 * (
            safe_critic_loss_1 + safe_critic_loss_2
        )
    else:
        safe_critic_2 = core.safe_critic_2
        safe_critic_loss_2 = jnp.asarray(
            0.0, dtype=safe_critic_loss_1.dtype
        )
        safe_critic_loss = safe_critic_loss_1

    qh, qh_1, qh_2 = SSMOnlineAgent._evaluate_safe_critics(
        core,
        batch["observations"],
        batch["actions"],
        use_target=False,
    )
    disagreement = jnp.abs(qh_1 - qh_2)
    head_2_active = (qh_2 > qh_1).astype(jnp.float32)
    safe_threshold = core.delta - core.hard_margin
    info = {
        "safe_critic_loss": safe_critic_loss,
        "safe_critic_loss_1": safe_critic_loss_1,
        "safe_critic_loss_2": safe_critic_loss_2,
        "qh_mean": jnp.mean(qh),
        "qh_max": jnp.max(qh),
        "qh_min": jnp.min(qh),
        "qh_1_mean": jnp.mean(qh_1),
        "qh_2_mean": jnp.mean(qh_2),
        "qh_disagreement_abs_mean": jnp.mean(disagreement),
        "qh_disagreement_abs_max": jnp.max(disagreement),
        "qh_head_2_active_frac": jnp.mean(head_2_active),
        "qh_twin_enabled": jnp.asarray(
            core.qh_critic_mode != "single", dtype=jnp.float32
        ),
        "qh_head0_reducer": jnp.asarray(
            core.qh_critic_mode == "twin_head0", dtype=jnp.float32
        ),
        "qh_max_reducer": jnp.asarray(
            core.qh_critic_mode == "twin_max", dtype=jnp.float32
        ),
        "target_qh_mean": jnp.mean(target_qh),
        "target_qh_max": jnp.max(target_qh),
        "target_qh_min": jnp.min(target_qh),
        "costs_mean": jnp.mean(batch["costs"]),
        "raw_h_mean": jnp.mean(batch["costs"]),
        "h_qh_mean": jnp.mean(h_qh),
        "h_hardgap": gap,
        "hardgap_active_frac": jnp.mean(
            ((batch["costs"] > 0.0) & (gap > 0.0)).astype(jnp.float32)
        ),
        "qh_bootstrap_mean": jnp.mean(next_qh),
        "safe_bootstrap_mean": jnp.mean(next_qh),
        "qh_bootstrap_candidate_count": candidate_count,
        "safe_bootstrap_candidate_count": candidate_count,
        "safe_threshold": jnp.asarray(safe_threshold, dtype=jnp.float32),
    }
    target_params_1 = optax.incremental_update(
        safe_critic.params, core.safe_target_critic.params, core.tau
    )
    safe_target_critic = core.safe_target_critic.replace(
        params=target_params_1
    )
    if core.qh_critic_mode != "single":
        target_params_2 = optax.incremental_update(
            safe_critic_2.params, core.safe_target_critic_2.params, core.tau
        )
        safe_target_critic_2 = core.safe_target_critic_2.replace(
            params=target_params_2
        )
    else:
        safe_target_critic_2 = core.safe_target_critic_2
    return core.replace(
        safe_critic=safe_critic,
        safe_target_critic=safe_target_critic,
        safe_critic_2=safe_critic_2,
        safe_target_critic_2=safe_target_critic_2,
    ), info


@struct.dataclass
class LPPSAgent:
    """Reference-anchored latent posterior-SNIS actor with legacy critics."""

    core: SSMOnlineAgent
    reference_score_params: Any
    actor_update_count: Array
    reference_block_count: Array

    posterior_samples: int = struct.field(pytree_node=False, default=64)
    likelihood_fraction: float = struct.field(pytree_node=False, default=1.0)
    proposal_mode: str = struct.field(
        pytree_node=False, default=LP_PROPOSAL_MODE
    )
    bridge_likelihood_fraction: float = struct.field(
        pytree_node=False, default=0.5
    )
    bridge_reference_samples: int = struct.field(pytree_node=False, default=0)
    bridge_latent_std: float = struct.field(pytree_node=False, default=0.0)
    bridge_component_weighting: str = struct.field(
        pytree_node=False, default=BRIDGE_COMPONENT_WEIGHTING_UNIFORM
    )
    proposal_chunk_size: int = struct.field(pytree_node=False, default=32)
    # ``eta`` is the serialized legacy name.  Its only semantics are
    # rho_anchor: the proximal interpolation between reference and posterior
    # epsilon targets. It is distinct from the proposal mixture fraction.
    eta: float = struct.field(pytree_node=False, default=0.5)
    reference_refresh_interval: int = struct.field(
        pytree_node=False, default=500
    )
    actor_lr: float = struct.field(pytree_node=False, default=1e-4)
    actor_grad_clip_norm: float = struct.field(pytree_node=False, default=1.0)
    safety_threshold: float = struct.field(pytree_node=False, default=0.0)
    epsilon_action: float = struct.field(pytree_node=False, default=1e-5)
    actor_safety_mode: str = struct.field(
        pytree_node=False, default=LP_ACTOR_SAFETY_QH_ROUTED
    )
    particle_mode: str = struct.field(
        pytree_node=False, default=LP_PARTICLE_MODE_S1_MIN
    )
    num_particles_data: int = struct.field(pytree_node=False, default=1)
    num_particles_target: int = struct.field(pytree_node=False, default=1)
    num_particles_train: int = struct.field(pytree_node=False, default=1)
    data_particle_ess_fraction: float = struct.field(
        pytree_node=False, default=0.5
    )
    target_particle_ess_fraction: float = struct.field(
        pytree_node=False, default=0.5
    )
    train_particle_ess_fraction: float = struct.field(
        pytree_node=False, default=0.5
    )
    reward_backup_mode: str = struct.field(
        pytree_node=False, default=LP_REWARD_BACKUP_ONLINE_K1
    )
    reward_backup_samples: int = struct.field(pytree_node=False, default=1)
    qh_backup_uses_reference: bool = struct.field(
        pytree_node=False, default=False
    )
    reward_tilt_mode: str = struct.field(
        pytree_node=False, default=LP_REWARD_TILT_FIXED_ALPHA
    )
    reward_kl_delta: float = struct.field(pytree_node=False, default=0.25)
    reward_row_kl_cap: float = struct.field(pytree_node=False, default=1.0)

    @classmethod
    def from_core(
        cls,
        core: SSMOnlineAgent,
        *,
        posterior_samples: int = 64,
        likelihood_fraction: float = 1.0,
        proposal_mode: str | None = None,
        bridge_likelihood_fraction: float = 0.5,
        bridge_reference_samples: int = 0,
        bridge_latent_std: float = 0.0,
        bridge_component_weighting: str = BRIDGE_COMPONENT_WEIGHTING_UNIFORM,
        proposal_chunk_size: int = 32,
        eta: float = 0.5,
        reference_refresh_interval: int = 500,
        actor_lr: float = 1e-4,
        actor_grad_clip_norm: float = 1.0,
        safety_threshold: float = 0.0,
        epsilon_action: float = 1e-5,
        actor_safety_mode: str = LP_ACTOR_SAFETY_QH_ROUTED,
        particle_mode: str = LP_PARTICLE_MODE_S1_MIN,
        num_particles_data: int = 1,
        num_particles_target: int = 1,
        num_particles_train: int = 1,
        data_particle_ess_fraction: float = 0.5,
        target_particle_ess_fraction: float = 0.5,
        train_particle_ess_fraction: float = 0.5,
        reward_backup_mode: str = LP_REWARD_BACKUP_ONLINE_K1,
        reward_backup_samples: int = 1,
        qh_backup_uses_reference: bool = False,
        reward_tilt_mode: str = LP_REWARD_TILT_FIXED_ALPHA,
        reward_kl_delta: float = 0.25,
        reward_row_kl_cap: float = 1.0,
    ) -> "LPPSAgent":
        """Convert a freshly initialized legacy critic container to LP-PS."""

        if core.stage_mode != "stage_b":
            raise ValueError("LP-PS-SSM requires stage_mode='stage_b'.")
        if actor_safety_mode not in LP_ACTOR_SAFETY_MODES:
            raise ValueError(
                f"actor_safety_mode must be one of {LP_ACTOR_SAFETY_MODES}."
            )
        if core.qh_critic_mode not in ("single", "twin_head0", "twin_max"):
            raise ValueError(
                "LP-PS-SSM qh_critic_mode must be single, twin_head0, or twin_max."
            )
        if core.qh_candidate_source != "policy_plus_gaussian":
            raise ValueError(
                "LP-PS-SSM requires qh_candidate_source='policy_plus_gaussian'."
            )
        if int(core.qh_min_k_samples) != 8:
            raise ValueError("The formal LP-PS gate requires K=8.")
        if abs(float(core.qh_min_k_sigma) - 0.3) > 1e-8:
            raise ValueError("The formal LP-PS gate requires sigma=0.3.")
        if posterior_samples <= 0:
            raise ValueError("posterior_samples must be positive.")
        if not 0.0 <= likelihood_fraction <= 1.0:
            raise ValueError("likelihood_fraction must lie in [0,1].")
        resolved_proposal_mode = lp_proposal_mode(
            likelihood_fraction, proposal_mode
        )
        if resolved_proposal_mode == LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE:
            if likelihood_fraction != 1.0:
                raise ValueError(
                    "Bridge mode keeps legacy likelihood_fraction fixed at 1.0; "
                    "use bridge_likelihood_fraction for proposal omega."
                )
            if not 0.0 <= bridge_likelihood_fraction <= 1.0:
                raise ValueError(
                    "bridge_likelihood_fraction must lie in [0,1]."
                )
            if bridge_reference_samples <= 0:
                raise ValueError(
                    "bridge_reference_samples must be positive in bridge mode."
                )
            if bridge_latent_std <= 0.0:
                raise ValueError(
                    "bridge_latent_std must be positive in bridge mode."
                )
            if bridge_component_weighting not in BRIDGE_COMPONENT_WEIGHTINGS:
                raise ValueError(
                    "bridge_component_weighting must be one of "
                    f"{BRIDGE_COMPONENT_WEIGHTINGS}."
                )
        if proposal_chunk_size <= 0:
            raise ValueError("proposal_chunk_size must be positive.")
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must lie in [0,1].")
        if reference_refresh_interval <= 0:
            raise ValueError("reference_refresh_interval must be positive.")
        if not 0.0 < epsilon_action < 1.0:
            raise ValueError("epsilon_action must lie in (0,1).")
        if particle_mode not in LP_PARTICLE_MODES:
            raise ValueError(
                f"particle_mode must be one of {LP_PARTICLE_MODES}."
            )
        particle_counts = (
            int(num_particles_data),
            int(num_particles_target),
            int(num_particles_train),
        )
        if particle_mode == LP_PARTICLE_MODE_S1_MIN:
            if particle_counts != (1, 1, 1):
                raise ValueError(
                    "s1_min requires inert particle counts "
                    "N_data=N_target=N_train=1."
                )
        elif particle_counts != (8, 8, 8):
            raise ValueError(
                f"{particle_mode} requires N_data=N_target=N_train=8."
            )
        for name, value in (
            ("data_particle_ess_fraction", data_particle_ess_fraction),
            ("target_particle_ess_fraction", target_particle_ess_fraction),
            ("train_particle_ess_fraction", train_particle_ess_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0,1].")
        if reward_backup_mode not in LP_REWARD_BACKUP_MODES:
            raise ValueError(
                f"reward_backup_mode must be one of {LP_REWARD_BACKUP_MODES}."
            )
        if reward_backup_samples <= 0:
            raise ValueError("reward_backup_samples must be positive.")
        if reward_backup_mode == LP_REWARD_BACKUP_ONLINE_K1:
            if reward_backup_samples != 1 or qh_backup_uses_reference:
                raise ValueError(
                    "online_policy_k1 requires reward_backup_samples=1 and "
                    "qh_backup_uses_reference=False."
                )
        if reward_tilt_mode not in LP_REWARD_TILT_MODES:
            raise ValueError(
                f"reward_tilt_mode must be one of {LP_REWARD_TILT_MODES}."
            )
        if reward_kl_delta < 0.0 or not float(reward_kl_delta) < float("inf"):
            raise ValueError("reward_kl_delta must be finite and non-negative.")
        if (
            reward_row_kl_cap < 0.0
            or not float(reward_row_kl_cap) < float("inf")
        ):
            raise ValueError(
                "reward_row_kl_cap must be finite and non-negative."
            )
        if (
            reward_tilt_mode == LP_REWARD_TILT_CAPPED_BATCH_KL
            and reward_row_kl_cap < reward_kl_delta
        ):
            raise ValueError(
                "capped_batch_kl requires reward_row_kl_cap >= "
                "reward_kl_delta."
            )

        core = rebuild_actor_optimizer(
            core,
            actor_lr=actor_lr,
            grad_clip_norm=actor_grad_clip_norm,
        )
        return cls(
            core=core,
            reference_score_params=copy_params(core.score_model.params),
            actor_update_count=jnp.asarray(0, dtype=jnp.int32),
            reference_block_count=jnp.asarray(0, dtype=jnp.int32),
            posterior_samples=int(posterior_samples),
            likelihood_fraction=float(likelihood_fraction),
            proposal_mode=resolved_proposal_mode,
            bridge_likelihood_fraction=float(bridge_likelihood_fraction),
            bridge_reference_samples=int(bridge_reference_samples),
            bridge_latent_std=float(bridge_latent_std),
            bridge_component_weighting=str(bridge_component_weighting),
            proposal_chunk_size=int(proposal_chunk_size),
            eta=float(eta),
            reference_refresh_interval=int(reference_refresh_interval),
            actor_lr=float(actor_lr),
            actor_grad_clip_norm=float(actor_grad_clip_norm),
            safety_threshold=float(safety_threshold),
            epsilon_action=float(epsilon_action),
            actor_safety_mode=str(actor_safety_mode),
            particle_mode=str(particle_mode),
            num_particles_data=int(num_particles_data),
            num_particles_target=int(num_particles_target),
            num_particles_train=int(num_particles_train),
            data_particle_ess_fraction=float(data_particle_ess_fraction),
            target_particle_ess_fraction=float(target_particle_ess_fraction),
            train_particle_ess_fraction=float(train_particle_ess_fraction),
            reward_backup_mode=str(reward_backup_mode),
            reward_backup_samples=int(reward_backup_samples),
            qh_backup_uses_reference=bool(qh_backup_uses_reference),
            reward_tilt_mode=str(reward_tilt_mode),
            reward_kl_delta=float(reward_kl_delta),
            reward_row_kl_cap=float(reward_row_kl_cap),
        )

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        *,
        actor_lr: float = 1e-4,
        actor_grad_clip_norm: float = 1.0,
        posterior_samples: int = 64,
        likelihood_fraction: float = 1.0,
        proposal_mode: str | None = None,
        bridge_likelihood_fraction: float = 0.5,
        bridge_reference_samples: int = 0,
        bridge_latent_std: float = 0.0,
        bridge_component_weighting: str = BRIDGE_COMPONENT_WEIGHTING_UNIFORM,
        proposal_chunk_size: int = 32,
        eta: float = 0.5,
        reference_refresh_interval: int = 500,
        safety_threshold: float = 0.0,
        epsilon_action: float = 1e-5,
        actor_safety_mode: str = LP_ACTOR_SAFETY_QH_ROUTED,
        qh_critic_mode: str = "twin_max",
        particle_mode: str = LP_PARTICLE_MODE_S1_MIN,
        num_particles_data: int = 1,
        num_particles_target: int = 1,
        num_particles_train: int = 1,
        data_particle_ess_fraction: float = 0.5,
        target_particle_ess_fraction: float = 0.5,
        train_particle_ess_fraction: float = 0.5,
        reward_backup_mode: str = LP_REWARD_BACKUP_ONLINE_K1,
        reward_backup_samples: int = 1,
        qh_backup_uses_reference: bool = False,
        reward_tilt_mode: str = LP_REWARD_TILT_FIXED_ALPHA,
        reward_kl_delta: float = 0.25,
        reward_row_kl_cap: float = 1.0,
        **legacy_core_kwargs,
    ) -> "LPPSAgent":
        """Create a Stage-B posterior actor with explicit Q_h semantics."""

        forced = {
            "actor_lr": actor_lr,
            "stage_mode": "stage_b",
            "qh_critic_mode": qh_critic_mode,
            "qh_candidate_source": "policy_plus_gaussian",
            "qh_min_k_samples": 8,
            "qh_min_k_sigma": 0.3,
            "clip_sampler": False,
            # Inert compatibility field: the LP actor never reads M_q.
            "M_q": 1.0,
        }
        for name, value in forced.items():
            legacy_core_kwargs[name] = value
        core = SSMOnlineAgent.create(
            seed=seed,
            observation_space=observation_space,
            action_space=action_space,
            **legacy_core_kwargs,
        )
        return cls.from_core(
            core,
            posterior_samples=posterior_samples,
            likelihood_fraction=likelihood_fraction,
            proposal_mode=proposal_mode,
            bridge_likelihood_fraction=bridge_likelihood_fraction,
            bridge_reference_samples=bridge_reference_samples,
            bridge_latent_std=bridge_latent_std,
            bridge_component_weighting=bridge_component_weighting,
            proposal_chunk_size=proposal_chunk_size,
            eta=eta,
            reference_refresh_interval=reference_refresh_interval,
            actor_lr=actor_lr,
            actor_grad_clip_norm=actor_grad_clip_norm,
            safety_threshold=safety_threshold,
            epsilon_action=epsilon_action,
            actor_safety_mode=actor_safety_mode,
            particle_mode=particle_mode,
            num_particles_data=num_particles_data,
            num_particles_target=num_particles_target,
            num_particles_train=num_particles_train,
            data_particle_ess_fraction=data_particle_ess_fraction,
            target_particle_ess_fraction=target_particle_ess_fraction,
            train_particle_ess_fraction=train_particle_ess_fraction,
            reward_backup_mode=reward_backup_mode,
            reward_backup_samples=reward_backup_samples,
            qh_backup_uses_reference=qh_backup_uses_reference,
            reward_tilt_mode=reward_tilt_mode,
            reward_kl_delta=reward_kl_delta,
            reward_row_kl_cap=reward_row_kl_cap,
        )

    @property
    def rho_anchor(self) -> float:
        """Explicit name for the legacy-serialized proximal anchor ``eta``."""

        return self.eta

    def policy_latents(
        self,
        observations: Array,
        rng: Array,
        *,
        use_reference: bool = False,
    ) -> tuple[Array, Array]:
        """Sample online or block-frozen reference latents."""

        params = (
            self.reference_score_params
            if use_reference
            else self.core.score_model.params
        )
        return latent_ddpm_sampler(
            self.core.score_model.apply_fn,
            params,
            num_steps=self.core.T,
            rng=rng,
            action_dim=self.core.act_dim,
            observations=observations,
            alphas=self.core.alphas,
            alpha_hats=self.core.alpha_hats,
            betas=self.core.betas,
            sample_temperature=self.core.ddpm_temperature,
        )

    def policy_actions(
        self,
        observations: Array,
        rng: Array,
        *,
        use_reference: bool = False,
    ) -> tuple[Array, Array]:
        """Sample bounded actions; tanh is the only policy-bound operation."""

        latent, rng = self.policy_latents(
            observations, rng, use_reference=use_reference
        )
        return bounded_latent_to_action(latent, self.epsilon_action), rng

    def policy_action_samples(
        self,
        observations: Array,
        rng: Array,
        *,
        num_samples: int,
        use_reference: bool,
    ) -> tuple[Array, Array]:
        """Draw independent full-policy actions with shape ``[B,L,A]``."""

        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.ndim != 2:
            raise ValueError("observations must have shape [B,D].")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        repeated = jnp.repeat(
            observations[:, None, :], num_samples, axis=1
        ).reshape((-1, observations.shape[-1]))
        actions, rng = self.policy_actions(
            repeated, rng, use_reference=use_reference
        )
        return actions.reshape(
            (observations.shape[0], num_samples, self.core.act_dim)
        ), rng

    def policy_particle_batch(
        self,
        observations: Array,
        rng: Array,
        *,
        num_particles: int,
        target_ess_fraction: float,
        use_reference: bool,
        use_target_critics: bool,
    ) -> tuple[LPPolicyParticleBatch, Array]:
        """Draw, score, and safely sample independent full-policy actions."""

        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.ndim != 2:
            raise ValueError("observations must have shape [B,D].")
        if num_particles != 8:
            raise ValueError("Formal policy-particle roles require N=8.")
        if not 0.0 < target_ess_fraction <= 1.0:
            raise ValueError("target_ess_fraction must lie in (0,1].")
        batch_size, observation_dim = observations.shape
        repeated = jnp.repeat(
            observations[:, None, :], num_particles, axis=1
        ).reshape((batch_size * num_particles, observation_dim))
        policy_key, categorical_key, next_rng = jax.random.split(rng, 3)
        flat_latents, _ = self.policy_latents(
            repeated, policy_key, use_reference=use_reference
        )
        latents = flat_latents.reshape(
            (batch_size, num_particles, self.core.act_dim)
        )
        actions = bounded_latent_to_action(latents, self.epsilon_action)
        reward_q = _evaluate_particle_reward_candidates(
            self.core,
            observations,
            actions,
            use_target=use_target_critics,
        )
        safety_q = _evaluate_particle_safety_candidates(
            self.core,
            observations,
            actions,
            use_target=use_target_critics,
        )
        selection = sample_safe_particle(
            reward_q,
            safety_q,
            categorical_key,
            safety_threshold=self.safety_threshold,
            target_ess_fraction=target_ess_fraction,
        )
        row = jnp.arange(batch_size, dtype=jnp.int32)
        return LPPolicyParticleBatch(
            latents=jax.lax.stop_gradient(latents),
            actions=jax.lax.stop_gradient(actions),
            reward_q=jax.lax.stop_gradient(reward_q),
            safety_q=jax.lax.stop_gradient(safety_q),
            selection=selection,
            selected_latents=jax.lax.stop_gradient(
                latents[row, selection.index]
            ),
            selected_actions=jax.lax.stop_gradient(
                actions[row, selection.index]
            ),
            rng_routing_valid=jnp.asarray(
                ~jnp.all(policy_key == categorical_key),
                dtype=jnp.float32,
            ),
            key_collision_rate=jnp.asarray(
                jnp.all(policy_key == categorical_key),
                dtype=jnp.float32,
            ),
        ), next_rng

    def data_particle_action_with_diagnostics(
        self, observation: Array
    ) -> tuple[Array, "LPPSAgent", dict[str, Array]]:
        """Sample a safe-softmax behavior action from N=8 policy particles."""

        if self.particle_mode == LP_PARTICLE_MODE_S1_MIN:
            raise ValueError("s1_min cannot execute the particle behavior path.")
        observation = jnp.asarray(observation, dtype=jnp.float32)
        if observation.ndim != 1:
            raise ValueError("particle behavior expects one observation [D].")
        particles, rng = self.policy_particle_batch(
            observation[None, :],
            self.core.rng,
            num_particles=self.num_particles_data,
            target_ess_fraction=self.data_particle_ess_fraction,
            use_reference=False,
            use_target_critics=False,
        )
        selection = particles.selection
        distribution = selection.distribution
        selected_index = selection.index[0]
        selected = particles.selected_actions[0]
        valid_row = (
            (distribution.eligible_count[0] > 0.0)
            & (distribution.invalid_candidate_fraction[0] == 0.0)
            & jnp.all(jnp.isfinite(selected))
        )
        # pack_behavior_guard already treats a non-finite action as fatal.  This
        # makes particle critic corruption fail before env.step().
        action = jnp.where(valid_row, selected, jnp.full_like(selected, jnp.nan))
        row = jnp.asarray(0, dtype=jnp.int32)
        diagnostics = {
            "behavior_latent_abs_max": jnp.max(jnp.abs(particles.latents[0])),
            "behavior_latent_norm": jnp.linalg.norm(
                particles.selected_latents[0]
            ),
            "behavior_latent_nonfinite_fraction": jnp.mean(
                (~jnp.isfinite(particles.latents[0])).astype(jnp.float32)
            ),
            "behavior_action_saturation_fraction": jnp.mean(
                (jnp.abs(selected) >= 0.999).astype(jnp.float32)
            ),
            "particle_data_active": jnp.asarray(1.0, dtype=jnp.float32),
            "particle_data_n": jnp.asarray(
                float(self.num_particles_data), dtype=jnp.float32
            ),
            "particle_data_temperature": distribution.temperature[0],
            "particle_data_ess": distribution.categorical_ess[0],
            "particle_data_target_ess": distribution.target_ess[0],
            "particle_data_feasible_fraction": distribution.feasible_fraction[0],
            "particle_data_no_feasible": distribution.no_feasible[0].astype(
                jnp.float32
            ),
            "particle_data_eligible_count": distribution.eligible_count[0],
            "particle_data_invalid_candidate_fraction": (
                distribution.invalid_candidate_fraction[0]
            ),
            "particle_data_selected_index": selected_index.astype(jnp.float32),
            "particle_data_selected_probability": selection.selected_probability[
                0
            ],
            "particle_data_selected_reward_rank": selection.selected_reward_rank[
                0
            ].astype(jnp.float32),
            "particle_data_selected_safety_rank": selection.selected_safety_rank[
                0
            ].astype(jnp.float32),
            "particle_data_selected_qr": particles.reward_q[row, selected_index],
            "particle_data_selected_qh": particles.safety_q[row, selected_index],
            "particle_data_selected_vs_particle0_l2": jnp.linalg.norm(
                selected - particles.actions[0, 0]
            ),
            "particle_data_row_valid": valid_row.astype(jnp.float32),
            "particle_data_rng_routing_valid": particles.rng_routing_valid,
            "particle_data_key_collision_rate": particles.key_collision_rate,
        }
        return (
            action,
            self.replace(core=self.core.replace(rng=rng)),
            diagnostics,
        )

    def sample_reference_bridge_centers(
        self, observations: Array, rng: Array
    ) -> Array:
        """Draw an independent per-state latent pool from the frozen actor."""

        if self.proposal_mode != LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE:
            raise ValueError("Reference bridge centres require bridge mode.")
        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.ndim != 2:
            raise ValueError("observations must have shape [B,D].")
        batch_size, observation_dim = observations.shape
        repeated = jnp.repeat(
            observations[:, None, :],
            self.bridge_reference_samples,
            axis=1,
        ).reshape((batch_size * self.bridge_reference_samples, observation_dim))
        centers, _ = self.policy_latents(
            repeated, rng, use_reference=True
        )
        return jax.lax.stop_gradient(
            centers.reshape(
                (batch_size, self.bridge_reference_samples, self.core.act_dim)
            )
        )

    def eval_actions(self, observation: Array) -> tuple[Array, "LPPSAgent"]:
        """Raw K=1 stochastic diffusion action for evaluation."""

        actions, agent, _ = self.action_with_diagnostics(observation)
        return actions, agent

    def action_with_diagnostics(
        self, observation: Array
    ) -> tuple[Array, "LPPSAgent", dict[str, Array]]:
        """Sample one raw action while exposing the unbounded latent.

        Bounds alone cannot diagnose a failed latent policy because ``tanh``
        maps even an infinite latent to a finite endpoint.  The online runner
        therefore consumes these diagnostics before executing the action.
        """

        observation = jnp.asarray(observation, dtype=jnp.float32)
        if observation.ndim != 1:
            raise ValueError(
                "action_with_diagnostics expects one observation [D], got "
                f"{observation.shape!r}."
            )
        latent, rng = self.policy_latents(
            observation[None, :], self.core.rng, use_reference=False
        )
        actions = bounded_latent_to_action(latent, self.epsilon_action)
        diagnostics = {
            "behavior_latent_abs_max": jnp.max(jnp.abs(latent)),
            "behavior_latent_norm": jnp.linalg.norm(latent[0]),
            "behavior_latent_nonfinite_fraction": jnp.mean(
                (~jnp.isfinite(latent)).astype(jnp.float32)
            ),
            "behavior_action_saturation_fraction": jnp.mean(
                (jnp.abs(actions) >= 0.999).astype(jnp.float32)
            ),
        }
        return (
            actions[0],
            self.replace(core=self.core.replace(rng=rng)),
            diagnostics,
        )

    def sample_actions(self, observation: Array) -> tuple[Array, "LPPSAgent"]:
        """Raw K=1 online action; DDPM stochasticity supplies exploration."""

        return self.eval_actions(observation)

    def estimate_stage_b_gate(
        self, observations: Array, rng: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Freeze one current-policy K=8 Gaussian Stage-B gate per actor row."""

        observations = jnp.asarray(observations, dtype=jnp.float32)
        base_action, rng = self.policy_actions(
            observations, rng, use_reference=False
        )
        gaussian_key, rng = jax.random.split(rng)
        neighbor_count = int(self.core.qh_min_k_samples) - 1
        gaussian_noise = jax.random.normal(
            gaussian_key,
            (observations.shape[0], neighbor_count, self.core.act_dim),
            dtype=jnp.float32,
        )
        raw_neighbors = (
            base_action[:, None, :]
            + self.core.qh_min_k_sigma * gaussian_noise
        )
        preclip_fraction = jnp.mean(
            (jnp.abs(raw_neighbors) >= 1.0).astype(jnp.float32)
        )
        strict_bound = jnp.nextafter(
            jnp.asarray(1.0, dtype=raw_neighbors.dtype),
            jnp.asarray(0.0, dtype=raw_neighbors.dtype),
        )
        neighbors = jnp.clip(
            raw_neighbors,
            -strict_bound,
            strict_bound,
        )
        candidates = jnp.concatenate([base_action[:, None, :], neighbors], axis=1)
        candidate_qh = _evaluate_online_safety_candidates(
            self.core, observations, candidates
        )
        gate_qh = jnp.min(candidate_qh, axis=1)
        state_threshold = self.core.delta - self.core.hard_margin
        feasible = gate_qh <= state_threshold
        return feasible, gate_qh, candidates, preclip_fraction, rng

    def posterior_target(
        self,
        observations: Array,
        noisy_latent: Array,
        alpha: Array,
        sigma: Array,
        feasible_state: Array,
        rng: Array,
    ) -> LPPosteriorTarget:
        """Compute the posterior epsilon target using the supplied PRNG key."""

        if self.proposal_mode == LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE:
            # Preserve the likelihood endpoint bit-for-bit: do not split the
            # key or sample a reference pool when omega=1.
            if self.bridge_likelihood_fraction == 1.0:
                return latent_balance_mis_posterior_target(
                    self.core,
                    observations,
                    noisy_latent,
                    alpha,
                    sigma,
                    feasible_state,
                    rng,
                    num_samples=self.posterior_samples,
                    likelihood_fraction=1.0,
                    proposal_chunk_size=self.proposal_chunk_size,
                    alpha_reward=self.core.alpha_r,
                    beta_safety=self.core.beta_safety,
                    reward_tilt_mode=self.reward_tilt_mode,
                    reward_kl_delta=self.reward_kl_delta,
                    reward_row_kl_cap=self.reward_row_kl_cap,
                    safety_threshold=self.safety_threshold,
                    boundary_epsilon=self.epsilon_action,
                    actor_safety_mode=self.actor_safety_mode,
                )
            center_key, proposal_key = jax.random.split(rng)
            reference_centers = self.sample_reference_bridge_centers(
                observations, center_key
            )
            return latent_defensive_bridge_mis_posterior_target(
                self.core,
                observations,
                noisy_latent,
                alpha,
                sigma,
                feasible_state,
                proposal_key,
                reference_centers,
                num_samples=self.posterior_samples,
                bridge_likelihood_fraction=self.bridge_likelihood_fraction,
                bridge_latent_std=self.bridge_latent_std,
                bridge_component_weighting=self.bridge_component_weighting,
                proposal_chunk_size=self.proposal_chunk_size,
                alpha_reward=self.core.alpha_r,
                beta_safety=self.core.beta_safety,
                reward_tilt_mode=self.reward_tilt_mode,
                reward_kl_delta=self.reward_kl_delta,
                reward_row_kl_cap=self.reward_row_kl_cap,
                safety_threshold=self.safety_threshold,
                boundary_epsilon=self.epsilon_action,
                actor_safety_mode=self.actor_safety_mode,
            )
        return latent_balance_mis_posterior_target(
            self.core,
            observations,
            noisy_latent,
            alpha,
            sigma,
            feasible_state,
            rng,
            num_samples=self.posterior_samples,
            likelihood_fraction=self.likelihood_fraction,
            proposal_chunk_size=self.proposal_chunk_size,
            alpha_reward=self.core.alpha_r,
            beta_safety=self.core.beta_safety,
            reward_tilt_mode=self.reward_tilt_mode,
            reward_kl_delta=self.reward_kl_delta,
            reward_row_kl_cap=self.reward_row_kl_cap,
            safety_threshold=self.safety_threshold,
            boundary_epsilon=self.epsilon_action,
            actor_safety_mode=self.actor_safety_mode,
        )

    def replay_actions_to_latents(self, actions: Array) -> tuple[Array, Array]:
        """Numerically safe inverse tanh and its clipping diagnostic."""

        return bounded_actions_to_latent(actions, self.epsilon_action)

    def update_actor(self, batch) -> tuple["LPPSAgent", dict[str, Array]]:
        """One reference-outer posterior update with one fixed Stage-B gate."""

        observations = jnp.asarray(batch["observations"], dtype=jnp.float32)
        batch_size = observations.shape[0]
        if "actions" in batch:
            _, replay_inverse_clip_mask = self.replay_actions_to_latents(
                batch["actions"]
            )
        else:
            # State-only batches omit inverse-action diagnostics. Online
            # replay batches include actions.
            replay_inverse_clip_mask = jnp.zeros(
                (batch_size, self.core.act_dim), dtype=jnp.bool_
            )
        train_particles = None
        if self.actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY:
            feasible = jnp.ones((batch_size,), dtype=jnp.bool_)
            gate_qh = jnp.zeros((batch_size,), dtype=jnp.float32)
            gate_candidates = jnp.zeros(
                (batch_size, 0, self.core.act_dim), dtype=jnp.float32
            )
            gate_preclip_fraction = jnp.asarray(0.0, dtype=jnp.float32)
            rng = self.core.rng
        elif self.particle_mode == LP_PARTICLE_HJ_FULL_CONSISTENT:
            train_particles, rng = self.policy_particle_batch(
                observations,
                self.core.rng,
                num_particles=self.num_particles_train,
                target_ess_fraction=self.train_particle_ess_fraction,
                use_reference=True,
                use_target_critics=False,
            )
            gate_candidates = train_particles.actions
            gate_qh = jnp.min(train_particles.safety_q, axis=1)
            state_threshold = self.core.delta - self.core.hard_margin
            feasible = gate_qh <= state_threshold
            gate_preclip_fraction = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            (
                feasible,
                gate_qh,
                gate_candidates,
                gate_preclip_fraction,
                rng,
            ) = self.estimate_stage_b_gate(observations, self.core.rng)

        # role_separated changes the actor outer sample but never the HJ gate:
        # the gate above remains an independent policy+Gaussian K=8 draw.
        if self.particle_mode == LP_PARTICLE_HJ_ROLE_SEPARATED:
            train_particles, rng = self.policy_particle_batch(
                observations,
                rng,
                num_particles=self.num_particles_train,
                target_ess_fraction=self.train_particle_ess_fraction,
                use_reference=True,
                use_target_critics=False,
            )
        if train_particles is None:
            outer_latent, rng = self.policy_latents(
                observations, rng, use_reference=True
            )
            particle_train_info = {
                "particle_train_active": jnp.asarray(0.0, dtype=jnp.float32),
                "particle_train_n": jnp.asarray(1.0, dtype=jnp.float32),
                "particle_train_ess_mean": jnp.asarray(1.0, dtype=jnp.float32),
                "particle_train_feasible_fraction_mean": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
                "particle_train_no_feasible_rate": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
                "particle_train_invalid_candidate_fraction_mean": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
                "particle_train_selected_qr_mean": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
                "particle_train_selected_qh_mean": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
                "particle_train_rng_routing_valid": jnp.asarray(
                    1.0, dtype=jnp.float32
                ),
                "particle_train_key_collision_rate": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
            }
        else:
            distribution = train_particles.selection.distribution
            row = jnp.arange(batch_size, dtype=jnp.int32)
            outer_latent = train_particles.selected_latents
            particle_train_info = {
                "particle_train_active": jnp.asarray(1.0, dtype=jnp.float32),
                "particle_train_n": jnp.asarray(
                    float(self.num_particles_train), dtype=jnp.float32
                ),
                "particle_train_ess_mean": jnp.mean(
                    distribution.categorical_ess
                ),
                "particle_train_feasible_fraction_mean": jnp.mean(
                    distribution.feasible_fraction
                ),
                "particle_train_no_feasible_rate": jnp.mean(
                    distribution.no_feasible.astype(jnp.float32)
                ),
                "particle_train_invalid_candidate_fraction_mean": jnp.mean(
                    distribution.invalid_candidate_fraction
                ),
                "particle_train_selected_qr_mean": jnp.mean(
                    train_particles.reward_q[
                        row, train_particles.selection.index
                    ]
                ),
                "particle_train_selected_qh_mean": jnp.mean(
                    train_particles.safety_q[
                        row, train_particles.selection.index
                    ]
                ),
                "particle_train_rng_routing_valid": (
                    train_particles.rng_routing_valid
                ),
                "particle_train_key_collision_rate": (
                    train_particles.key_collision_rate
                ),
            }
        time_key, outer_noise_key, posterior_key, dropout_key, rng = (
            jax.random.split(rng, 5)
        )
        time = jax.random.randint(time_key, (batch_size,), 0, self.core.T)
        outer_noise = jax.random.normal(
            outer_noise_key, outer_latent.shape, dtype=jnp.float32
        )
        alpha_hat = self.core.alpha_hats[time]
        alpha = jnp.sqrt(alpha_hat)
        sigma = jnp.sqrt(1.0 - alpha_hat)
        noisy_latent = (
            alpha[:, None] * outer_latent + sigma[:, None] * outer_noise
        )
        posterior = self.posterior_target(
            observations,
            noisy_latent,
            alpha,
            sigma,
            feasible,
            posterior_key,
        )
        proposal_span_info = _proposal_span_diagnostics(
            posterior.reward_q_min,
            posterior.safety_q_max,
            feasible,
            posterior.valid,
            alpha_reward=self.core.alpha_r,
            beta_safety=self.core.beta_safety,
        )

        reference_prediction = self.core.score_model.apply_fn(
            {"params": self.reference_score_params},
            observations,
            noisy_latent,
            time[:, None],
            rngs={"dropout": dropout_key},
            training=True,
        )
        reference_prediction = jax.lax.stop_gradient(reference_prediction)
        blend = reference_anchored_epsilon_target(
            reference_prediction,
            posterior,
            feasible,
            eta=self.rho_anchor,
        )
        row_valid = posterior.valid.astype(jnp.float32)
        effective_eta = blend.effective_eta
        blend_target = blend.epsilon

        def actor_loss_fn(params):
            # The identical dropout key is essential: eta=0 and identical
            # online/reference params then produce an exactly zero residual.
            prediction = self.core.score_model.apply_fn(
                {"params": params},
                observations,
                noisy_latent,
                time[:, None],
                rngs={"dropout": dropout_key},
                training=True,
            )
            row_loss = jnp.mean(jnp.square(prediction - blend_target), axis=-1)
            loss = jnp.mean(row_loss)
            return loss, (prediction, row_loss)

        (loss, (prediction, row_loss)), grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.core.score_model.params)
        raw_grad_norm = tree_global_norm(grads)
        score_model = self.core.score_model.apply_gradients(grads=grads)
        next_actor_update_count = self.actor_update_count + jnp.asarray(
            1, dtype=jnp.int32
        )
        refresh = (
            next_actor_update_count % self.reference_refresh_interval
        ) == 0
        reference_params = jax.lax.cond(
            refresh,
            lambda _: copy_params(score_model.params),
            lambda _: self.reference_score_params,
            operand=None,
        )
        reference_block_count = self.reference_block_count + refresh.astype(
            jnp.int32
        )
        core = self.core.replace(score_model=score_model, rng=rng)

        feasible_float = feasible.astype(jnp.float32)
        feasible_valid = feasible_float * row_valid
        feasible_valid_denominator = jnp.maximum(
            jnp.sum(feasible_valid), 1.0
        )
        invalid_feasible = blend.invalid_feasible
        invalid_recovery = (~feasible) & blend.invalid_any
        valid_denominator = jnp.maximum(jnp.sum(row_valid), 1.0)
        reward_tilt_active = (
            posterior.reward_tilt_active.astype(jnp.float32)
            * feasible_valid
        )
        reward_tilt_active_count = jnp.sum(reward_tilt_active)
        reward_tilt_active_denominator = jnp.maximum(
            reward_tilt_active_count, 1.0
        )
        reward_tilt_active_kl = (
            posterior.reward_tilt_achieved_kl * reward_tilt_active
        )
        reward_tilt_kl_sum = jnp.sum(reward_tilt_active_kl)
        reward_tilt_kl_square_sum = jnp.sum(
            jnp.square(reward_tilt_active_kl)
        )
        reward_tilt_kl_max = jnp.max(reward_tilt_active_kl)
        reward_tilt_effective_state_fraction = jnp.where(
            reward_tilt_active_count > 0.0,
            jnp.square(reward_tilt_kl_sum)
            / jnp.maximum(
                reward_tilt_kl_square_sum,
                jnp.finfo(jnp.float32).tiny,
            )
            / reward_tilt_active_denominator,
            0.0,
        )
        gate_boundary_fraction = (
            jnp.asarray(0.0, dtype=jnp.float32)
            if self.actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY
            else jnp.mean(
                (jnp.abs(gate_candidates) >= 1.0).astype(jnp.float32)
            )
        )
        metrics = {
            "actor_loss": loss,
            "actor_grad_norm_raw": raw_grad_norm,
            "actor_grad_norm_after_clip_upper_bound": jnp.minimum(
                raw_grad_norm, self.actor_grad_clip_norm
            ),
            "actor_prediction_norm_mean": jnp.mean(
                jnp.linalg.norm(prediction, axis=-1)
            ),
            "actor_output_abs_max": jnp.max(jnp.abs(prediction)),
            "reference_prediction_norm_mean": jnp.mean(
                jnp.linalg.norm(reference_prediction, axis=-1)
            ),
            "reference_output_abs_max": jnp.max(
                jnp.abs(reference_prediction)
            ),
            "actor_reference_output_mse": jnp.mean(
                jnp.square(prediction - reference_prediction)
            ),
            "posterior_target_norm_mean": jnp.mean(
                jnp.linalg.norm(posterior.epsilon, axis=-1)
            ),
            "blend_target_norm_mean": jnp.mean(
                jnp.linalg.norm(blend_target, axis=-1)
            ),
            "posterior_ess_mean": jnp.sum(posterior.ess * row_valid)
            / valid_denominator,
            "posterior_ess_over_m_mean": (
                jnp.sum(posterior.ess * row_valid) / valid_denominator
            )
            / float(self.posterior_samples),
            "posterior_max_weight_mean": jnp.sum(
                posterior.max_weight * row_valid
            )
            / valid_denominator,
            "posterior_reward_tilt_kl_budget_active": jnp.asarray(
                self.reward_tilt_mode != LP_REWARD_TILT_FIXED_ALPHA,
                dtype=jnp.float32,
            ),
            "posterior_reward_tilt_batch_kl_active": jnp.asarray(
                self.reward_tilt_mode
                in (
                    LP_REWARD_TILT_BATCH_KL,
                    LP_REWARD_TILT_CAPPED_BATCH_KL,
                ),
                dtype=jnp.float32,
            ),
            "posterior_reward_tilt_capped_batch_kl_active": jnp.asarray(
                self.reward_tilt_mode == LP_REWARD_TILT_CAPPED_BATCH_KL,
                dtype=jnp.float32,
            ),
            "posterior_reward_tilt_kl_delta": jnp.asarray(
                self.reward_kl_delta, dtype=jnp.float32
            ),
            "posterior_reward_tilt_row_kl_cap": jnp.asarray(
                self.reward_row_kl_cap, dtype=jnp.float32
            ),
            "posterior_reward_tilt_achieved_kl_mean": jnp.sum(
                posterior.reward_tilt_achieved_kl * feasible_valid
            )
            / feasible_valid_denominator,
            "posterior_reward_tilt_active_rate": (
                reward_tilt_active_count / feasible_valid_denominator
            ),
            "posterior_reward_tilt_active_kl_mean": (
                reward_tilt_kl_sum / reward_tilt_active_denominator
            ),
            "posterior_reward_tilt_active_kl_max": reward_tilt_kl_max,
            "posterior_reward_tilt_top1_kl_fraction": jnp.where(
                reward_tilt_kl_sum > 0.0,
                reward_tilt_kl_max / reward_tilt_kl_sum,
                0.0,
            ),
            "posterior_reward_tilt_effective_state_fraction": (
                reward_tilt_effective_state_fraction
            ),
            "posterior_reward_tilt_row_cap_active_rate": jnp.sum(
                posterior.reward_tilt_row_cap_active.astype(jnp.float32)
                * reward_tilt_active
            )
            / reward_tilt_active_denominator,
            "posterior_reward_tilt_target_hit_rate": jnp.sum(
                posterior.reward_tilt_target_hit.astype(jnp.float32)
                * feasible_valid
            )
            / feasible_valid_denominator,
            "posterior_reward_tilt_attainable_rate": jnp.sum(
                posterior.reward_tilt_attainable.astype(jnp.float32)
                * feasible_valid
            )
            / feasible_valid_denominator,
            "posterior_reward_tilt_ess_over_m_mean": jnp.sum(
                posterior.reward_tilt_ess_over_m * feasible_valid
            )
            / feasible_valid_denominator,
            "posterior_reward_tilt_max_weight_mean": jnp.sum(
                posterior.reward_tilt_max_weight * feasible_valid
            )
            / feasible_valid_denominator,
            "posterior_reward_tilt_q_gain_mean": jnp.sum(
                posterior.reward_tilt_q_gain * feasible_valid
            )
            / feasible_valid_denominator,
            "posterior_reward_tilt_flat_rate": jnp.sum(
                posterior.reward_tilt_flat.astype(jnp.float32)
                * feasible_valid
            )
            / feasible_valid_denominator,
            "posterior_reward_tilt_effective_kappa_mean": jnp.sum(
                posterior.reward_tilt_effective_kappa * feasible_valid
            )
            / feasible_valid_denominator,
            "posterior_reward_tilt_tau_proxy_mean": jnp.sum(
                posterior.reward_tilt_tau_proxy * feasible_valid
            )
            / feasible_valid_denominator,
            "posterior_likelihood_weight_mass_mean": jnp.mean(
                posterior.likelihood_weight_mass
            ),
            "posterior_likelihood_fraction_actual": (
                posterior.likelihood_fraction_actual
            ),
            "posterior_valid_rate": jnp.mean(row_valid),
            "posterior_all_zero_rate": 1.0 - jnp.mean(row_valid),
            "invalid_feasible_rate": jnp.mean(
                invalid_feasible.astype(jnp.float32)
            ),
            "invalid_feasible_rate_conditional": jnp.sum(
                invalid_feasible.astype(jnp.float32)
            )
            / jnp.maximum(jnp.sum(feasible_float), 1.0),
            "invalid_recovery_rate": jnp.mean(
                invalid_recovery.astype(jnp.float32)
            ),
            "invalid_recovery_rate_conditional": jnp.sum(
                invalid_recovery.astype(jnp.float32)
            )
            / jnp.maximum(jnp.sum(1.0 - feasible_float), 1.0),
            "feasible_gate_fraction": jnp.mean(feasible_float),
            "recovery_gate_fraction": 1.0 - jnp.mean(feasible_float),
            "stage_b_gate_qh_mean": jnp.mean(gate_qh),
            "stage_b_gate_qh_min": jnp.min(gate_qh),
            "stage_b_gate_qh_max": jnp.max(gate_qh),
            "stage_b_candidate_count": jnp.asarray(
                gate_candidates.shape[1], dtype=jnp.float32
            ),
            "stage_b_candidate_boundary_fraction": gate_boundary_fraction,
            "stage_b_candidate_preclip_fraction": gate_preclip_fraction,
            "actor_safety_qh_routed": jnp.asarray(
                self.actor_safety_mode == LP_ACTOR_SAFETY_QH_ROUTED,
                dtype=jnp.float32,
            ),
            "actor_safety_reward_only": jnp.asarray(
                self.actor_safety_mode == LP_ACTOR_SAFETY_REWARD_ONLY,
                dtype=jnp.float32,
            ),
            "posterior_feasible_acceptance_mean": jnp.sum(
                feasible_float * posterior.feasible_acceptance
            )
            / jnp.maximum(jnp.sum(feasible_float), 1.0),
            "posterior_critic_nonfinite_fraction": jnp.mean(
                posterior.critic_nonfinite_fraction
            ),
            "proposal_action_saturation_fraction": jnp.mean(
                (jnp.abs(posterior.proposal_actions) >= 0.999).astype(jnp.float32)
            ),
            "outer_latent_norm_mean": jnp.mean(
                jnp.linalg.norm(outer_latent, axis=-1)
            ),
            "outer_latent_abs_max": jnp.max(jnp.abs(outer_latent)),
            "noisy_latent_norm_mean": jnp.mean(
                jnp.linalg.norm(noisy_latent, axis=-1)
            ),
            "noisy_latent_abs_max": jnp.max(jnp.abs(noisy_latent)),
            "proposal_latent_abs_max": jnp.max(
                jnp.abs(posterior.proposal_latents)
            ),
            "effective_eta_mean": jnp.mean(effective_eta),
            "replay_action_inverse_tanh_clip_fraction": jnp.mean(
                replay_inverse_clip_mask.astype(jnp.float32)
            ),
            "replay_action_inverse_tanh_clip_row_fraction": jnp.mean(
                jnp.any(replay_inverse_clip_mask, axis=-1).astype(jnp.float32)
            ),
            "replay_action_inverse_tanh_diagnostic_rows": jnp.asarray(
                batch_size if "actions" in batch else 0, dtype=jnp.float32
            ),
            "invalid_rows_reference_only_mse": jnp.sum(
                row_loss * (1.0 - row_valid)
            )
            / jnp.maximum(jnp.sum(1.0 - row_valid), 1.0),
            "reference_refreshed": refresh.astype(jnp.float32),
            "actor_update_count": next_actor_update_count.astype(jnp.float32),
            "reference_block_count": reference_block_count.astype(jnp.float32),
            "legacy_dzfill_actor_used": jnp.asarray(0.0, dtype=jnp.float32),
            "particle_hj_role_separated": jnp.asarray(
                self.particle_mode == LP_PARTICLE_HJ_ROLE_SEPARATED,
                dtype=jnp.float32,
            ),
            "particle_hj_full_consistent": jnp.asarray(
                self.particle_mode == LP_PARTICLE_HJ_FULL_CONSISTENT,
                dtype=jnp.float32,
            ),
            "particle_rng_routing_valid": (
                particle_train_info["particle_train_rng_routing_valid"]
            ),
            "particle_key_collision_rate": (
                particle_train_info["particle_train_key_collision_rate"]
            ),
            **particle_train_info,
            **proposal_span_info,
        }
        if self.proposal_mode == LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE:
            recovery_valid = (1.0 - feasible_float) * row_valid

            def branch_mean(values: Array, mask: Array) -> Array:
                return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)

            metrics.update(
                {
                    "posterior_likelihood_weight_mass_mean": branch_mean(
                        posterior.likelihood_weight_mass, row_valid
                    ),
                    "posterior_likelihood_component_ess_mean": branch_mean(
                        posterior.likelihood_component_ess, row_valid
                    ),
                    "posterior_reference_bridge_component_ess_mean": branch_mean(
                        posterior.reference_bridge_component_ess, row_valid
                    ),
                    "posterior_reference_bridge_weight_mass_mean": branch_mean(
                        posterior.reference_bridge_weight_mass, row_valid
                    ),
                    "posterior_reference_bridge_fraction_actual": (
                        posterior.reference_bridge_fraction_actual
                    ),
                    "posterior_likelihood_sample_count": (
                        posterior.likelihood_sample_count
                    ),
                    "posterior_reference_bridge_sample_count": (
                        posterior.reference_bridge_sample_count
                    ),
                    "posterior_likelihood_feasible_acceptance_mean": jnp.mean(
                        posterior.likelihood_feasible_acceptance
                    ),
                    "posterior_reference_bridge_feasible_acceptance_mean": jnp.mean(
                        posterior.reference_bridge_feasible_acceptance
                    ),
                    "reference_center_latent_norm_mean": jnp.mean(
                        posterior.reference_center_latent_norm_mean
                    ),
                    "posterior_ess_feasible_mean": branch_mean(
                        posterior.ess, feasible_valid
                    ),
                    "posterior_ess_recovery_mean": branch_mean(
                        posterior.ess, recovery_valid
                    ),
                    "posterior_reference_bridge_weight_mass_feasible_mean": branch_mean(
                        posterior.reference_bridge_weight_mass, feasible_valid
                    ),
                    "posterior_reference_bridge_weight_mass_recovery_mean": branch_mean(
                        posterior.reference_bridge_weight_mass, recovery_valid
                    ),
                    "posterior_reference_bridge_component_ess_feasible_mean": branch_mean(
                        posterior.reference_bridge_component_ess, feasible_valid
                    ),
                    "posterior_reference_bridge_component_ess_recovery_mean": branch_mean(
                        posterior.reference_bridge_component_ess, recovery_valid
                    ),
                }
            )
        for time_index in range(self.core.T):
            time_mask = (time == time_index).astype(jnp.float32)
            valid_time = time_mask * row_valid
            time_count = jnp.maximum(jnp.sum(time_mask), 1.0)
            metrics[f"posterior_sample_fraction_t{time_index}"] = jnp.mean(
                time_mask
            )
            metrics[f"posterior_sample_count_t{time_index}"] = jnp.sum(
                time_mask
            )
            metrics[f"posterior_all_zero_rate_t{time_index}"] = jnp.sum(
                time_mask * (1.0 - row_valid)
            ) / time_count
            metrics[f"posterior_ess_mean_t{time_index}"] = jnp.sum(
                posterior.ess * valid_time
            ) / jnp.maximum(jnp.sum(valid_time), 1.0)
            metrics[f"posterior_max_weight_mean_t{time_index}"] = jnp.sum(
                posterior.max_weight * valid_time
            ) / jnp.maximum(jnp.sum(valid_time), 1.0)
            if self.proposal_mode == LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE:
                metrics[
                    f"posterior_reference_bridge_weight_mass_mean_t{time_index}"
                ] = jnp.sum(
                    posterior.reference_bridge_weight_mass * valid_time
                ) / jnp.maximum(jnp.sum(valid_time), 1.0)
                metrics[
                    f"posterior_reference_bridge_component_ess_mean_t{time_index}"
                ] = jnp.sum(
                    posterior.reference_bridge_component_ess * valid_time
                ) / jnp.maximum(jnp.sum(valid_time), 1.0)

        return self.replace(
            core=core,
            reference_score_params=reference_params,
            actor_update_count=next_actor_update_count,
            reference_block_count=reference_block_count,
        ), metrics

    def update_critic_only(self, batch) -> tuple["LPPSAgent", dict[str, Array]]:
        """Call only legacy Qh/V/Q updates with LP bounded next actions.

        Passing ``next_policy_actions``/``next_actions`` is important: it
        prevents the legacy critic helpers from invoking their clipped actor
        sampler.  This method never calls ``core.update_actor`` or
        ``core.update`` and therefore cannot execute the historical dzfill
        actor path.
        """

        if self.core.qh_candidate_source != "policy_plus_gaussian":
            raise NotImplementedError(
                "LP critic update supports only Stage-B policy_plus_gaussian."
            )
        if self.particle_mode == LP_PARTICLE_MODE_S1_MIN:
            if self.reward_backup_mode == LP_REWARD_BACKUP_ONLINE_K1:
                next_actions, rng = self.policy_actions(
                    batch["next_observations"],
                    self.core.rng,
                    use_reference=False,
                )
                qh_base_actions = next_actions
            elif self.reward_backup_mode == LP_REWARD_BACKUP_REFERENCE_MEAN:
                next_actions, rng = self.policy_action_samples(
                    batch["next_observations"],
                    self.core.rng,
                    num_samples=self.reward_backup_samples,
                    use_reference=True,
                )
                if self.qh_backup_uses_reference:
                    # The HJ Stage-B set keeps exactly one policy centre plus
                    # seven Gaussian neighbours.  Its centre is the first
                    # independent reference action; the reward target uses
                    # the mean over all L actions.
                    qh_base_actions = next_actions[:, 0, :]
                else:
                    qh_base_actions, rng = self.policy_actions(
                        batch["next_observations"],
                        rng,
                        use_reference=False,
                    )
            else:
                raise ValueError(
                    f"Unsupported reward_backup_mode={self.reward_backup_mode!r}."
                )
            core = self.core.replace(rng=rng)
            core, qh_info = core.update_qh(
                batch, next_policy_actions=qh_base_actions
            )
            core, reward_info = core.update_q(
                batch, next_actions=next_actions
            )
            reward_info = {
                **reward_info,
                "reward_backup_reference_policy": jnp.asarray(
                    self.reward_backup_mode
                    == LP_REWARD_BACKUP_REFERENCE_MEAN,
                    dtype=jnp.float32,
                ),
                "qh_backup_reference_policy": jnp.asarray(
                    self.qh_backup_uses_reference,
                    dtype=jnp.float32,
                ),
            }
            target_particle_info = {
                "particle_target_active": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
                "particle_target_n": jnp.asarray(1.0, dtype=jnp.float32),
                "particle_target_ess_mean": jnp.asarray(
                    1.0, dtype=jnp.float32
                ),
                "particle_target_feasible_fraction_mean": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
                "particle_target_no_feasible_rate": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
                "particle_target_invalid_candidate_fraction_mean": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
                "particle_target_rng_routing_valid": jnp.asarray(
                    1.0, dtype=jnp.float32
                ),
                "particle_target_key_collision_rate": jnp.asarray(
                    0.0, dtype=jnp.float32
                ),
            }
        else:
            particles, rng = self.policy_particle_batch(
                batch["next_observations"],
                self.core.rng,
                num_particles=self.num_particles_target,
                target_ess_fraction=self.target_particle_ess_fraction,
                use_reference=False,
                use_target_critics=True,
            )
            core = self.core.replace(rng=rng)
            if self.particle_mode == LP_PARTICLE_HJ_FULL_CONSISTENT:
                core, qh_info = _update_qh_with_particle_candidates(
                    core, batch, particles.actions
                )
            else:
                # A fresh raw policy draw anchors an independent Gaussian
                # K=8 HJ set.  No selected/data/train particle enters Q_h.
                qh_base_actions, rng = self.policy_actions(
                    batch["next_observations"],
                    core.rng,
                    use_reference=False,
                )
                core = core.replace(rng=rng)
                core, qh_info = core.update_qh(
                    batch, next_policy_actions=qh_base_actions
                )
            core, reward_info = core.update_q(
                batch, next_actions=particles.selected_actions
            )
            distribution = particles.selection.distribution
            target_particle_info = {
                "particle_target_active": jnp.asarray(
                    1.0, dtype=jnp.float32
                ),
                "particle_target_n": jnp.asarray(
                    float(self.num_particles_target), dtype=jnp.float32
                ),
                "particle_target_ess_mean": jnp.mean(
                    distribution.categorical_ess
                ),
                "particle_target_feasible_fraction_mean": jnp.mean(
                    distribution.feasible_fraction
                ),
                "particle_target_no_feasible_rate": jnp.mean(
                    distribution.no_feasible.astype(jnp.float32)
                ),
                "particle_target_invalid_candidate_fraction_mean": jnp.mean(
                    distribution.invalid_candidate_fraction
                ),
                "particle_target_rng_routing_valid": (
                    particles.rng_routing_valid
                ),
                "particle_target_key_collision_rate": (
                    particles.key_collision_rate
                ),
            }
        # Stage-B is Qh-gated and has no state-only Vh update.  Do not call the
        # legacy disabled stub because it intentionally returns NaN vc_*
        # sentinels, which would pollute otherwise finite formal logs.
        value_info = {
            "safe_value_enabled": jnp.asarray(0.0, dtype=jnp.float32),
            "safe_value_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "vc_mean": jnp.asarray(0.0, dtype=jnp.float32),
            "vc_min": jnp.asarray(0.0, dtype=jnp.float32),
            "vc_max": jnp.asarray(0.0, dtype=jnp.float32),
        }
        return self.replace(core=core), {
            **qh_info,
            **value_info,
            **reward_info,
            **target_particle_info,
            "particle_hj_role_separated": jnp.asarray(
                self.particle_mode == LP_PARTICLE_HJ_ROLE_SEPARATED,
                dtype=jnp.float32,
            ),
            "particle_hj_full_consistent": jnp.asarray(
                self.particle_mode == LP_PARTICLE_HJ_FULL_CONSISTENT,
                dtype=jnp.float32,
            ),
            "particle_rng_routing_valid": (
                target_particle_info["particle_target_rng_routing_valid"]
            ),
            "particle_key_collision_rate": (
                target_particle_info["particle_target_key_collision_rate"]
            ),
            "legacy_dzfill_actor_used": jnp.asarray(0.0, dtype=jnp.float32),
        }

    def update(
        self, critic_batch, actor_batch=None
    ) -> tuple["LPPSAgent", dict[str, Array]]:
        """Optional convenience update; runner may supply independent batches."""

        agent, critic_info = self.update_critic_only(critic_batch)
        agent, actor_info = agent.update_actor(
            critic_batch if actor_batch is None else actor_batch
        )
        return agent, {
            **{f"critic/{key}": value for key, value in critic_info.items()},
            **{f"actor/{key}": value for key, value in actor_info.items()},
        }


__all__ = [
    "LPPSAgent",
    "LPPosteriorTarget",
    "LPBlendTarget",
    "LP_ACTOR_COORDINATE",
    "LP_TARGET_ESTIMATOR",
    "LP_PROPOSAL_MODE",
    "LP_BALANCE_MIS_PROPOSAL_MODE",
    "LP_DEFENSIVE_BRIDGE_MIS_PROPOSAL_MODE",
    "LP_LEGACY_AUTO_PROPOSAL_MODE",
    "LP_LEGACY_ACTOR_USED",
    "LP_ACTOR_SAFETY_QH_ROUTED",
    "LP_ACTOR_SAFETY_REWARD_ONLY",
    "LP_ACTOR_SAFETY_MODES",
    "LP_PARTICLE_MODE_S1_MIN",
    "LP_PARTICLE_HJ_ROLE_SEPARATED",
    "LP_PARTICLE_HJ_FULL_CONSISTENT",
    "LP_PARTICLE_MODES",
    "LP_REWARD_TILT_FIXED_ALPHA",
    "LP_REWARD_TILT_KL_BUDGET",
    "LP_REWARD_TILT_BATCH_KL",
    "LP_REWARD_TILT_CAPPED_BATCH_KL",
    "LP_REWARD_TILT_MODES",
    "PARTICLE_BEHAVIOR_INFO_KEYS",
    "LPPolicyParticleBatch",
    "KLBudgetRewardTilt",
    "actions_to_latent",
    "bounded_actions_to_latent",
    "bounded_latent_to_action",
    "stable_log_bounded_tanh_jacobian",
    "reference_anchored_epsilon_target",
    "stable_log_tanh_jacobian",
    "latent_ddpm_sampler",
    "latent_balance_mis_posterior_target",
    "latent_defensive_bridge_mis_posterior_target",
    "kl_budget_reward_tilt",
    "batch_kl_reward_tilt",
    "apply_reward_tilt_to_legacy_weights",
    "lp_proposal_mode",
    "make_lp_actor_train_state",
    "rebuild_actor_optimizer",
    "tree_global_norm",
]

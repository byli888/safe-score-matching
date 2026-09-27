"""
SSM online agent with Hamilton-Jacobi safety routing.

The single public implementation supports both safety estimators used in the
paper recipes:

- ``expectile``: train V_h with a reversed-expectile target.
- ``sampled_qh``: route with action-conditioned Q_h estimates. ``stage_a`` uses
  Q_h(s, pi(s)); ``stage_b`` uses sampled pessimism over local candidates.
"""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from ssm.common.gym_compat import gym
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import (
    MLP, StateActionValue, StateValue,
    DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler,
    vp_beta_schedule, get_weight_decay_mask,
)

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def safe_expectile_loss(diff, expectile=0.8):
    """Reversed expectile loss for V_h (approximates min_a Q_h).
    From FISOR fisor.py line 25-27."""
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def huber_loss(x, target, delta=1.0):
    """Huber loss: MSE for small errors, linear for large errors.
    Prevents catastrophic gradient amplification from outlier TD targets."""
    abs_diff = jnp.abs(x - target)
    return jnp.where(abs_diff <= delta,
                     0.5 * (x - target) ** 2,
                     delta * (abs_diff - 0.5 * delta))


def tensorstats(tensor, prefix=None):
    metrics = {
        'mean': tensor.mean(),
        'std': tensor.std(),
        'mag': jnp.abs(tensor).max(),
        'min': tensor.min(),
        'max': tensor.max(),
    }
    if prefix:
        metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
    return metrics


class SSMOnlineAgent(Agent):
    # ---- Actor (DDPM epsilon network) ----
    score_model: TrainState

    # ---- Reward critics (two separate networks, SAC-style) ----
    critic_1: TrainState
    critic_2: TrainState
    target_critic_1: TrainState
    target_critic_2: TrainState

    # ---- Safety critic Q_h (single network + EMA target) ----
    safe_critic: TrainState
    safe_target_critic: TrainState

    # ---- Safety value V_h (single network + EMA target) ----
    safe_value: TrainState
    safe_target_value: TrainState

    # ---- Hyperparameters ----
    discount: float
    discount_h: float
    tau: float
    M_q: float
    alpha_r: float
    beta_safety: float
    delta: float
    hard_margin: float
    cost_critic_hyperparam: float
    fuse_clip_min: float
    fuse_clip_max: float
    huber_delta: float  # 0 = MSE, >0 = Huber with this delta
    aux_coef: float
    lambda_safe: float
    safety_estimator: str = struct.field(pytree_node=False)
    vc_mode: str = struct.field(pytree_node=False)
    vc_policy_samples: int = struct.field(pytree_node=False)
    vc_proposal_samples: int = struct.field(pytree_node=False)
    vc_proposal_std: float
    stage_mode: str = struct.field(pytree_node=False)
    qh_candidate_source: str = struct.field(pytree_node=False)
    qh_min_k_samples: int = struct.field(pytree_node=False)
    qh_min_k_sigma: float

    # ---- DDPM schedule ----
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    clip_sampler: bool = struct.field(pytree_node=False)
    pure_qsm: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        # Architecture
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256),
        # Learning rates
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        safe_critic_lr: float = 3e-4,
        safe_value_lr: float = 3e-4,
        # Discount / EMA
        discount: float = 0.99,
        discount_h: float = 0.999,
        tau: float = 0.005,
        # SSM-specific
        M_q: float = 50.0,
        alpha_r: float = 1.0,
        beta_safety: float = 5.0,
        delta: float = 0.0,
        hard_margin: float = 0.0,
        cost_critic_hyperparam: float = 0.9,
        fuse_clip_min: float = -50.0,
        fuse_clip_max: float = 100.0,
        huber_delta: float = 10.0,
        aux_coef: float = 0.0,
        lambda_safe: float = 0.0,
        safety_estimator: str = "sampled_qh",
        vc_mode: str = "buffer_expectile",
        vc_policy_samples: int = 0,
        vc_proposal_samples: int = 0,
        vc_proposal_std: float = 0.6,
        stage_mode: str = "stage_a",
        qh_candidate_source: str = "policy_plus_gaussian",
        qh_min_k_samples: int = 8,
        qh_min_k_sigma: float = 0.3,
        # DDPM
        T: int = 5,
        time_dim: int = 64,
        clip_sampler: bool = True,
        ddpm_temperature: float = 1.0,
        beta_schedule: str = 'vp',
        # Misc
        decay_steps: Optional[int] = int(2e6),
        actor_weight_decay: Optional[float] = None,
        pure_qsm: bool = False,
    ):
        if safety_estimator not in ("expectile", "sampled_qh"):
            raise ValueError(
                f"Unsupported safety_estimator={safety_estimator!r}; "
                "expected 'expectile' or 'sampled_qh'."
            )
        if stage_mode not in ("stage_a", "stage_b"):
            raise ValueError(f"Unsupported stage_mode={stage_mode!r}; expected 'stage_a' or 'stage_b'.")
        if qh_candidate_source not in ("policy_only", "policy_plus_gaussian"):
            raise ValueError(
                f"Unsupported qh_candidate_source={qh_candidate_source!r}; "
                "expected 'policy_only' or 'policy_plus_gaussian'."
            )
        if qh_min_k_samples <= 0:
            raise ValueError(f"qh_min_k_samples must be positive, got {qh_min_k_samples}.")
        if qh_min_k_sigma < 0.0:
            raise ValueError(f"qh_min_k_sigma must be non-negative, got {qh_min_k_sigma}.")

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key_1, critic_key_2, safe_critic_key, safe_value_key = jax.random.split(rng, 6)

        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        # ==================== Actor (DDPM) ====================
        preprocess_time_cls = partial(
            FourierFeatures, output_size=time_dim, learnable=True)
        cond_model_cls = partial(
            MLP, hidden_dims=(128, 128), activations=mish, activate_final=False)

        if decay_steps is not None:
            actor_lr_schedule = optax.cosine_decay_schedule(actor_lr, decay_steps)
        else:
            actor_lr_schedule = actor_lr

        base_model_cls = partial(
            MLP,
            hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
            activations=mish,
            activate_final=False)

        actor_def = DDPM(
            time_preprocess_cls=preprocess_time_cls,
            cond_encoder_cls=cond_model_cls,
            reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        obs_init = jnp.expand_dims(observations, axis=0)
        act_init = jnp.expand_dims(actions, axis=0)
        actor_params = actor_def.init(actor_key, obs_init, act_init, time)['params']

        if actor_weight_decay is not None and actor_weight_decay > 0:
            actor_tx = optax.adamw(
                learning_rate=actor_lr_schedule,
                weight_decay=actor_weight_decay,
                mask=get_weight_decay_mask)
        else:
            actor_tx = optax.adam(learning_rate=actor_lr_schedule)

        score_model = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params, tx=actor_tx)

        # ==================== Reward Critics (two separate, SAC-style) ====================
        critic_base_cls = partial(
            MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_def = StateActionValue(critic_base_cls)

        critic_params_1 = critic_def.init(critic_key_1, obs_init, act_init)["params"]
        critic_params_2 = critic_def.init(critic_key_2, obs_init, act_init)["params"]

        critic_1 = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params_1,
            tx=optax.adam(learning_rate=critic_lr))
        critic_2 = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params_2,
            tx=optax.adam(learning_rate=critic_lr))

        no_op_tx = optax.GradientTransformation(lambda _: None, lambda _: None)
        target_critic_1 = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params_1, tx=no_op_tx)
        target_critic_2 = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params_2, tx=no_op_tx)

        # ==================== Safety Critic Q_h (single network) ====================
        safe_critic_def = StateActionValue(critic_base_cls)
        safe_critic_params = safe_critic_def.init(safe_critic_key, obs_init, act_init)["params"]

        safe_critic = TrainState.create(
            apply_fn=safe_critic_def.apply, params=safe_critic_params,
            tx=optax.adam(learning_rate=safe_critic_lr))
        safe_target_critic = TrainState.create(
            apply_fn=safe_critic_def.apply, params=safe_critic_params, tx=no_op_tx)

        # ==================== Safety Value V_h + EMA target ====================
        value_base_cls = partial(
            MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        safe_value_def = StateValue(base_cls=value_base_cls)
        safe_value_params = safe_value_def.init(safe_value_key, obs_init)["params"]

        safe_value = TrainState.create(
            apply_fn=safe_value_def.apply, params=safe_value_params,
            tx=optax.adam(learning_rate=safe_value_lr))
        safe_target_value = TrainState.create(
            apply_fn=safe_value_def.apply, params=safe_value_params, tx=no_op_tx)

        # ==================== DDPM Schedule ====================
        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas_arr = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas_arr[:i + 1]) for i in range(T)])

        return cls(
            actor=None,
            rng=rng,
            score_model=score_model,
            critic_1=critic_1,
            critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            safe_target_value=safe_target_value,
            discount=discount,
            discount_h=discount_h,
            tau=tau,
            M_q=M_q,
            alpha_r=alpha_r,
            beta_safety=beta_safety,
            delta=delta,
            hard_margin=hard_margin,
            cost_critic_hyperparam=cost_critic_hyperparam,
            fuse_clip_min=fuse_clip_min,
            fuse_clip_max=fuse_clip_max,
            # huber_delta=0 means MSE: use huge delta so Huber degenerates to quadratic
            huber_delta=huber_delta if huber_delta > 0 else 1e10,
            aux_coef=aux_coef,
            lambda_safe=lambda_safe,
            safety_estimator=safety_estimator,
            vc_mode=vc_mode,
            vc_policy_samples=vc_policy_samples,
            vc_proposal_samples=vc_proposal_samples,
            vc_proposal_std=vc_proposal_std,
            stage_mode=stage_mode,
            qh_candidate_source=qh_candidate_source,
            qh_min_k_samples=qh_min_k_samples,
            qh_min_k_sigma=qh_min_k_sigma,
            act_dim=action_dim,
            T=T,
            clip_sampler=clip_sampler,
            pure_qsm=pure_qsm,
            ddpm_temperature=ddpm_temperature,
            betas=betas,
            alphas=alphas_arr,
            alpha_hats=alpha_hat,
        )

    @staticmethod
    def _sample_vc_policy_actions(agent, observations: jnp.ndarray, num_samples: int, rng):
        if num_samples <= 0:
            return jnp.zeros((observations.shape[0], 0, agent.act_dim), dtype=jnp.float32), rng
        obs_rep = jnp.repeat(observations, repeats=num_samples, axis=0)
        actions_flat, rng = ddpm_sampler(
            agent.score_model.apply_fn,
            agent.score_model.params,
            agent.T,
            rng,
            agent.act_dim,
            obs_rep,
            agent.alphas,
            agent.alpha_hats,
            agent.betas,
            agent.ddpm_temperature,
            0,
            agent.clip_sampler,
        )
        return actions_flat.reshape(observations.shape[0], num_samples, agent.act_dim), rng

    @staticmethod
    def _sample_vc_gaussian_actions(agent, center_actions: jnp.ndarray, num_samples: int, proposal_std: float, rng):
        if num_samples <= 0:
            return jnp.zeros((center_actions.shape[0], 0, agent.act_dim), dtype=jnp.float32), rng
        key, rng = jax.random.split(rng)
        noise = jax.random.normal(
            key,
            shape=(center_actions.shape[0], num_samples, agent.act_dim),
            dtype=jnp.float32,
        )
        actions = center_actions[:, None, :] + proposal_std * noise
        return jnp.clip(actions, -1.0, 1.0), rng

    @staticmethod
    def _evaluate_safe_critic_candidates(agent, observations: jnp.ndarray, actions: jnp.ndarray):
        if actions.shape[1] == 0:
            return jnp.zeros((observations.shape[0], 0), dtype=jnp.float32)
        obs_rep = jnp.repeat(observations, repeats=actions.shape[1], axis=0)
        actions_flat = actions.reshape(-1, agent.act_dim)
        q_flat = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params},
            obs_rep,
            actions_flat,
        )
        return q_flat.reshape(observations.shape[0], actions.shape[1])

    @staticmethod
    def _evaluate_safe_critic_candidates_with_params(
        agent,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        params,
        *,
        use_target: bool,
    ):
        if actions.shape[1] == 0:
            return jnp.zeros((observations.shape[0], 0), dtype=jnp.float32)
        obs_rep = jnp.repeat(observations, repeats=actions.shape[1], axis=0)
        actions_flat = actions.reshape(-1, agent.act_dim)
        critic_apply_fn = agent.safe_target_critic.apply_fn if use_target else agent.safe_critic.apply_fn
        q_flat = critic_apply_fn({"params": params}, obs_rep, actions_flat)
        return q_flat.reshape(observations.shape[0], actions.shape[1])

    @staticmethod
    def _sample_qh_candidates(
        agent,
        observations: jnp.ndarray,
        rng,
        *,
        base_policy_actions: Optional[jnp.ndarray] = None,
    ):
        if base_policy_actions is not None and base_policy_actions.ndim != 2:
            raise ValueError(
                f"base_policy_actions must have shape [B, act_dim], got {base_policy_actions.shape!r}."
            )

        if agent.stage_mode == "stage_a":
            if base_policy_actions is not None:
                return base_policy_actions[:, None, :], rng
            return SSMOnlineAgent._sample_vc_policy_actions(agent, observations, 1, rng)

        if agent.stage_mode != "stage_b":
            raise ValueError(f"Unsupported stage_mode={agent.stage_mode!r}")

        num_samples = max(int(agent.qh_min_k_samples), 1)
        if agent.qh_candidate_source == "policy_only":
            if base_policy_actions is None:
                return SSMOnlineAgent._sample_vc_policy_actions(agent, observations, num_samples, rng)
            if num_samples <= 1:
                return base_policy_actions[:, None, :], rng
            extra_actions, rng = SSMOnlineAgent._sample_vc_policy_actions(
                agent, observations, num_samples - 1, rng
            )
            base = base_policy_actions[:, None, :]
            return jnp.concatenate([base, extra_actions], axis=1), rng

        if agent.qh_candidate_source == "policy_plus_gaussian":
            if base_policy_actions is not None:
                policy_actions = base_policy_actions[:, None, :]
            else:
                policy_actions, rng = SSMOnlineAgent._sample_vc_policy_actions(agent, observations, 1, rng)
            if num_samples <= 1:
                return policy_actions, rng
            gaussian_actions, rng = SSMOnlineAgent._sample_vc_gaussian_actions(
                agent,
                policy_actions[:, 0, :],
                num_samples - 1,
                agent.qh_min_k_sigma,
                rng,
            )
            return jnp.concatenate([policy_actions, gaussian_actions], axis=1), rng

        raise ValueError(f"Unsupported qh_candidate_source={agent.qh_candidate_source!r}")

    @staticmethod
    def _reduce_qh_candidates(agent, q_candidates: jnp.ndarray):
        if q_candidates.shape[1] == 0:
            raise ValueError("Q_h candidate reduction received zero candidates.")
        if agent.stage_mode == "stage_a":
            return q_candidates[:, 0]
        if agent.stage_mode == "stage_b":
            return q_candidates.min(axis=1)
        raise ValueError(f"Unsupported stage_mode={agent.stage_mode!r}")

    @staticmethod
    def _estimate_qh_from_candidates(
        agent,
        observations: jnp.ndarray,
        rng,
        *,
        params,
        use_target: bool,
        base_policy_actions: Optional[jnp.ndarray] = None,
    ):
        candidate_actions, rng = SSMOnlineAgent._sample_qh_candidates(
            agent,
            observations,
            rng,
            base_policy_actions=base_policy_actions,
        )
        q_candidates = SSMOnlineAgent._evaluate_safe_critic_candidates_with_params(
            agent,
            observations,
            candidate_actions,
            params,
            use_target=use_target,
        )
        q_est = SSMOnlineAgent._reduce_qh_candidates(agent, q_candidates)
        candidate_count = jnp.asarray(candidate_actions.shape[1], dtype=jnp.float32)
        return q_est, candidate_count, rng

    # ==================================================================
    # update_q: SAC-style double-Q TD for reward critic
    # ==================================================================
    def update_q(
        agent,
        batch: DatasetDict,
        *,
        next_actions: Optional[jnp.ndarray] = None,
    ) -> Tuple[Agent, Dict[str, float]]:

        # Sample next actions from current policy
        if next_actions is None:
            key, rng = jax.random.split(agent.rng)
            next_actions, rng = ddpm_sampler(
                agent.score_model.apply_fn,
                agent.score_model.params,
                agent.T, rng, agent.act_dim,
                batch['next_observations'],
                agent.alphas, agent.alpha_hats,
                agent.betas, agent.ddpm_temperature,
                0,  # repeat_last_step
                agent.clip_sampler)
        else:
            rng = agent.rng

        # No target smoothing noise

        # Target Q = min(Q1', Q2')
        next_q_1 = agent.target_critic_1.apply_fn(
            {"params": agent.target_critic_1.params},
            batch["next_observations"], next_actions)
        next_q_2 = agent.target_critic_2.apply_fn(
            {"params": agent.target_critic_2.params},
            batch["next_observations"], next_actions)
        next_v = jnp.minimum(next_q_1, next_q_2)
        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        # Clip the bootstrap target to the configured bounds.
        target_q_unclipped = target_q
        target_q = jnp.clip(target_q, agent.fuse_clip_min, agent.fuse_clip_max)
        fuse_clip_frac = (target_q != target_q_unclipped).mean()
        fuse_clip_low_frac = (target_q_unclipped < agent.fuse_clip_min).mean()

        target_q = sg(target_q)

        metrics = {
            **tensorstats(target_q, 'target_q'),
            'fuse_clip_frac': fuse_clip_frac,
            'fuse_clip_low_frac': fuse_clip_low_frac,
        }

        def critic_loss_fn_1(params):
            q = agent.critic_1.apply_fn({"params": params},
                                        batch["observations"], batch["actions"])
            abs_td = jnp.abs(q - target_q)
            loss = huber_loss(q, target_q, delta=agent.huber_delta).mean()
            return loss, {
                "critic_loss_1": loss,
                "q1_mean": q.mean(),
                "q1_min": q.min(),
                "q1_max": q.max(),
                "td_err_abs_mean_1": abs_td.mean(),
                "td_err_abs_p99_1": jnp.quantile(abs_td, 0.99),
                "huber_linear_frac_1": (abs_td > agent.huber_delta).mean(),
            }

        grads_1, info_1 = jax.grad(critic_loss_fn_1, has_aux=True)(agent.critic_1.params)
        critic_1 = agent.critic_1.apply_gradients(grads=grads_1)

        def critic_loss_fn_2(params):
            q = agent.critic_2.apply_fn({"params": params},
                                        batch["observations"], batch["actions"])
            abs_td = jnp.abs(q - target_q)
            loss = huber_loss(q, target_q, delta=agent.huber_delta).mean()
            return loss, {
                "critic_loss_2": loss,
                "q2_mean": q.mean(),
                "q2_min": q.min(),
                "q2_max": q.max(),
                "td_err_abs_mean_2": abs_td.mean(),
                "td_err_abs_p99_2": jnp.quantile(abs_td, 0.99),
                "huber_linear_frac_2": (abs_td > agent.huber_delta).mean(),
            }

        grads_2, info_2 = jax.grad(critic_loss_fn_2, has_aux=True)(agent.critic_2.params)
        critic_2 = agent.critic_2.apply_gradients(grads=grads_2)

        # EMA target update
        tc1_params = optax.incremental_update(critic_1.params, agent.target_critic_1.params, agent.tau)
        tc2_params = optax.incremental_update(critic_2.params, agent.target_critic_2.params, agent.tau)
        target_critic_1 = agent.target_critic_1.replace(params=tc1_params)
        target_critic_2 = agent.target_critic_2.replace(params=tc2_params)

        new_agent = agent.replace(
            critic_1=critic_1, critic_2=critic_2,
            target_critic_1=target_critic_1, target_critic_2=target_critic_2,
            rng=rng)
        td_metrics = {
            "td_err_abs_mean": 0.5 * (info_1["td_err_abs_mean_1"] + info_2["td_err_abs_mean_2"]),
            "td_err_abs_p99": jnp.maximum(info_1["td_err_abs_p99_1"], info_2["td_err_abs_p99_2"]),
            "huber_linear_frac": 0.5 * (
                info_1["huber_linear_frac_1"] + info_2["huber_linear_frac_2"]
            ),
        }
        return new_agent, {**metrics, **info_1, **info_2, **td_metrics}

    # ==================================================================
    # update_qc: HJ max-Bellman for Q_h.
    #   expectile: bootstrap with TARGET V_h(s')
    #   stage_a: bootstrap with Q_h'(s', pi(s'))
    #   stage_b: bootstrap with min_k Q_h'(s', a'_k)
    # ==================================================================
    def update_qc(
        agent,
        batch: DatasetDict,
        *,
        next_policy_actions: Optional[jnp.ndarray] = None,
    ) -> Tuple[Agent, Dict[str, float]]:
        if agent.safety_estimator == "expectile":
            next_safe = agent.safe_target_value.apply_fn(
                {"params": agent.safe_target_value.params}, batch["next_observations"])
            candidate_count = jnp.asarray(0.0, dtype=jnp.float32)
            rng = agent.rng
        else:
            key, rng = jax.random.split(agent.rng)
            next_safe, candidate_count, rng = SSMOnlineAgent._estimate_qh_from_candidates(
                agent,
                batch["next_observations"],
                key,
                params=agent.safe_target_critic.params,
                use_target=True,
                base_policy_actions=next_policy_actions,
            )

        # HJ Bellman: y = (1-γh)h + γh max(h, safe_estimate(s'))
        qc_nonterminal = (
            (1.0 - agent.discount_h) * batch["costs"]
            + agent.discount_h * jnp.maximum(batch["costs"], next_safe)
        )
        target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])

        def safe_critic_loss_fn(params):
            qc = agent.safe_critic.apply_fn(
                {"params": params}, batch["observations"], batch["actions"])
            loss = ((qc - target_qc) ** 2).mean()
            return loss, {
                "safe_critic_loss": loss,
                "qc_mean": qc.mean(),
                "qc_max": qc.max(),
                "qc_min": qc.min(),
                "costs_mean": batch["costs"].mean(),
                "safe_bootstrap_mean": next_safe.mean(),
                "vh_bootstrap_mean": next_safe.mean(),
                "qh_bootstrap_mean": next_safe.mean(),
                "qh_bootstrap_candidate_count": candidate_count,
                "safe_bootstrap_candidate_count": candidate_count,
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)

        # EMA target update for Q_h
        stc_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau)
        safe_target_critic = agent.safe_target_critic.replace(params=stc_params)

        new_agent = agent.replace(
            safe_critic=safe_critic, safe_target_critic=safe_target_critic, rng=rng)
        return new_agent, info

    # ==================================================================
    # update_vc: expectile V_h estimator for recipes that use state-value routing
    # ==================================================================
    def update_vc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        if agent.safety_estimator != "expectile":
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
            return agent, {
                "safe_value_loss": zero,
                "vc_mean": nan,
                "vc_min": nan,
                "vc_max": nan,
            }

        qc_buffer = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params},
            batch["observations"], batch["actions"])
        qc_candidates = [qc_buffer[:, None]]
        rng = agent.rng

        if agent.vc_mode != "buffer_expectile":
            policy_actions = None
            if agent.vc_policy_samples > 0:
                policy_actions, rng = SSMOnlineAgent._sample_vc_policy_actions(
                    agent,
                    batch["observations"],
                    agent.vc_policy_samples,
                    rng,
                )
                qc_candidates.append(
                    SSMOnlineAgent._evaluate_safe_critic_candidates(
                        agent, batch["observations"], policy_actions
                    )
                )
            if agent.vc_proposal_samples > 0:
                proposal_centers = batch["actions"] if policy_actions is None else policy_actions[:, 0, :]
                proposal_actions, rng = SSMOnlineAgent._sample_vc_gaussian_actions(
                    agent,
                    proposal_centers,
                    agent.vc_proposal_samples,
                    agent.vc_proposal_std,
                    rng,
                )
                qc_candidates.append(
                    SSMOnlineAgent._evaluate_safe_critic_candidates(
                        agent, batch["observations"], proposal_actions
                    )
                )

        qc_all = jnp.concatenate(qc_candidates, axis=1)
        vc_target = qc_all.min(axis=1)

        def safe_value_loss_fn(params):
            vc = agent.safe_value.apply_fn({"params": params}, batch["observations"])
            if agent.vc_mode == "buffer_expectile":
                loss = safe_expectile_loss(qc_buffer - vc, agent.cost_critic_hyperparam).mean()
            elif agent.vc_mode == "hybrid_expectile":
                loss = safe_expectile_loss(
                    qc_all - vc[:, None], agent.cost_critic_hyperparam
                ).mean()
            elif agent.vc_mode == "hybrid_min":
                loss = ((vc_target - vc) ** 2).mean()
            else:
                raise ValueError(f"Unsupported vc_mode={agent.vc_mode!r}")
            return loss, {
                "safe_value_loss": loss,
                "vc_mean": vc.mean(),
                "vc_max": vc.max(),
                "vc_min": vc.min(),
                "vc_target_mean": vc_target.mean(),
                "vc_candidate_q_mean": qc_all.mean(),
                "vc_candidate_q_min_mean": qc_all.min(axis=1).mean(),
                "vc_candidate_q_max_mean": qc_all.max(axis=1).mean(),
                "vc_candidate_count": jnp.asarray(qc_all.shape[1], dtype=jnp.float32),
            }

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)

        stv_params = optax.incremental_update(
            safe_value.params, agent.safe_target_value.params, agent.tau)
        safe_target_value = agent.safe_target_value.replace(params=stv_params)

        new_agent = agent.replace(
            safe_value=safe_value, safe_target_value=safe_target_value, rng=rng)
        return new_agent, info

    # ==================================================================
    # update_actor: routed critic-gradient score targets
    # ==================================================================
    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        B = batch['actions'].shape[0]
        A = agent.act_dim

        # ---- 1. Forward diffusion process ----
        key, rng = jax.random.split(agent.rng, 2)
        time = jax.random.randint(key, (B,), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(key, (B, A))
        key, rng = jax.random.split(rng, 2)

        alpha_hats = agent.alpha_hats[time]
        time_input = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        # ---- 2. ∇_a Q_r: differentiate min(Q1, Q2) ----
        critic_jacobian = jax.grad(
            lambda a: jnp.minimum(
                agent.critic_1.apply_fn(
                    {"params": agent.critic_1.params},
                    batch['observations'], a),
                agent.critic_2.apply_fn(
                    {"params": agent.critic_2.params},
                    batch['observations'], a),
            ).sum()
        )(noisy_actions)

        # Always-normalize: keep direction, set magnitude to 1.0
        raw_grad_qr_norm = jnp.linalg.norm(critic_jacobian, axis=-1, keepdims=True)
        critic_jacobian = critic_jacobian / (raw_grad_qr_norm + 1e-8)
        clipped_grad_qr_norm = jnp.linalg.norm(critic_jacobian, axis=-1)
        clip_frac = (raw_grad_qr_norm.squeeze(-1) > 1.0).mean()

        # ---- 3. ∇_a Q_h: single network, just differentiate ----
        safe_jacobian = jax.grad(
            lambda a: agent.safe_critic.apply_fn(
                {"params": agent.safe_critic.params},
                batch['observations'], a).sum()
        )(noisy_actions)

        # Normalized version of ∇Q_h — used ONLY for dead zone fill term
        raw_grad_qh_norm = jnp.linalg.norm(safe_jacobian, axis=-1, keepdims=True)
        safe_jacobian_normalized = safe_jacobian / (raw_grad_qh_norm + 1e-8)
        grad_cos_all = jnp.sum(critic_jacobian * safe_jacobian_normalized, axis=-1)

        # ---- 4. Indicator: 1_{Q_h(s,a) ≤ 0} ----
        qh_val = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params},
            batch['observations'], noisy_actions)
        indicator = jnp.where(qh_val <= 0, 1.0, 0.0)

        # ---- 5. State routing with hard margin ----
        safe_threshold = agent.delta - agent.hard_margin
        if agent.safety_estimator == "expectile":
            vc = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params}, batch["observations"])
            gate_qh = jnp.full_like(vc, jnp.nan)
            gate_candidate_count = jnp.asarray(0.0, dtype=jnp.float32)
            is_safe = jnp.where(vc <= safe_threshold, 1.0, 0.0)
        else:
            gate_qh, gate_candidate_count, rng = SSMOnlineAgent._estimate_qh_from_candidates(
                agent,
                batch["observations"],
                rng,
                params=agent.safe_critic.params,
                use_target=False,
            )
            vc = jnp.full_like(gate_qh, jnp.nan)
            is_safe = jnp.where(gate_qh <= safe_threshold, 1.0, 0.0)

        # ---- Pure QSM ablation: bypass all safety gates ----
        if agent.pure_qsm:
            indicator = jnp.ones_like(indicator)
            is_safe = jnp.ones_like(is_safe)

        reward_coverage = (is_safe * indicator).mean()
        safe_frac = is_safe.mean()
        deadzone_mask = is_safe * (1.0 - indicator)
        deadzone_count = deadzone_mask.sum()
        grad_cos_deadzone = jnp.where(
            deadzone_count > 0,
            jnp.sum(grad_cos_all * deadzone_mask) / deadzone_count,
            jnp.nan,
        )

        # ---- 6. SSM Eq 12: piecewise score target + dead zone fill ----
        # Safe state + safe action (indicator=1): reward guidance
        # Safe state + dangerous action (indicator=0): safety recovery push (dead zone fill)
        # Unsafe state: safety recovery (unchanged)
        safe_score = (agent.alpha_r * critic_jacobian * sg(indicator[:, None])
                    + (-agent.alpha_r * safe_jacobian_normalized) * (1.0 - sg(indicator[:, None])))
        unsafe_score = -agent.beta_safety * safe_jacobian
        score_target = (
            sg(is_safe[:, None]) * safe_score
            + (1.0 - sg(is_safe[:, None])) * unsafe_score
        )

        # ---- 7. Epsilon target (NEGATIVE sign) ----
        eps_target = -agent.M_q * sg(score_target)
        safe_mask = sg(is_safe[:, None])
        unsafe_mask = 1.0 - safe_mask
        unsafe_energy = jnp.sum((eps_target * unsafe_mask) ** 2)
        total_energy = jnp.sum(eps_target ** 2) + 1e-8
        unsafe_eps_norm = jnp.linalg.norm(eps_target * unsafe_mask, axis=-1)
        safe_eps_norm = jnp.linalg.norm(eps_target * safe_mask, axis=-1)

        # ---- 8. Score matching loss ----
        def actor_loss_fn(score_model_params):
            eps_pred = agent.score_model.apply_fn(
                {'params': score_model_params},
                batch['observations'], noisy_actions, time_input,
                rngs={'dropout': key}, training=True)

            actor_loss = jnp.power(eps_pred - eps_target, 2).mean(-1)

            metrics = {
                'actor_loss': actor_loss.mean(),
                'safe_frac': safe_frac,
                'indicator_frac': indicator.mean(),
                'reward_coverage': reward_coverage,
                'indicator_given_safe': reward_coverage / jnp.maximum(safe_frac, 1e-6),
                'deadzone_frac': deadzone_mask.mean(),
                'grad_qr_norm_raw': raw_grad_qr_norm.squeeze(-1).mean(),
                'grad_qr_norm_clipped': clipped_grad_qr_norm.mean(),
                'grad_qr_clip_frac': clip_frac,
                'grad_qh_norm': jnp.linalg.norm(safe_jacobian, axis=-1).mean(),
                'unsafe_score_norm_mean': jnp.linalg.norm(unsafe_score, axis=-1).mean(),
                'grad_cos_qr_qh_deadzone_mean': grad_cos_deadzone,
                'grad_cos_qr_qh_all_mean': grad_cos_all.mean(),
                'qh_indicator_mean': qh_val.mean(),
                'qh_policy_mean': gate_qh.mean(),
                'qh_gate_mean': gate_qh.mean(),
                'qh_gate_candidate_count': gate_candidate_count,
                'vh_mean': vc.mean(),
                'safe_threshold': safe_threshold,
                'hard_margin': agent.hard_margin,
                'eps_target_norm': jnp.linalg.norm(eps_target, axis=-1).mean(),
                'eps_pred_norm': jnp.linalg.norm(eps_pred, axis=-1).mean(),
                'actor_unsafe_energy_frac': unsafe_energy / total_energy,
                'eps_target_unsafe_norm_mean': unsafe_eps_norm.mean(),
                'eps_target_safe_norm_mean': safe_eps_norm.mean(),
            }
            return actor_loss.mean(), metrics

        grads, metrics = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        new_agent = agent.replace(score_model=score_model, rng=rng)
        return new_agent, metrics

    # ==================================================================
    # Action sampling / evaluation
    # ==================================================================
    @jax.jit
    def sample_actions(self, observations: jnp.ndarray):
        """Sample actions with exploration noise for training."""
        actions, new_agent = self.eval_actions(observations)
        key, rng = jax.random.split(new_agent.rng, 2)
        noise = jax.random.normal(key, shape=actions.shape) * 0.1
        actions = jnp.clip(actions + noise, -1.0, 1.0)
        return actions, new_agent.replace(rng=rng)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        """Draw one DDPM action for evaluation and advance the RNG."""
        rng = self.rng
        assert len(observations.shape) == 1
        observations = observations[None]

        actions, rng = ddpm_sampler(
            self.score_model.apply_fn,
            self.score_model.params,
            self.T, rng, self.act_dim, observations,
            self.alphas, self.alpha_hats,
            self.betas, self.ddpm_temperature,
            0,  # repeat_last_step
            self.clip_sampler)
        assert actions.shape == (1, self.act_dim)
        _, rng = jax.random.split(rng, 2)
        return jnp.squeeze(actions), self.replace(rng=rng)

    # ==================================================================
    # Full update step (with actor)
    # ==================================================================
    @jax.jit
    def update(self, batch: DatasetDict):
        if self.safety_estimator == "expectile":
            new_agent = self
            new_agent, qc_info = new_agent.update_qc(batch)
            new_agent, vc_info = new_agent.update_vc(batch)
            new_agent, q_info = new_agent.update_q(batch)
            new_agent, actor_info = new_agent.update_actor(batch)
            return new_agent, {**qc_info, **vc_info, **q_info, **actor_info}

        key, rng = jax.random.split(self.rng)
        shared_next_actions, rng = SSMOnlineAgent._sample_vc_policy_actions(
            self,
            batch["next_observations"],
            1,
            key,
        )
        shared_next_actions = shared_next_actions[:, 0, :]

        new_agent = self.replace(rng=rng)
        new_agent, qc_info = new_agent.update_qc(batch, next_policy_actions=shared_next_actions)
        new_agent, vc_info = new_agent.update_vc(batch)
        new_agent, q_info = new_agent.update_q(batch, next_actions=shared_next_actions)
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, {**qc_info, **vc_info, **q_info, **actor_info}

    # ==================================================================
    # Critic-only update step (delayed actor: skip actor update)
    # ==================================================================
    @jax.jit
    def update_critic_only(self, batch: DatasetDict):
        if self.safety_estimator == "expectile":
            new_agent = self
            new_agent, qc_info = new_agent.update_qc(batch)
            new_agent, vc_info = new_agent.update_vc(batch)
            new_agent, q_info = new_agent.update_q(batch)
            return new_agent, {**qc_info, **vc_info, **q_info}

        key, rng = jax.random.split(self.rng)
        shared_next_actions, rng = SSMOnlineAgent._sample_vc_policy_actions(
            self,
            batch["next_observations"],
            1,
            key,
        )
        shared_next_actions = shared_next_actions[:, 0, :]

        new_agent = self.replace(rng=rng)
        new_agent, qc_info = new_agent.update_qc(batch, next_policy_actions=shared_next_actions)
        new_agent, vc_info = new_agent.update_vc(batch)
        new_agent, q_info = new_agent.update_q(batch, next_actions=shared_next_actions)
        return new_agent, {**qc_info, **vc_info, **q_info}

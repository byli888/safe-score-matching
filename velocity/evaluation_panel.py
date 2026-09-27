"""Reproducible multi-environment evaluation for safety policies.

The evaluator deliberately requests exactly one stochastic action sample for
each active environment and decision.  It never ranks candidates at test time.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Any, Callable, Mapping, Protocol

import gymnasium as gym
import numpy as np


MIN_SCIENTIFIC_EPISODES = 50


class StochasticP1Actor(Protocol):
    """Batched stochastic policy: one action per observation and sample seed."""

    def __call__(
        self, observations: np.ndarray, sample_seeds: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class EvaluationProtocol:
    env_id: str
    layout_mode: str
    num_episodes: int = MIN_SCIENTIFIC_EPISODES
    min_episodes: int = MIN_SCIENTIFIC_EPISODES
    num_workers: int = 8
    vector_backend: str = "async"
    max_episode_steps: int = 1_000
    evaluation_seed: int = 0
    fixed_layout_seed: int = 10_000
    actor_seed: int = 20_000
    bootstrap_seed: int = 30_000
    bootstrap_samples: int = 10_000
    confidence: float = 0.95

    def validate(self) -> None:
        if self.layout_mode not in ("fixed", "random"):
            raise ValueError("layout_mode must be 'fixed' or 'random'")
        if self.vector_backend not in ("sync", "async"):
            raise ValueError("vector_backend must be 'sync' or 'async'")
        if self.num_episodes < self.min_episodes:
            raise ValueError(
                f"scientific checkpoint evaluation requires at least "
                f"{self.min_episodes} episodes"
            )
        for name in ("num_workers", "max_episode_steps", "bootstrap_samples"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one")
        for name in (
            "evaluation_seed",
            "fixed_layout_seed",
            "actor_seed",
            "bootstrap_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")


@dataclass(frozen=True)
class EvaluationResult:
    protocol: Mapping[str, Any]
    episodes: Mapping[str, np.ndarray]
    summary: Mapping[str, Mapping[str, Any]]

    def json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": dict(self.protocol),
            "episodes": {
                name: np.asarray(value).tolist()
                for name, value in self.episodes.items()
            },
            "summary": self.summary,
        }


def _seed_stream(seed: int, count: int, stream: int) -> np.ndarray:
    """Stable, explicit uint32 seeds independent of worker batching."""
    sequence = np.random.SeedSequence([int(seed), int(stream)])
    return sequence.generate_state(count, dtype=np.uint32)


def _action_seed(policy_seed: np.ndarray, step: int) -> np.ndarray:
    # SplitMix-style integer mixing.  uint64 overflow is intentional, and the
    # low 32 bits are a portable seed consumed by actor adapters.
    policy_seed = np.asarray(policy_seed, dtype=np.uint64)
    with np.errstate(over="ignore"):
        value = (
            policy_seed
            + np.uint64(0xBF58476D1CE4E5B9) * np.uint64(step + 1)
        )
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return (value & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def _flat_observations(observations: Any) -> np.ndarray:
    if isinstance(observations, Mapping):
        if "observation" not in observations:
            raise TypeError("dict observations must contain an 'observation' field")
        observations = observations["observation"]
    values = np.asarray(observations, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("vector observations must include an environment axis")
    return values.reshape((values.shape[0], -1))


def _single_info(infos: Mapping[str, Any], index: int, terminal: bool) -> dict:
    if terminal and "final_info" in infos:
        mask = np.asarray(infos.get("_final_info", True))
        present = bool(mask if mask.ndim == 0 else mask[index])
        if present:
            candidate = np.asarray(infos["final_info"], dtype=object)
            value = candidate.item() if candidate.ndim == 0 else candidate[index]
            if isinstance(value, Mapping):
                return dict(value)

    result = {}
    for name, values in infos.items():
        if name.startswith("_") or name in ("final_info", "final_observation"):
            continue
        mask_name = f"_{name}"
        if mask_name in infos:
            mask = np.asarray(infos[mask_name])
            if not bool(mask if mask.ndim == 0 else mask[index]):
                continue
        array = np.asarray(values, dtype=object)
        result[name] = array.item() if array.ndim == 0 else array[index]
    return result


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
    confidence: float,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    means = np.empty((samples,), dtype=np.float64)
    # Chunking avoids a large [bootstrap_samples, episodes] allocation.
    for start in range(0, samples, 2_048):
        stop = min(start + 2_048, samples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = np.mean(values[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    ]


def summarize_episodes(
    episodes: Mapping[str, np.ndarray],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence: float,
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(bootstrap_seed)
    summary = {}
    for name in (
        "reward",
        "cost",
        "any_violation",
        "violation_count",
        "length",
        "success",
    ):
        values = np.asarray(episodes[name], dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            f"mean_ci{int(round(confidence * 100))}": _bootstrap_mean_ci(
                values,
                rng=rng,
                samples=bootstrap_samples,
                confidence=confidence,
            ),
        }
    return summary


def evaluate_stochastic_policy(
    env_factory: Callable[[int, str], gym.Env],
    actor: StochasticP1Actor,
    protocol: EvaluationProtocol,
) -> EvaluationResult:
    """Evaluate a stochastic policy using P=1 and bounded vector-env pools.

    Every vector environment is used for exactly one scored episode.  Thus a
    random-layout episode has an explicit independent reset seed, while all
    fixed-layout episodes reuse ``fixed_layout_seed``.  Autoreset transitions
    after an environment finishes are stepped only as vector placeholders and
    are never scored.
    """
    protocol.validate()
    random_layout_seeds = _seed_stream(
        protocol.evaluation_seed, protocol.num_episodes, stream=0
    )
    policy_seeds = _seed_stream(
        protocol.actor_seed, protocol.num_episodes, stream=1
    )
    if np.unique(random_layout_seeds).size != protocol.num_episodes:
        raise RuntimeError("random-layout seed stream contains a collision")
    if np.unique(policy_seeds).size != protocol.num_episodes:
        raise RuntimeError("policy seed stream contains a collision")
    if protocol.layout_mode == "fixed":
        layout_seeds = np.full(
            (protocol.num_episodes,), protocol.fixed_layout_seed, dtype=np.uint32
        )
    else:
        layout_seeds = random_layout_seeds

    fields = {
        "episode_index": np.arange(protocol.num_episodes, dtype=np.int64),
        "layout_seed": layout_seeds.astype(np.uint32),
        "policy_seed": policy_seeds.astype(np.uint32),
        "reward": np.zeros((protocol.num_episodes,), dtype=np.float64),
        "cost": np.zeros((protocol.num_episodes,), dtype=np.float64),
        "any_violation": np.zeros((protocol.num_episodes,), dtype=np.bool_),
        "violation_count": np.zeros((protocol.num_episodes,), dtype=np.int64),
        "length": np.zeros((protocol.num_episodes,), dtype=np.int64),
        "success": np.zeros((protocol.num_episodes,), dtype=np.bool_),
        "terminated": np.zeros((protocol.num_episodes,), dtype=np.bool_),
        "truncated": np.zeros((protocol.num_episodes,), dtype=np.bool_),
    }
    # Accumulate velocity over scored transitions without changing bootstrap draws.
    velocity_sum = np.zeros((protocol.num_episodes,), dtype=np.float64)
    velocity_count = np.zeros((protocol.num_episodes,), dtype=np.int64)

    workers = min(protocol.num_workers, protocol.num_episodes)
    for batch_start in range(0, protocol.num_episodes, workers):
        batch_stop = min(batch_start + workers, protocol.num_episodes)
        global_indices = np.arange(batch_start, batch_stop, dtype=np.int64)
        seeds = layout_seeds[global_indices]
        constructors = [
            (lambda one_seed=int(seed): env_factory(one_seed, protocol.layout_mode))
            for seed in seeds
        ]
        if protocol.vector_backend == "async":
            # Spawn avoids unsafe fork-after-JAX/MuJoCo behavior.  Environment
            # workers step in parallel while the actor remains batched in the
            # parent process (and may execute on one GPU).
            vector_env = gym.vector.AsyncVectorEnv(constructors, context="spawn")
        else:
            vector_env = gym.vector.SyncVectorEnv(constructors)
        try:
            observations, _ = vector_env.reset()
            active = np.ones((global_indices.size,), dtype=np.bool_)
            action_space = vector_env.single_action_space
            if not isinstance(action_space, gym.spaces.Box):
                raise TypeError("evaluation currently requires a Box action space")
            neutral_action = np.clip(
                np.zeros(action_space.shape, dtype=action_space.dtype),
                action_space.low,
                action_space.high,
            )

            for step in range(protocol.max_episode_steps):
                active_local = np.flatnonzero(active)
                if active_local.size == 0:
                    break
                actions = np.broadcast_to(
                    neutral_action, (global_indices.size,) + action_space.shape
                ).copy()
                flat = _flat_observations(observations)
                sample_seeds = _action_seed(
                    policy_seeds[global_indices[active_local]], step
                )
                sampled = np.asarray(
                    actor(flat[active_local], sample_seeds), dtype=action_space.dtype
                )
                expected = (active_local.size,) + action_space.shape
                if sampled.shape != expected:
                    raise ValueError(
                        f"P=1 actor must return shape {expected}, got {sampled.shape}"
                    )
                if not np.isfinite(sampled).all():
                    raise ValueError("actor returned a non-finite action")
                actions[active_local] = np.clip(
                    sampled, action_space.low, action_space.high
                )

                observations, rewards, terminated, truncated, infos = vector_env.step(
                    actions
                )
                for local in active_local:
                    episode = int(global_indices[local])
                    terminal = bool(terminated[local] or truncated[local])
                    one_info = _single_info(infos, int(local), terminal)
                    if "cost" not in one_info:
                        raise KeyError("environment transition is missing info['cost']")
                    cost = float(np.asarray(one_info["cost"]))
                    reward = float(rewards[local])
                    if not np.isfinite(cost) or not np.isfinite(reward):
                        raise ValueError("environment returned non-finite reward or cost")
                    fields["reward"][episode] += reward
                    fields["cost"][episode] += cost
                    fields["length"][episode] += 1
                    fields["any_violation"][episode] |= cost > 0.0
                    fields["violation_count"][episode] += int(cost > 0.0)
                    if "x_velocity" in one_info:
                        velocity_sum[episode] += float(np.asarray(one_info["x_velocity"]))
                        velocity_count[episode] += 1
                    fields["success"][episode] |= bool(
                        one_info.get("success", one_info.get("goal_met", False))
                    )
                    reached_limit = step + 1 >= protocol.max_episode_steps
                    if terminal or reached_limit:
                        fields["terminated"][episode] = bool(terminated[local])
                        fields["truncated"][episode] = bool(
                            truncated[local] or (reached_limit and not terminated[local])
                        )
                        active[local] = False
        finally:
            vector_env.close()

    summary = summarize_episodes(
        fields,
        bootstrap_seed=protocol.bootstrap_seed,
        bootstrap_samples=protocol.bootstrap_samples,
        confidence=protocol.confidence,
    )
    if velocity_count.any():
        velocity = np.where(velocity_count > 0, velocity_sum / np.maximum(velocity_count, 1), np.nan)
        fields["x_velocity"] = velocity
        finite = velocity[np.isfinite(velocity)]
        summary["x_velocity"] = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "q05": float(np.quantile(finite, 0.05)),
            "q25": float(np.quantile(finite, 0.25)),
            "q75": float(np.quantile(finite, 0.75)),
            "q95": float(np.quantile(finite, 0.95)),
            "episodes": int(finite.size),
        }
    metadata = {
        "env_id": protocol.env_id,
        "layout_mode": protocol.layout_mode,
        "layout_semantics": (
            "same reset seed for every episode"
            if protocol.layout_mode == "fixed"
            else "one recorded reset seed per episode"
        ),
        "num_episodes": protocol.num_episodes,
        "num_workers": workers,
        "vector_backend": protocol.vector_backend,
        "multiprocessing_context": (
            "spawn" if protocol.vector_backend == "async" else None
        ),
        "max_episode_steps": protocol.max_episode_steps,
        "evaluation_seed": protocol.evaluation_seed,
        "fixed_layout_seed": protocol.fixed_layout_seed,
        "actor_seed": protocol.actor_seed,
        "actor_samples_per_decision": 1,
        "action_selection": "stochastic P=1; no best-of-N or ranking",
        "bootstrap_seed": protocol.bootstrap_seed,
        "bootstrap_samples": protocol.bootstrap_samples,
        "confidence": protocol.confidence,
        "random_layout_seed_stream": "numpy.SeedSequence([evaluation_seed, 0])",
        "policy_seed_stream": "numpy.SeedSequence([actor_seed, 1])",
        "action_seed_stream": "documented _action_seed(policy_seed, step)",
    }
    return EvaluationResult(metadata, fields, summary)



class _LayoutResetProtocol(gym.Wrapper):
    """Make navigation layout randomization an explicit experiment variable.

    Safety-Gymnasium resamples placements inside every reset. ``random`` accepts
    the trainer's explicit episode-derived reset seed (or advances normally if
    none is supplied). ``fixed`` reuses one seed on every reset, fixing all
    randomness controlled by that reset, including obstacles, goal, and initial
    agent placement.
    """

    def __init__(self, env, seed: int, layout_mode: str):
        super().__init__(env)
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
            raise TypeError("seed must be an integer")
        self._initial_seed = int(seed)
        if layout_mode not in ("random", "fixed"):
            raise ValueError("layout_mode must be 'random' or 'fixed'")
        self.layout_mode = layout_mode
        self._has_reset = False

    def reset(self, *, seed=None, options=None):
        if self.layout_mode == "fixed":
            if seed is not None and int(seed) != self._initial_seed:
                raise ValueError(
                    "fixed layout reset seed must equal the configured layout seed"
                )
            result = self.env.reset(seed=self._initial_seed, options=options)
            self._has_reset = True
            return result
        if not self._has_reset:
            seed = self._initial_seed if seed is None else seed
            result = self.env.reset(seed=seed, options=options)
            self._has_reset = True
            return result
        return self.env.reset(seed=seed, options=options)


def make_safety_gymnasium_env(
    env_id: str,
    seed: int | None = 0,
    *,
    layout_mode: str = "random",
    **kwargs,
):
    """Return an un-reset 5-tuple env with transition cost in ``info['cost']``.

    Random-layout mode applies ``seed`` on the first reset and then advances
    normally. Fixed-layout mode reapplies it on every reset. This preserves the
    first seeded observation instead of discarding it in the factory.
    """
    import safety_gymnasium

    env = safety_gymnasium.make(env_id, **kwargs)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    if seed is not None:
        env = _LayoutResetProtocol(env, seed, layout_mode)
    elif layout_mode != "random":
        raise ValueError("fixed layout_mode requires an explicit integer seed")
    return env

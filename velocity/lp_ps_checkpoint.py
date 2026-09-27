"""Full, atomic schema-v4 checkpoints for LP-PS-SSM.

Unlike the older model-only checkpoints, this module captures the state needed
to continue an *online* run: every TrainState in ``agent.core``, the frozen
reference actor, replay storage and sampling RNG, runner counters/current
observation, observation statistics, process/environment RNGs, and (when the
environment exposes it) MuJoCo simulator and episode-accumulator state.

The loader deliberately takes freshly constructed agent/replay/environment
templates.  ``apply_fn`` and optimizer transforms contain Python callables and
are therefore supplied by the templates; only the numerical TrainState fields
(``params``, ``opt_state`` and ``step``) are serialized and restored.

Pickle files must only be loaded from trusted local experiment directories.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import dataclasses
import errno
import hashlib
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
from typing import Any

import jax
import numpy as np


CHECKPOINT_SCHEMA_VERSION = 4
CHECKPOINT_SCHEMA_NAME = "lp_ps_ssm_full_online"

_RUNNER_KEY_ALIASES = {
    "global_step": ("global_step", "global_env_step", "env_step"),
    "update_step": (
        "update_step",
        "gradient_step",
        "global_update_step",
        "gradient_update_step",
    ),
    "episode": ("episode", "episodes", "episode_index", "episode_count"),
    "current_observation": ("current_observation", "observation"),
}

_EPISODE_ACCUMULATOR_FIELDS = (
    "_ep_reward",
    "_ep_cost_binary",
    "_ep_sdf_violation",
    "_ep_length",
    "_ep_v_sum",
    "_ep_v_max",
)

_MUJOCO_ARRAY_FIELDS = (
    "act",
    "ctrl",
    "qacc_warmstart",
    "mocap_pos",
    "mocap_quat",
    "userdata",
)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"LP-PS checkpoint requires mapping {name!r}.")
    return value


def _host_tree(value: Any) -> Any:
    """Copy a PyTree to host memory so the pickle owns a stable snapshot."""
    return jax.tree_util.tree_map(
        lambda leaf: np.array(jax.device_get(leaf), copy=True)
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
        else copy.deepcopy(leaf),
        value,
    )


def _tree_compatible(name: str, expected: Any, loaded: Any) -> None:
    """Require identical PyTree structure, leaf shape and numeric dtype."""
    expected_leaves, expected_def = jax.tree_util.tree_flatten(expected)
    loaded_leaves, loaded_def = jax.tree_util.tree_flatten(loaded)
    if expected_def != loaded_def:
        raise ValueError(f"Checkpoint {name} PyTree structure mismatch.")
    if len(expected_leaves) != len(loaded_leaves):
        raise ValueError(f"Checkpoint {name} leaf-count mismatch.")
    for index, (expected_leaf, loaded_leaf) in enumerate(
        zip(expected_leaves, loaded_leaves)
    ):
        expected_shape = tuple(np.shape(expected_leaf))
        loaded_shape = tuple(np.shape(loaded_leaf))
        if expected_shape != loaded_shape:
            raise ValueError(
                f"Checkpoint {name} leaf {index} shape mismatch: "
                f"expected {expected_shape}, got {loaded_shape}."
            )
        expected_dtype = getattr(expected_leaf, "dtype", None)
        loaded_dtype = getattr(loaded_leaf, "dtype", None)
        if expected_dtype is not None and loaded_dtype is not None:
            if np.dtype(expected_dtype) != np.dtype(loaded_dtype):
                raise ValueError(
                    f"Checkpoint {name} leaf {index} dtype mismatch: "
                    f"expected {np.dtype(expected_dtype)}, "
                    f"got {np.dtype(loaded_dtype)}."
                )


def _core_field_names(core: Any) -> tuple[str, ...]:
    fields = getattr(core, "__dataclass_fields__", None)
    if fields:
        return tuple(str(name) for name in fields)
    if dataclasses.is_dataclass(core):
        return tuple(field.name for field in dataclasses.fields(core))
    namespace = getattr(core, "__dict__", None)
    if isinstance(namespace, dict):
        return tuple(str(name) for name in namespace)
    raise TypeError(
        "agent.core must expose dataclass fields or an instance namespace."
    )


def _is_train_state(value: Any) -> bool:
    return all(
        hasattr(value, attribute)
        for attribute in ("params", "opt_state", "step", "replace")
    )


def _capture_agent(agent: Any) -> dict[str, Any]:
    if not hasattr(agent, "core"):
        raise TypeError("LP-PS agent must expose .core.")
    core = agent.core
    train_states: dict[str, Any] = {}
    for name in _core_field_names(core):
        state = getattr(core, name)
        if _is_train_state(state):
            train_states[name] = {
                "params": _host_tree(state.params),
                "opt_state": _host_tree(state.opt_state),
                "step": _host_tree(state.step),
            }
    if not train_states:
        raise ValueError("LP-PS agent.core contains no detectable TrainState.")
    if not hasattr(core, "rng"):
        raise AttributeError("LP-PS agent.core must expose .rng.")

    required_agent_fields = (
        "reference_score_params",
        "actor_update_count",
        "reference_block_count",
    )
    missing = [name for name in required_agent_fields if not hasattr(agent, name)]
    if missing:
        raise AttributeError(f"LP-PS agent missing checkpoint fields: {missing}.")

    return {
        "core_train_states": train_states,
        "core_rng": _host_tree(core.rng),
        "reference_score_params": _host_tree(agent.reference_score_params),
        "actor_update_count": _host_tree(agent.actor_update_count),
        "reference_block_count": _host_tree(agent.reference_block_count),
    }


def restore_agent(agent_template: Any, checkpoint_agent: Mapping[str, Any]) -> Any:
    """Install schema-v4 numerical state into a freshly created LPPSAgent."""
    checkpoint_agent = _require_mapping(checkpoint_agent, "agent")
    train_states = _require_mapping(
        checkpoint_agent.get("core_train_states"), "agent.core_train_states"
    )
    core = agent_template.core
    template_states = {
        name: getattr(core, name)
        for name in _core_field_names(core)
        if _is_train_state(getattr(core, name))
    }
    if set(train_states) != set(template_states):
        missing = sorted(set(template_states) - set(train_states))
        unexpected = sorted(set(train_states) - set(template_states))
        raise ValueError(
            "Checkpoint/template TrainState names differ: "
            f"missing={missing}, unexpected={unexpected}."
        )

    core_updates: dict[str, Any] = {}
    for name, template_state in template_states.items():
        saved = _require_mapping(
            train_states[name], f"agent.core_train_states[{name!r}]"
        )
        for field in ("params", "opt_state", "step"):
            if field not in saved:
                raise KeyError(f"Checkpoint TrainState {name!r} missing {field!r}.")
        _tree_compatible(f"{name}.params", template_state.params, saved["params"])
        _tree_compatible(
            f"{name}.opt_state", template_state.opt_state, saved["opt_state"]
        )
        _tree_compatible(f"{name}.step", template_state.step, saved["step"])
        core_updates[name] = template_state.replace(
            params=saved["params"],
            opt_state=saved["opt_state"],
            step=saved["step"],
        )

    if "core_rng" not in checkpoint_agent:
        raise KeyError("Checkpoint agent missing core_rng.")
    _tree_compatible("core_rng", core.rng, checkpoint_agent["core_rng"])
    core_updates["rng"] = checkpoint_agent["core_rng"]
    restored_core = core.replace(**core_updates)

    if "reference_score_params" not in checkpoint_agent:
        raise KeyError("Checkpoint agent missing reference_score_params.")
    _tree_compatible(
        "reference_score_params",
        agent_template.reference_score_params,
        checkpoint_agent["reference_score_params"],
    )
    agent_updates: dict[str, Any] = {
        "core": restored_core,
        "reference_score_params": checkpoint_agent["reference_score_params"],
    }
    for name in ("actor_update_count", "reference_block_count"):
        if name not in checkpoint_agent:
            raise KeyError(f"Checkpoint agent missing {name}.")
        template_value = getattr(agent_template, name)
        _tree_compatible(name, template_value, checkpoint_agent[name])
        raw_count = np.asarray(checkpoint_agent[name])
        if raw_count.shape != () or not np.issubdtype(raw_count.dtype, np.integer):
            raise ValueError(f"Checkpoint {name} must be an integer scalar.")
        if int(raw_count) < 0:
            raise ValueError(f"Checkpoint {name} must be nonnegative.")
        agent_updates[name] = checkpoint_agent[name]
    return agent_template.replace(**agent_updates)


def _capture_numpy_rng(rng: Any) -> dict[str, Any] | None:
    if rng is None:
        return None
    if hasattr(rng, "bit_generator"):
        return {
            "kind": "generator",
            "bit_generator": type(rng.bit_generator).__name__,
            "state": copy.deepcopy(rng.bit_generator.state),
        }
    if hasattr(rng, "get_state"):
        return {"kind": "random_state", "state": copy.deepcopy(rng.get_state())}
    raise TypeError(f"Unsupported NumPy RNG type {type(rng)!r}.")


def _restore_numpy_rng(rng: Any, saved: Mapping[str, Any], name: str) -> None:
    saved = _require_mapping(saved, name)
    kind = saved.get("kind")
    if kind == "generator":
        if not hasattr(rng, "bit_generator"):
            raise TypeError(f"{name} template is not a NumPy Generator.")
        actual = type(rng.bit_generator).__name__
        expected = str(saved.get("bit_generator", ""))
        if actual != expected:
            raise ValueError(
                f"{name} bit-generator mismatch: expected {expected}, got {actual}."
            )
        rng.bit_generator.state = copy.deepcopy(saved["state"])
        return
    if kind == "random_state":
        if not hasattr(rng, "set_state"):
            raise TypeError(f"{name} template is not a NumPy RandomState.")
        rng.set_state(copy.deepcopy(saved["state"]))
        return
    raise ValueError(f"Checkpoint {name} has unsupported RNG kind {kind!r}.")


def _slice_replay_tree(tree: Any, *, size: int, capacity: int, path: str) -> Any:
    if isinstance(tree, Mapping):
        return {
            str(key): _slice_replay_tree(
                value, size=size, capacity=capacity, path=f"{path}.{key}"
            )
            for key, value in tree.items()
        }
    value = np.asarray(tree)
    if value.ndim < 1 or value.shape[0] != capacity:
        raise ValueError(
            f"Replay array {path} must have leading capacity {capacity}, "
            f"got {value.shape}."
        )
    return np.array(value[:size], copy=True)


def _capture_replay(replay: Any) -> dict[str, Any]:
    for name in ("dataset_dict", "_capacity", "_size", "_insert_index"):
        if not hasattr(replay, name):
            raise TypeError(f"Replay template missing {name}.")
    capacity = int(replay._capacity)
    size = int(replay._size)
    cursor = int(replay._insert_index)
    if capacity <= 0 or not 0 <= size <= capacity or not 0 <= cursor < capacity:
        raise ValueError(
            f"Invalid replay state capacity={capacity}, size={size}, cursor={cursor}."
        )
    if size < capacity and cursor != size:
        raise ValueError(
            "Partially-filled SafeReplayBuffer must have cursor == size; "
            f"got size={size}, cursor={cursor}."
        )

    # Accessing .np_random initializes the exact RNG that the next .sample()
    # would otherwise create, making continuation deterministic from this save.
    replay_rng = replay.np_random
    payload: dict[str, Any] = {
        "capacity": capacity,
        "size": size,
        "cursor": cursor,
        "storage": _slice_replay_tree(
            replay.dataset_dict, size=size, capacity=capacity, path="replay.storage"
        ),
        "np_random": _capture_numpy_rng(replay_rng),
        "seed": copy.deepcopy(getattr(replay, "_seed", None)),
    }
    # Dataset.sample_jax is not used by the LP runner, but preserve its key if
    # a caller initialized it.  A restored buffer must already have the same
    # cached sampler; otherwise fail rather than silently claim exactness.
    if hasattr(replay, "rng"):
        payload["jax_sampling_rng"] = _host_tree(replay.rng)
        payload["jax_sampler_initialized"] = bool(hasattr(replay, "_sample_jax"))
    else:
        payload["jax_sampler_initialized"] = False
    return payload


def _validate_replay_storage(template: Any, saved: Any, size: int, path: str) -> None:
    if isinstance(template, Mapping):
        saved = _require_mapping(saved, path)
        if set(template) != set(saved):
            raise ValueError(
                f"Checkpoint {path} keys differ: expected {sorted(template)}, "
                f"got {sorted(saved)}."
            )
        for key in template:
            _validate_replay_storage(
                template[key], saved[key], size, f"{path}.{key}"
            )
        return
    template_array = np.asarray(template)
    saved_array = np.asarray(saved)
    expected_shape = (size, *template_array.shape[1:])
    if saved_array.shape != expected_shape:
        raise ValueError(
            f"Checkpoint {path} shape mismatch: expected {expected_shape}, "
            f"got {saved_array.shape}."
        )
    if saved_array.dtype != template_array.dtype:
        raise ValueError(
            f"Checkpoint {path} dtype mismatch: expected {template_array.dtype}, "
            f"got {saved_array.dtype}."
        )


def _copy_replay_storage(template: Any, saved: Any, size: int) -> None:
    if isinstance(template, Mapping):
        for key in template:
            _copy_replay_storage(template[key], saved[key], size)
        return
    if size:
        template[:size] = saved


def restore_replay_buffer(replay_template: Any, checkpoint_replay: Mapping[str, Any]) -> Any:
    """Restore effective storage, ring position, and sampling RNG in place."""
    checkpoint_replay = _require_mapping(checkpoint_replay, "replay")
    capacity = int(checkpoint_replay.get("capacity", -1))
    size = int(checkpoint_replay.get("size", -1))
    cursor = int(checkpoint_replay.get("cursor", -1))
    if capacity != int(replay_template._capacity):
        raise ValueError(
            f"Replay capacity mismatch: checkpoint={capacity}, "
            f"template={replay_template._capacity}."
        )
    if not 0 <= size <= capacity or not 0 <= cursor < capacity:
        raise ValueError(
            f"Invalid checkpoint replay size/cursor: size={size}, cursor={cursor}."
        )
    if size < capacity and cursor != size:
        raise ValueError("Partially-filled checkpoint replay has cursor != size.")
    storage = checkpoint_replay.get("storage")
    _validate_replay_storage(
        replay_template.dataset_dict, storage, size, "replay.storage"
    )
    if checkpoint_replay.get("np_random") is None:
        raise ValueError("Checkpoint replay is missing its NumPy sampling RNG.")

    # All potentially failing shape/schema checks precede mutation.
    _copy_replay_storage(replay_template.dataset_dict, storage, size)
    replay_template._size = size
    replay_template._insert_index = cursor
    replay_template._seed = copy.deepcopy(checkpoint_replay.get("seed"))
    _restore_numpy_rng(
        replay_template.np_random,
        checkpoint_replay["np_random"],
        "replay.np_random",
    )

    sampler_was_initialized = bool(
        checkpoint_replay.get("jax_sampler_initialized", False)
    )
    if sampler_was_initialized:
        if not hasattr(replay_template, "_sample_jax"):
            raise ValueError(
                "Checkpoint used Dataset.sample_jax, but replay_template has no "
                "identically configured cached JAX sampler. Recreate that sampler "
                "before restoring or use the LP runner's NumPy sample path."
            )
        if "jax_sampling_rng" not in checkpoint_replay:
            raise KeyError("Checkpoint replay missing jax_sampling_rng.")
        if hasattr(replay_template, "rng"):
            _tree_compatible(
                "replay.jax_sampling_rng",
                replay_template.rng,
                checkpoint_replay["jax_sampling_rng"],
            )
        replay_template.rng = checkpoint_replay["jax_sampling_rng"]
    else:
        # Prevent a template's stale cached sampler/device storage from being
        # used with the newly restored arrays.
        for attribute in ("rng", "_sample_jax"):
            if hasattr(replay_template, attribute):
                delattr(replay_template, attribute)
    return replay_template


def _environment_chain(env: Any) -> list[Any]:
    chain: list[Any] = []
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        next_env = getattr(current, "env", None)
        if next_env is None or id(next_env) in seen:
            break
        current = next_env
    return chain


def _environment_id(env: Any) -> str:
    env_id = getattr(env, "env_id", None)
    if env_id:
        return str(env_id)
    for candidate in reversed(_environment_chain(env)):
        spec = getattr(candidate, "spec", None)
        spec_id = getattr(spec, "id", None)
        if spec_id:
            return str(spec_id)
    return ""


def _capture_observation_stats(env: Any) -> dict[str, Any]:
    if not hasattr(env, "get_obs_stats"):
        raise TypeError("LP-PS environment must implement get_obs_stats().")
    mean, var, count = env.get_obs_stats()
    mean = np.array(mean, copy=True)
    var = np.array(var, copy=True)
    count = int(count)
    expected_shape = tuple(getattr(env.observation_space, "shape", ()))
    if mean.shape != expected_shape or var.shape != expected_shape:
        raise ValueError(
            "Observation-stat shape mismatch: "
            f"expected {expected_shape}, got {mean.shape}/{var.shape}."
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise ValueError("Observation statistics must be finite.")
    if np.any(var < 0.0) or count < 0:
        raise ValueError("Observation variance/count must be nonnegative.")
    return {"mean": mean, "var": var, "count": count}


def _find_mujoco_env(env: Any) -> Any | None:
    unwrapped = getattr(getattr(env, "env", env), "unwrapped", None)
    candidates = ([unwrapped] if unwrapped is not None else []) + list(
        reversed(_environment_chain(env))
    )
    for candidate in candidates:
        if candidate is None:
            continue
        namespace = getattr(candidate, "__dict__", {})
        if "data" in namespace and "model" in namespace:
            data = namespace["data"]
            if hasattr(data, "qpos") and hasattr(data, "qvel"):
                return candidate
    return None


def _capture_environment(env: Any) -> dict[str, Any]:
    chain = _environment_chain(env)
    wrapper_state: list[dict[str, Any]] = []
    for index, layer in enumerate(chain):
        state: dict[str, Any] = {
            "index": index,
            "type": f"{type(layer).__module__}.{type(layer).__qualname__}",
        }
        for field in ("_elapsed_steps", "_has_reset", "checked_reset", "checked_step"):
            if field in getattr(layer, "__dict__", {}):
                state[field] = copy.deepcopy(getattr(layer, field))
        rng = getattr(layer, "__dict__", {}).get("_np_random")
        if rng is not None:
            state["np_random"] = _capture_numpy_rng(rng)
        if len(state) > 2:
            wrapper_state.append(state)

    episode_accumulators = {
        name: copy.deepcopy(getattr(env, name))
        for name in _EPISODE_ACCUMULATOR_FIELDS
        if hasattr(env, name)
    }
    result: dict[str, Any] = {
        "env_id": _environment_id(env),
        "observation_shape": tuple(getattr(env.observation_space, "shape", ())),
        "action_shape": tuple(getattr(env.action_space, "shape", ())),
        "wrapper_seed": copy.deepcopy(getattr(env, "_seed", None)),
        "wrapper_state": wrapper_state,
        "episode_accumulators": episode_accumulators,
        "update_stats": copy.deepcopy(getattr(env, "_update_stats", None)),
        "action_space_rng": _capture_numpy_rng(env.action_space.np_random),
        "mujoco": {"available": False},
    }

    raw = _find_mujoco_env(env)
    if raw is not None:
        data = raw.data
        mujoco_state: dict[str, Any] = {
            "available": True,
            "qpos": np.array(data.qpos, copy=True),
            "qvel": np.array(data.qvel, copy=True),
            "time": float(getattr(data, "time", 0.0)),
        }
        for name in _MUJOCO_ARRAY_FIELDS:
            if hasattr(data, name):
                mujoco_state[name] = np.array(getattr(data, name), copy=True)
        raw_rng = getattr(raw, "np_random", None)
        if raw_rng is not None:
            mujoco_state["env_np_random"] = _capture_numpy_rng(raw_rng)
        result["mujoco"] = mujoco_state
    return result


def _validate_environment_compatibility(
    env: Any,
    checkpoint_env: Mapping[str, Any],
    observation_stats: Mapping[str, Any],
) -> None:
    checkpoint_env = _require_mapping(checkpoint_env, "environment")
    expected_id = str(checkpoint_env.get("env_id", ""))
    actual_id = _environment_id(env)
    if not expected_id or expected_id != actual_id:
        raise ValueError(
            f"Environment-id mismatch: checkpoint={expected_id!r}, "
            f"template={actual_id!r}."
        )
    for name, space in (
        ("observation_shape", env.observation_space),
        ("action_shape", env.action_space),
    ):
        expected = tuple(checkpoint_env.get(name, ()))
        actual = tuple(getattr(space, "shape", ()))
        if expected != actual:
            raise ValueError(
                f"Environment {name} mismatch: checkpoint={expected}, "
                f"template={actual}."
            )

    stats = _require_mapping(observation_stats, "observation_stats")
    obs_shape = tuple(getattr(env.observation_space, "shape", ()))
    mean = np.asarray(stats.get("mean"))
    var = np.asarray(stats.get("var"))
    count = stats.get("count")
    if mean.shape != obs_shape or var.shape != obs_shape:
        raise ValueError("Checkpoint observation-stat shape mismatch.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise ValueError("Checkpoint observation statistics are non-finite.")
    if np.any(var < 0.0) or isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("Checkpoint observation variance/count is invalid.")
    if count < 0 or not hasattr(env, "set_obs_stats"):
        raise ValueError("Environment cannot accept checkpoint observation stats.")

    mujoco_state = _require_mapping(checkpoint_env.get("mujoco"), "environment.mujoco")
    if bool(mujoco_state.get("available", False)):
        raw = _find_mujoco_env(env)
        if raw is None:
            raise ValueError("Checkpoint has MuJoCo state but template does not expose it.")
        for name in ("qpos", "qvel"):
            expected_shape = np.asarray(getattr(raw.data, name)).shape
            saved_shape = np.asarray(mujoco_state.get(name)).shape
            if saved_shape != expected_shape:
                raise ValueError(
                    f"MuJoCo {name} shape mismatch: expected {expected_shape}, "
                    f"got {saved_shape}."
                )
        for name in _MUJOCO_ARRAY_FIELDS:
            if name in mujoco_state:
                if not hasattr(raw.data, name):
                    raise ValueError(f"Template MuJoCo data lacks saved field {name}.")
                if np.asarray(mujoco_state[name]).shape != np.asarray(
                    getattr(raw.data, name)
                ).shape:
                    raise ValueError(f"MuJoCo {name} shape mismatch.")


def _initialize_lazy_environment_rngs(
    env: Any,
    checkpoint_env: Mapping[str, Any],
) -> None:
    """Materialize Gym's lazy wrapper RNGs before installing saved states.

    Several Gym/Gymnasium wrappers create ``_np_random`` only on their first
    reset.  A freshly constructed MuJoCo environment therefore cannot accept
    an otherwise complete mid-episode checkpoint: the saved wrapper RNG has no
    destination object yet.  One reset is safe here because every state it can
    affect (simulator, wrapper counters/RNGs, observation statistics, episode
    accumulators and process RNGs) is restored from the checkpoint afterwards.
    """

    chain = _environment_chain(env)
    missing_indices = []
    for saved_layer in checkpoint_env.get("wrapper_state", ()):
        saved_layer = _require_mapping(
            saved_layer, "environment.wrapper_state[]"
        )
        if "np_random" not in saved_layer:
            continue
        index = int(saved_layer.get("index", -1))
        if not 0 <= index < len(chain):
            # Compatibility validation reports the more useful structural
            # error; do not turn it into an IndexError here.
            continue
        if getattr(chain[index], "__dict__", {}).get("_np_random") is None:
            missing_indices.append(index)

    if not missing_indices:
        return

    env.reset()
    chain = _environment_chain(env)
    still_missing = [
        index
        for index in missing_indices
        if getattr(chain[index], "__dict__", {}).get("_np_random") is None
    ]
    if still_missing:
        raise ValueError(
            "Environment reset did not initialize saved wrapper RNGs at "
            f"indices {still_missing}."
        )


def restore_environment(
    env: Any,
    checkpoint_env: Mapping[str, Any],
    observation_stats: Mapping[str, Any],
) -> Any:
    """Restore normalization, wrapper RNG/state, and available simulator state."""
    _validate_environment_compatibility(env, checkpoint_env, observation_stats)
    _initialize_lazy_environment_rngs(env, checkpoint_env)
    stats = observation_stats
    env.set_obs_stats(
        np.array(stats["mean"], copy=True),
        np.array(stats["var"], copy=True),
        int(stats["count"]),
    )
    if checkpoint_env.get("wrapper_seed") is not None and hasattr(env, "_seed"):
        env._seed = copy.deepcopy(checkpoint_env["wrapper_seed"])
    if checkpoint_env.get("update_stats") is not None and hasattr(env, "_update_stats"):
        env._update_stats = bool(checkpoint_env["update_stats"])
    for name, value in _require_mapping(
        checkpoint_env.get("episode_accumulators", {}),
        "environment.episode_accumulators",
    ).items():
        if not hasattr(env, name):
            raise AttributeError(f"Environment lacks saved accumulator {name!r}.")
        setattr(env, name, copy.deepcopy(value))

    chain = _environment_chain(env)
    for saved_layer in checkpoint_env.get("wrapper_state", ()):
        saved_layer = _require_mapping(saved_layer, "environment.wrapper_state[]")
        index = int(saved_layer.get("index", -1))
        if not 0 <= index < len(chain):
            raise ValueError(f"Saved environment wrapper index {index} is unavailable.")
        layer = chain[index]
        expected_type = str(saved_layer.get("type", ""))
        actual_type = f"{type(layer).__module__}.{type(layer).__qualname__}"
        if expected_type != actual_type:
            raise ValueError(
                f"Environment wrapper type mismatch at {index}: "
                f"expected {expected_type}, got {actual_type}."
            )
        for field in ("_elapsed_steps", "_has_reset", "checked_reset", "checked_step"):
            if field in saved_layer:
                setattr(layer, field, copy.deepcopy(saved_layer[field]))
        if "np_random" in saved_layer:
            rng = getattr(layer, "__dict__", {}).get("_np_random")
            if rng is None:
                raise ValueError(f"Environment wrapper {index} lacks saved RNG.")
            _restore_numpy_rng(rng, saved_layer["np_random"], f"env.wrapper[{index}].rng")

    if checkpoint_env.get("action_space_rng") is not None:
        _restore_numpy_rng(
            env.action_space.np_random,
            checkpoint_env["action_space_rng"],
            "environment.action_space_rng",
        )

    mujoco_state = checkpoint_env["mujoco"]
    if bool(mujoco_state.get("available", False)):
        raw = _find_mujoco_env(env)
        raw.set_state(
            np.array(mujoco_state["qpos"], copy=True),
            np.array(mujoco_state["qvel"], copy=True),
        )
        for name in _MUJOCO_ARRAY_FIELDS:
            if name in mujoco_state:
                np.copyto(getattr(raw.data, name), np.asarray(mujoco_state[name]))
        raw.data.time = float(mujoco_state.get("time", raw.data.time))
        if "env_np_random" in mujoco_state:
            _restore_numpy_rng(
                raw.np_random,
                mujoco_state["env_np_random"],
                "environment.mujoco.env_np_random",
            )
    return env


def _validate_runner_state(runner_state: Mapping[str, Any]) -> dict[str, str]:
    runner_state = _require_mapping(runner_state, "runner_state")
    selected: dict[str, str] = {}
    for semantic_name, aliases in _RUNNER_KEY_ALIASES.items():
        found = [name for name in aliases if name in runner_state]
        if len(found) != 1:
            raise ValueError(
                f"runner_state must contain exactly one {semantic_name} key "
                f"from {aliases}; found {found}."
            )
        selected[semantic_name] = found[0]
    for semantic_name in ("global_step", "update_step", "episode"):
        value = runner_state[selected[semantic_name]]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"runner_state {semantic_name} must be an integer.")
        if int(value) < 0:
            raise ValueError(f"runner_state {semantic_name} must be nonnegative.")
    observation = np.asarray(runner_state[selected["current_observation"]])
    if observation.ndim < 1 or not np.all(np.isfinite(observation)):
        raise ValueError("runner_state current observation must be a finite array.")
    return selected


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    if dataclasses.is_dataclass(config):
        return copy.deepcopy(dataclasses.asdict(config))
    namespace = getattr(config, "__dict__", None)
    if isinstance(namespace, dict):
        return copy.deepcopy(dict(namespace))
    raise TypeError("config must be a mapping, dataclass, or namespace object.")


class _HashingWriter:
    def __init__(self, file_object: Any):
        self.file_object = file_object
        self.hasher = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self.hasher.update(data)
        return self.file_object.write(data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.file_object, name)


def _atomic_pickle(path: Path, payload: Mapping[str, Any]) -> tuple[int, str]:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file_object:
            writer = _HashingWriter(file_object)
            pickle.dump(payload, writer, protocol=pickle.HIGHEST_PROTOCOL)
            file_object.flush()
            os.fsync(file_object.fileno())
            digest = writer.hasher.hexdigest()
        size_bytes = temporary.stat().st_size
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            # Some mounted filesystems (notably WSL drvfs) do not support
            # directory fsync. File fsync + same-directory atomic replace still
            # provides the strongest primitive that filesystem offers.
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
                raise
        return size_bytes, digest
    finally:
        if temporary.exists():
            temporary.unlink()


def save_lp_checkpoint(
    path: str | Path,
    *,
    agent: Any,
    replay: Any,
    env: Any,
    runner_state: Mapping[str, Any],
    config: Mapping[str, Any] | Any,
    wandb_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save a complete schema-v4 LP-PS online checkpoint.

    Returns a small manifest suitable for logging alongside the checkpoint.
    """
    runner_keys = _validate_runner_state(runner_state)
    config_payload = _config_dict(config)
    wandb_payload = dict(_require_mapping(wandb_info, "wandb_info"))
    created_unix = time.time()
    payload: dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_schema_name": CHECKPOINT_SCHEMA_NAME,
        "resume_scope": "full_online_state",
        "created_unix": created_unix,
        "agent": _capture_agent(agent),
        "replay": _capture_replay(replay),
        "observation_stats": _capture_observation_stats(env),
        "environment": _capture_environment(env),
        "runner_state": _host_tree(dict(runner_state)),
        "runner_state_keys": runner_keys,
        "random_state": {
            "python": copy.deepcopy(random.getstate()),
            "numpy_global": copy.deepcopy(np.random.get_state()),
        },
        "config": config_payload,
        "wandb": copy.deepcopy(wandb_payload),
    }
    target = Path(path).expanduser().resolve()
    size_bytes, sha256 = _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_schema_name": CHECKPOINT_SCHEMA_NAME,
        "created_unix": created_unix,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "global_step": int(runner_state[runner_keys["global_step"]]),
        "update_step": int(runner_state[runner_keys["update_step"]]),
        "episode": int(runner_state[runner_keys["episode"]]),
        "wandb_run_id": wandb_payload.get("run_id"),
        "mujoco_state_saved": bool(payload["environment"]["mujoco"]["available"]),
    }


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Read and validate the top-level structure of a trusted schema-v4 file."""
    checkpoint = Path(path).expanduser().resolve()
    with checkpoint.open("rb") as file_object:
        payload = pickle.load(file_object)
    if not isinstance(payload, dict):
        raise TypeError("LP-PS checkpoint payload must be a dictionary.")
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "LP-PS checkpoint schema mismatch: expected "
            f"{CHECKPOINT_SCHEMA_VERSION}, got "
            f"{payload.get('checkpoint_schema_version')!r}."
        )
    if payload.get("checkpoint_schema_name") != CHECKPOINT_SCHEMA_NAME:
        raise ValueError(
            f"LP-PS checkpoint schema name must be {CHECKPOINT_SCHEMA_NAME!r}."
        )
    if payload.get("resume_scope") != "full_online_state":
        raise ValueError("LP-PS checkpoint does not claim full online resume state.")
    for name in (
        "agent",
        "replay",
        "observation_stats",
        "environment",
        "runner_state",
        "random_state",
        "config",
        "wandb",
    ):
        _require_mapping(payload.get(name), name)
    saved_runner_keys = _require_mapping(
        payload.get("runner_state_keys"), "runner_state_keys"
    )
    actual_runner_keys = _validate_runner_state(payload["runner_state"])
    if dict(saved_runner_keys) != actual_runner_keys:
        raise ValueError("Checkpoint runner_state_keys disagree with runner_state.")
    random_state = payload["random_state"]
    if "python" not in random_state or "numpy_global" not in random_state:
        raise KeyError("Checkpoint missing Python or global NumPy RNG state.")
    return payload


def load_lp_checkpoint(
    path: str | Path,
    *,
    agent_template: Any,
    replay_template: Any,
    env: Any,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Restore full online state into compatible templates.

    Returns ``(agent, replay, runner_state, payload)``.  Process-global Python
    and NumPy RNGs are restored last, after all structural validation succeeds.
    """
    payload = load_checkpoint_payload(path)
    restored_agent = restore_agent(agent_template, payload["agent"])

    # Validate environment before mutating replay.  restore_replay_buffer and
    # restore_environment each perform their own complete shape checks before
    # installing array contents.
    _validate_environment_compatibility(
        env, payload["environment"], payload["observation_stats"]
    )
    restored_replay = restore_replay_buffer(replay_template, payload["replay"])
    restore_environment(env, payload["environment"], payload["observation_stats"])

    random.setstate(copy.deepcopy(payload["random_state"]["python"]))
    np.random.set_state(copy.deepcopy(payload["random_state"]["numpy_global"]))
    runner_state = _host_tree(dict(payload["runner_state"]))
    return restored_agent, restored_replay, runner_state, payload


# Concise aliases for callers that do not need to distinguish checkpoint
# families in their local namespace.
save_checkpoint = save_lp_checkpoint
load_checkpoint = load_lp_checkpoint


__all__ = (
    "CHECKPOINT_SCHEMA_NAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "load_checkpoint",
    "load_checkpoint_payload",
    "load_lp_checkpoint",
    "restore_agent",
    "restore_environment",
    "restore_replay_buffer",
    "save_checkpoint",
    "save_lp_checkpoint",
)

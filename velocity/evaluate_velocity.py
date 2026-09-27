"""Evaluate one trusted velocity checkpoint on the specified stochastic panel."""

from __future__ import annotations

import argparse
from argparse import Namespace
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path

# Evaluation is CPU-only, including spawned environment workers.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import evaluation_panel
from evaluation_panel import (
    EvaluationProtocol,
    evaluate_stochastic_policy,
    make_safety_gymnasium_env,
)
import lp_ps_checkpoint
from lp_ps_checkpoint import load_checkpoint_payload, restore_agent
# This import installs the process-local Swimmer environment registration.
from swimmer_role_particles import train_swimmer_role_particles as trainer
from swimmer_role_particles import particle_lp_ps_ssm_agent as agent_module
from swimmer_role_particles.particle_lp_ps_ssm_agent import bounded_latent_to_action
import mujoco_velocity
import jaxrl5


TASK_ENVS = {
    "cheetah": "SafetyHalfCheetahVelocity-v1",
    "swimmer": "SafetySwimmerVelocity-v1",
}


@jax.jit
def _sample(agent, observations, keys):
    def one(observation, key):
        latent, _ = agent.policy_latents(observation[None, :], key, use_reference=False)
        return (
            bounded_latent_to_action(latent, agent.epsilon_action)[0],
            jnp.max(jnp.abs(latent)),
            jnp.mean((~jnp.isfinite(latent)).astype(jnp.float32)),
        )

    return jax.vmap(one)(observations, keys)


def _validate(actions, latent_abs_max, latent_nonfinite):
    """Abort on an invalid latent or action without resampling or dropping rows."""
    bad = np.flatnonzero(
        (latent_nonfinite != 0.0)
        | ~np.isfinite(latent_abs_max)
        | (latent_abs_max > 1e4)
    )
    if bad.size:
        raise FloatingPointError(
            f"SSM evaluation sampled a non-finite/exploding latent, rows {bad.tolist()}"
        )
    bad = np.flatnonzero(
        (~np.isfinite(actions)).any(axis=1) | (np.abs(actions) >= 1.0).any(axis=1)
    )
    if bad.size:
        raise FloatingPointError(
            f"SSM evaluation produced a non-finite/boundary action, rows {bad.tolist()}"
        )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(f"output already exists: {a.out}")
    protocol_path = a.protocol.resolve()
    specification = yaml.safe_load(protocol_path.read_text())
    task = specification["task"]
    if specification["backend"] != "velocity" or task not in TASK_ENVS:
        p.error("protocol must name a velocity task: cheetah or swimmer")
    proto = EvaluationProtocol(**specification["protocol"])
    if proto.env_id != TASK_ENVS[task]:
        p.error("protocol task and environment do not match")
    proto.validate()

    checkpoint = a.checkpoint.resolve()
    checkpoint_sha = _sha256(checkpoint)
    payload = load_checkpoint_payload(checkpoint)
    config = Namespace(**dict(payload["config"]))
    if str(config.env_id) != proto.env_id:
        raise ValueError(f"checkpoint env {config.env_id} != protocol env {proto.env_id}")
    if config.particle_mode != "s1_min":
        raise ValueError("this evaluator requires the s1_min raw K=1 velocity recipe")
    step = int(payload["runner_state"][payload["runner_state_keys"]["global_step"]])
    stats = payload["observation_stats"]
    mean = np.asarray(stats["mean"], np.float64)
    var = np.asarray(stats["var"], np.float64)

    template = mujoco_velocity.make_velocity_env(env_id=proto.env_id, seed=int(config.seed))
    try:
        template.set_obs_stats(mean, var, int(stats["count"]))
        template.freeze_obs_stats()
        agent = restore_agent(trainer.make_agent(config, template), payload["agent"])
        clip = float(template._obs_norm_clip)
        low, high = template._act_low, template._act_high
    finally:
        template.env.close()
    std = np.sqrt(var + 1e-8)

    def actor(observations, sample_seeds):
        normed = np.clip(
            (np.asarray(observations, np.float64) - mean) / std, -clip, clip
        ).astype(np.float32)
        keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in np.asarray(sample_seeds)])
        action, latent_abs_max, latent_nonfinite = (
            np.asarray(x) for x in _sample(agent, jnp.asarray(normed), keys)
        )
        action = action.astype(np.float32)
        _validate(action, latent_abs_max.astype(np.float64), latent_nonfinite.astype(np.float64))
        return low + (action + 1.0) * 0.5 * (high - low)

    result = evaluate_stochastic_policy(
        lambda sd, md: make_safety_gymnasium_env(proto.env_id, seed=sd, layout_mode=md),
        actor,
        proto,
    )
    result.protocol["ssm_eval_action"] = (
        "raw K=1 DDPM latent (T=%d, temperature %s) from PRNGKey(per-step panel seed), "
        "tanh bound, [-1,1]->env box; the s1_min behaviour sampler; no selector or shield"
        % (int(config.T), config.ddpm_temperature)
    )
    result.protocol["ssm_observation_normalisation"] = (
        "checkpoint frozen Welford stats, (o-mean)/sqrt(var+1e-8), clip +-%g" % clip
    )
    sources = {
        "evaluator": Path(__file__).resolve(),
        **{
            module.__name__: Path(module.__file__).resolve()
            for module in (evaluation_panel, lp_ps_checkpoint, trainer, agent_module, mujoco_velocity, jaxrl5)
        },
    }
    document = result.json_dict()
    document["numerics"] = {
        "backend": jax.default_backend(),
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "episodes": proto.num_episodes,
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "checkpoint_sha256": checkpoint_sha,
        "train_seed": int(config.seed),
        "protocol_file": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "evaluator_sha256": _sha256(__file__),
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "versions": {
            name: metadata.version(name)
            for name in ("jax", "jaxlib", "flax", "numpy", "gymnasium", "safety-gymnasium", "mujoco")
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with a.out.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(
        f"{task} s{int(config.seed)} step {step} "
        f"R {result.summary['reward']['mean']:.8f} C {result.summary['cost']['mean']:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

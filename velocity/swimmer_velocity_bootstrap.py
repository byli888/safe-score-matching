"""Process-local SafetySwimmerVelocity-v1 environment registration."""

from __future__ import annotations

from typing import Any

import numpy as np

import mujoco_velocity


SWIMMER_ENV_ID = "SafetySwimmerVelocity-v1"
SWIMMER_VELOCITY_THRESHOLD = 0.2282
SWIMMER_EPISODE_LENGTH = 1000

_CONFIG: dict[str, Any] = {
    "v_threshold": SWIMMER_VELOCITY_THRESHOLD,
    "episode_length": SWIMMER_EPISODE_LENGTH,
    "terminates": False,
}


class SwimmerVelocityEnvWrapper(mujoco_velocity.VelocityEnvWrapper):
    """Swimmer wrapper with episode motion diagnostics."""

    def _reset_swimmer_diagnostics(self) -> None:
        self._swimmer_x_start = float(self.env.unwrapped.data.qpos[0])
        self._swimmer_x_final = self._swimmer_x_start
        self._swimmer_control_cost = 0.0
        self._swimmer_action_saturated = 0
        self._swimmer_action_elements = 0

    def reset(self) -> np.ndarray:
        observation = super().reset()
        self._reset_swimmer_diagnostics()
        return observation

    def step(self, action: np.ndarray):
        policy_action = np.asarray(action, dtype=np.float32)
        real_action = self._rescale_action(policy_action)
        output = super().step(policy_action)
        self._swimmer_x_final = float(self.env.unwrapped.data.qpos[0])
        # Swimmer-v4 uses ctrl_cost_weight=1e-4. Prefer the native reward term
        # when exposed, while keeping an exact fallback for older env versions.
        info = output[-1]
        native_ctrl = info.get("reward_ctrl")
        if native_ctrl is None:
            native_ctrl = info.get("reward_control")
        self._swimmer_control_cost += (
            -float(native_ctrl)
            if native_ctrl is not None
            else 1e-4 * float(np.sum(np.square(real_action)))
        )
        self._swimmer_action_saturated += int(
            np.sum(np.abs(policy_action) >= 0.99)
        )
        self._swimmer_action_elements += int(policy_action.size)
        return output

    @property
    def episode_info(self) -> dict[str, float]:
        result = dict(super().episode_info)
        result.update(
            displacement_x=self._swimmer_x_final - self._swimmer_x_start,
            control_cost=self._swimmer_control_cost,
            action_saturation_fraction=(
                self._swimmer_action_saturated
                / max(1, self._swimmer_action_elements)
            ),
            max_penetration=max(0.0, result["max_velocity"] - self.v_threshold),
        )
        return result


def install() -> None:
    """Install Swimmer support and deterministic warmup locally.

    Call before importing the environment factory in the training entrypoint.
    """

    existing = mujoco_velocity.VELOCITY_ENV_CONFIG.get(SWIMMER_ENV_ID)
    if existing is not None and existing != _CONFIG:
        raise RuntimeError(
            f"Conflicting {SWIMMER_ENV_ID} registry entry: {existing!r}."
        )
    mujoco_velocity.VELOCITY_ENV_CONFIG[SWIMMER_ENV_ID] = dict(_CONFIG)

    current_factory = mujoco_velocity.make_velocity_env
    if getattr(current_factory, "_swimmer_process_local_patch", False):
        return

    def seeded_factory(
        env_id: str = "SafetyHalfCheetahVelocity-v0",
        seed: int = 0,
        normalize_obs: bool = True,
        obs_norm_clip: float = 5.0,
    ):
        if env_id == SWIMMER_ENV_ID:
            env = SwimmerVelocityEnvWrapper(
                env_id=env_id,
                seed=seed,
                normalize_obs=normalize_obs,
                obs_norm_clip=obs_norm_clip,
            )
            # VelocityEnvWrapper creates a fresh Gym Box, so seed that Box
            # explicitly for reproducible random warmup actions.
            env.action_space.seed(int(seed))
            return env
        return current_factory(
            env_id=env_id,
            seed=seed,
            normalize_obs=normalize_obs,
            obs_norm_clip=obs_norm_clip,
        )

    seeded_factory._swimmer_process_local_patch = True  # type: ignore[attr-defined]
    mujoco_velocity.make_velocity_env = seeded_factory

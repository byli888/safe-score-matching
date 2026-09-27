"""Safety-Gymnasium velocity environments for SSM training.

Wrapper settings cover HalfCheetah v0/v1; swimmer_velocity_bootstrap adds
Swimmer v1. These tasks use 1000-step time limits without early termination.
The signed safety scalar is h = x_velocity - v_threshold, and the binary cost
is 1[x_velocity > v_threshold], checked against the environment's native cost.
The learner applies h_hardgap; this wrapper keeps the unshifted safety scalar.

Observations use Welford normalization with statistics that can be copied and
frozen for evaluation. Actions are mapped from [-1, 1] to the environment box.
"""
from typing import Tuple

import gym
import gym.spaces
import numpy as np
import safety_gymnasium


# ============================================================
# Supported envs registry
# ============================================================
# HalfCheetah velocity v0/v1 never terminate on unhealthy state; episodes end
# by time-limit truncation, which keeps HJ Q_h bootstrap semantics clean.
# Tasks with unhealthy-state termination require different bootstrap semantics.
# Swimmer is added to this registry by swimmer_velocity_bootstrap.install().
VELOCITY_ENV_CONFIG = {
    "SafetyHalfCheetahVelocity-v0": {
        "v_threshold": 2.8795,
        # Episode length is fixed by the env; kept here for documentation
        # and for wandb config logging.
        "episode_length": 1000,
        "terminates": False,
    },
    "SafetyHalfCheetahVelocity-v1": {
        "v_threshold": 3.2096,
        # Episode length is fixed by the env; kept here for documentation
        # and for wandb config logging.
        "episode_length": 1000,
        "terminates": False,
    },
}


# ============================================================
# Env wrapper
# ============================================================
class VelocityEnvWrapper:
    """Wraps a Safety-Gymnasium velocity env for SSM training.

    Responsibilities:
      1. Compute continuous SDF h = x_velocity - v_threshold (signed).
      2. Compute matching binary cost = float(x_velocity > v_threshold).
      3. Welford running obs normalization (train-mode updates stats,
         eval-mode freezes them).
      4. Action rescale from policy space [-1, 1] to raw env bounds.
      5. Per-episode reward, cost, and velocity accumulators.

    step() returns:
        (obs, reward, h_val, binary_cost, done, info)
    """

    def __init__(
        self,
        env_id: str = "SafetyHalfCheetahVelocity-v0",
        seed: int = 0,
        normalize_obs: bool = True,
        obs_norm_clip: float = 5.0,
    ):
        if env_id not in VELOCITY_ENV_CONFIG:
            raise ValueError(
                f"VelocityEnvWrapper: env_id={env_id!r} not in registry. "
                f"Supported: {list(VELOCITY_ENV_CONFIG.keys())}"
            )
        self.env_id = env_id
        cfg = VELOCITY_ENV_CONFIG[env_id]
        self.v_threshold: float = float(cfg["v_threshold"])
        self.episode_length_cfg: int = int(cfg["episode_length"])

        self.env = safety_gymnasium.make(env_id)
        self._seed = int(seed)

        # ---- Episode accumulators ----
        self._reset_episode_accumulators()

        # ---- Welford running obs stats ----
        self._normalize_obs = normalize_obs
        self._obs_norm_clip = float(obs_norm_clip)
        raw_dim = self.env.observation_space.shape[0]
        self._obs_mean = np.zeros(raw_dim, dtype=np.float64)
        self._obs_var = np.ones(raw_dim, dtype=np.float64)
        self._obs_count = 0
        self._update_stats = True  # set False during eval

        # ---- Observation space (post-normalization) ----
        if normalize_obs:
            low = -obs_norm_clip * np.ones(raw_dim, dtype=np.float32)
            high = obs_norm_clip * np.ones(raw_dim, dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=np.float32
            )
        else:
            raw_space = self.env.observation_space
            self.observation_space = gym.spaces.Box(
                low=raw_space.low.astype(np.float32),
                high=raw_space.high.astype(np.float32),
                shape=raw_space.shape,
                dtype=np.float32,
            )

        # ---- Action space: policy outputs in [-1, 1], rescaled internally ----
        raw_act = self.env.action_space
        self.action_space = gym.spaces.Box(
            low=-np.ones(raw_act.shape, dtype=np.float32),
            high=np.ones(raw_act.shape, dtype=np.float32),
            dtype=np.float32,
        )
        self._act_low = raw_act.low.astype(np.float32)
        self._act_high = raw_act.high.astype(np.float32)

    # --------------------------------------------------------
    # Episode accumulators
    # --------------------------------------------------------
    def _reset_episode_accumulators(self) -> None:
        self._ep_reward = 0.0
        self._ep_cost_binary = 0.0      # sum_t 1[v_t > v_thr]
        self._ep_sdf_violation = 0.0    # sum_t max(0, h_sdf_t)
        self._ep_length = 0
        self._ep_v_sum = 0.0            # sum_t v_t (signed)
        self._ep_v_max = -np.inf        # max_t v_t (signed)

    # --------------------------------------------------------
    # Welford observation normalization
    # --------------------------------------------------------
    def _update_obs_running_stats(self, obs: np.ndarray) -> None:
        if not self._update_stats:
            return
        self._obs_count += 1
        delta = obs.astype(np.float64) - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs.astype(np.float64) - self._obs_mean
        self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self._normalize_obs:
            return obs.astype(np.float32)
        self._update_obs_running_stats(obs)
        std = np.sqrt(self._obs_var + 1e-8)
        normed = (obs.astype(np.float64) - self._obs_mean) / std
        return np.clip(
            normed, -self._obs_norm_clip, self._obs_norm_clip
        ).astype(np.float32)

    def freeze_obs_stats(self) -> None:
        """Freeze observation statistics after calibration."""
        self._update_stats = False

    def set_train_mode(self) -> None:
        """Resume updating obs running stats."""
        self._update_stats = True

    def get_obs_stats(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Snapshot current running stats (for eval env sync)."""
        return self._obs_mean.copy(), self._obs_var.copy(), self._obs_count

    def set_obs_stats(
        self, mean: np.ndarray, var: np.ndarray, count: int
    ) -> None:
        """Install running stats from another wrapper (train → eval sync)."""
        self._obs_mean = mean.copy()
        self._obs_var = var.copy()
        self._obs_count = int(count)

    # --------------------------------------------------------
    # Action rescale
    # --------------------------------------------------------
    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        """Map policy action in [-1, 1] to raw env action range."""
        a = np.asarray(action, dtype=np.float32)
        return self._act_low + (a + 1.0) * 0.5 * (self._act_high - self._act_low)

    # --------------------------------------------------------
    # Gym API
    # --------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset env and return normalized initial observation.

        Handles both gymnasium 2-tuple (obs, info) and legacy obs-only
        return formats defensively.
        """
        out = self.env.reset(seed=self._seed)
        self._seed += 1
        if isinstance(out, tuple):
            obs = out[0]
        else:
            obs = out
        self._reset_episode_accumulators()
        return self._norm_obs(obs)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float, bool, dict]:
        """Execute one env step.

        Returns:
            obs:         normalized next observation (np.float32)
            reward:      scalar reward
            h_val:       SIGNED SDF = x_velocity - v_threshold
                         (stored as the replay safety scalar)
            binary_cost: float(x_velocity > v_threshold)
            done:        terminated or truncated
            info:        dict with x_velocity, cost_binary, h_sdf,
                         v_threshold, episode_*, terminated, truncated
        """
        real_action = self._rescale_action(action)
        out = self.env.step(real_action)

        # Safety-Gymnasium 6-tuple: (obs, reward, cost, terminated, truncated, info)
        # Plain gymnasium 5-tuple: (obs, reward, terminated, truncated, info)
        # Both safety quantities come from x_velocity; the native binary cost
        # is checked below to detect an incompatible environment threshold.
        if len(out) == 6:
            obs, reward, _native_cost, terminated, truncated, info = out
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            _native_cost = None
        else:
            raise RuntimeError(
                f"VelocityEnvWrapper: unexpected step return arity {len(out)}. "
                f"Expected 5 (gymnasium) or 6 (safety-gymnasium)."
            )

        # ---- Single source of truth: compute both costs from x_velocity ----
        if "x_velocity" not in info:
            raise RuntimeError(
                "VelocityEnvWrapper requires info['x_velocity'] from the env. "
                f"env={self.env_id} info keys: {sorted(info.keys())}"
            )
        v = float(info["x_velocity"])
        h_sdf = v - self.v_threshold
        binary_cost = float(v > self.v_threshold)

        # The recomputed binary cost must agree with the environment cost.
        if _native_cost is not None:
            if abs(float(_native_cost) - binary_cost) > 1e-9:
                raise RuntimeError(
                    f"VelocityEnvWrapper: env native cost {float(_native_cost)!r} "
                    f"disagrees with computed binary cost {binary_cost!r} at "
                    f"x_velocity={v}, v_threshold={self.v_threshold}. "
                    f"This breaks SSM vs SACLag cost comparability — fix the "
                    f"env version or v_threshold before continuing."
                )

        # ---- Episode-level accumulators ----
        self._ep_reward += float(reward)
        self._ep_cost_binary += binary_cost
        self._ep_sdf_violation += max(0.0, h_sdf)
        self._ep_length += 1
        self._ep_v_sum += v
        if v > self._ep_v_max:
            self._ep_v_max = v

        # ---- Surface both cost signals + episode stats via info ----
        info["cost_binary"] = binary_cost
        info["h_sdf"] = h_sdf
        info["v_threshold"] = self.v_threshold
        info["episode_reward"] = self._ep_reward
        info["episode_cost_binary"] = self._ep_cost_binary
        info["episode_sdf_violation"] = self._ep_sdf_violation
        info["episode_length"] = self._ep_length
        info["episode_avg_velocity"] = self._ep_v_sum / max(1, self._ep_length)
        info["episode_max_velocity"] = (
            self._ep_v_max if self._ep_length > 0 else 0.0
        )
        # Backward-compat alias: legacy train loops read `episode_cost`,
        # which for velocity tasks maps to the binary cost sum.
        info["episode_cost"] = info["episode_cost_binary"]
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)

        done = bool(terminated or truncated)
        obs = self._norm_obs(obs)

        return obs, float(reward), float(h_sdf), binary_cost, done, info

    # --------------------------------------------------------
    # Episode info snapshot (read by train loop at done=True)
    # --------------------------------------------------------
    @property
    def episode_info(self) -> dict:
        return {
            "reward": self._ep_reward,
            "cost_binary": self._ep_cost_binary,
            # Backward-compat alias for legacy train loops that read `cost`.
            "cost": self._ep_cost_binary,
            "sdf_violation": self._ep_sdf_violation,
            "length": self._ep_length,
            "avg_velocity": self._ep_v_sum / max(1, self._ep_length),
            "max_velocity": (
                self._ep_v_max if self._ep_length > 0 else 0.0
            ),
        }


# ============================================================
# Factory
# ============================================================
def make_velocity_env(
    env_id: str = "SafetyHalfCheetahVelocity-v0",
    seed: int = 0,
    normalize_obs: bool = True,
    obs_norm_clip: float = 5.0,
) -> VelocityEnvWrapper:
    """Create a normalized velocity environment with policy-space actions."""
    return VelocityEnvWrapper(
        env_id=env_id,
        seed=seed,
        normalize_obs=normalize_obs,
        obs_norm_clip=obs_norm_clip,
    )

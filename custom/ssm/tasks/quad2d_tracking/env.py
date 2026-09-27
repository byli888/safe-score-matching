"""
quad2d_env.py — Self-contained 2D Quadrotor environment for SSM testing.

Based on RCRL (Yu et al. 2022) Sec 6.2 / reference implementation SafeScoreMatching repo.
Adapted for our SSM agent: continuous SDF cost, old-gym-compatible interface.

Environment:
  - 2D planar quadrotor tracking a circular trajectory
  - State: [x, ẋ, z, ż, θ, θ̇]  (6D)
  - Observation: [state, reference_waypoint]  (12D)
  - Action: [T1, T2] ∈ [-1, 1]^2 (symmetric, rescaled internally to [0,1])
  - Reward: -quadratic tracking error (dense, negative)
  - Safety: altitude constraint 0.5 ≤ z ≤ 1.5
  - Cost (SDF): h(s) = max(0.5 - z, z - 1.5), negative = safe
  - Episode: 360 steps, dt=1/60
  - Termination: |x| > 2 or |z| > 3
"""
import numpy as np
import gym
import gym.spaces


# ============================================================
# Dynamics (pure numpy, no JAX dependency for env stepping)
# ============================================================
def quad2d_step(state, action_raw, dt, m=1.0, I=0.02, g=9.81,
                thrust_scale=None, torque_scale=0.1):
    """Semi-implicit Euler integration for planar quadrotor.

    Args:
        state: [x, ẋ, z, ż, θ, θ̇]
        action_raw: [T1, T2] in [0, 1] (normalized thrust)
    """
    if thrust_scale is None:
        thrust_scale = m * g

    a = np.clip(action_raw, 0.0, 1.0)
    x, xdot, z, zdot, theta, thetadot = state

    F1 = thrust_scale * a[0]
    F2 = thrust_scale * a[1]
    u1 = F1 + F2
    tau = torque_scale * (a[1] - a[0])

    xddot = -(u1 / m) * np.sin(theta)
    zddot = (u1 / m) * np.cos(theta) - g
    thetaddot = tau / I

    # Semi-implicit Euler (symplectic)
    xdot = xdot + xddot * dt
    x = x + xdot * dt
    zdot = zdot + zddot * dt
    z = z + zdot * dt
    thetadot = thetadot + thetaddot * dt
    theta = theta + thetadot * dt

    return np.array([x, xdot, z, zdot, theta, thetadot], dtype=np.float32)


# ============================================================
# Environment
# ============================================================
class Quad2DEnv:
    """2D Quadrotor tracking environment with continuous SDF cost.

    Compatible with our SSM training loop (old gym-style interface).
    """

    def __init__(self, dt=1/60, max_episode_steps=360, seed=0,
                 ref_velocity=True, q_vel=5.0, action_penalty=1e-2,
                 sdf_mode="baseline"):
        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # SDF mode: controls _sdf() output shape (cost signal for critic)
        # All modes preserve sign(h) and the h=0 boundary, so binary_cost
        # and termination logic are unaffected.
        #
        #   baseline     : original max-SDF (raw geometry, continuous everywhere)
        #   hard1        : safe side baseline; unsafe side hard-replaced to +1.0
        #   hard5        : safe side baseline; unsafe side hard-replaced to +5.0
        #   hard10       : safe side baseline; unsafe side hard-replaced to +10.0
        #   hard5_scale10: RAA-style. Safe unchanged, unsafe → const 5, then
        #                  global ×10 → safe range [-5, 0], unsafe = 50.
        #                  Note: hard5 → hard5_scale10 is a single-variable test
        #                  of "global scale alone" (same const, same ratio).
        valid_modes = {"baseline", "hard1", "hard5", "hard10", "hard5_scale10"}
        if sdf_mode not in valid_modes:
            raise ValueError(f"Unknown sdf_mode: {sdf_mode!r}. Valid: {sorted(valid_modes)}")
        self._sdf_mode = sdf_mode

        # Physics
        self.m = 1.0
        self.I = 0.02
        self.g = 9.81
        self.thrust_scale = self.m * self.g
        self.torque_scale = 0.1

        # Tracking reference: circle centered at (0, 1), radius 1
        self.center = (0.0, 1.0)
        self.radius = 1.0
        self.ref_velocity = ref_velocity  # True = RCRL-style moving ref with tangent velocity

        # Reward weights
        # Q = diag(Q_pos, q_vel, Q_pos, q_vel, Q_angle, Q_angle)
        # RCRL paper: q_vel=1, action_penalty=1e-4
        # Our default: q_vel=5, action_penalty=1e-2
        self.Q = np.diag([10.0, q_vel, 10.0, q_vel, 0.2, 0.2]).astype(np.float32)
        self.R = np.diag([action_penalty, action_penalty]).astype(np.float32)
        self.a_ref = np.array([0.5, 0.5], dtype=np.float32)  # hover

        # Spaces (action in [-1, 1], rescaled to [0, 1] internally)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32), dtype=np.float32)

        # Safety constraint: 0.5 ≤ z ≤ 1.5
        self.z_low = 0.5
        self.z_high = 1.5

        # State
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(6, dtype=np.float32)
        self._waypoints = self._create_waypoints()
        self._waypoint_idx = 0
        self._t = 0

        # Episode tracking
        self._episode_reward = 0.0
        self._episode_cost = 0.0  # cumulative binary violations
        self._episode_length = 0

        # Obs normalization (Welford)
        self._normalize_obs = True
        self._obs_norm_clip = 5.0
        self._obs_mean = np.zeros(12, dtype=np.float64)
        self._obs_var = np.ones(12, dtype=np.float64)
        self._obs_count = 0
        self._update_stats = True

    def _create_waypoints(self):
        """360 waypoints on a unit circle (1 degree each).

        Each waypoint: [x, ẋ, z, ż, θ, θ̇].
        If ref_velocity=True (default), ẋ and ż are the circle's tangent
        velocity at that point, matching RCRL's reference trajectory.
        """
        angles = np.deg2rad(np.arange(0, 360))
        xs = self.center[0] + self.radius * np.cos(angles)
        zs = self.center[1] + self.radius * np.sin(angles)
        zeros = np.zeros_like(xs)

        if self.ref_velocity:
            # Angular velocity: one full circle in max_episode_steps * dt seconds
            omega = 2.0 * np.pi / (len(angles) * self.dt)  # π/3 rad/s
            x_dots = -self.radius * omega * np.sin(angles)
            z_dots =  self.radius * omega * np.cos(angles)
        else:
            x_dots = zeros
            z_dots = zeros

        waypoints = np.stack([xs, x_dots, zs, z_dots, zeros, zeros], axis=1)
        return waypoints.astype(np.float32)

    def _nearest_waypoint_idx(self, x, z):
        deltas = self._waypoints[:, [0, 2]] - np.array([x, z], dtype=np.float32)
        return int(np.argmin(np.sum(np.square(deltas), axis=1)))

    def _get_obs(self):
        ref = self._waypoints[self._waypoint_idx]
        obs_raw = np.concatenate([self.state, ref]).astype(np.float32)
        return self._norm_obs(obs_raw)

    def _sdf(self, x, z):
        """Signed distance function: altitude + OOB boundaries.
        Matches reference implementation cost = float((h > 0) or out_of_bounds).
        h < 0 → safe, h > 0 → violation.

        SDF mode dispatch (sign(h) and h=0 boundary preserved in all modes):
            baseline      : raw max-SDF
            hard1         : unsafe → +1.0,  safe → raw
            hard5         : unsafe → +5.0,  safe → raw
            hard10        : unsafe → +10.0, safe → raw
            hard5_scale10 : RAA-style. (where(h>0, 5, h_raw)) * 10
                            = unsafe → +50.0, safe → raw * 10
                            (hard5 with global ×10 — clean scale-only ablation)
        """
        h_altitude = max(self.z_low - z, z - self.z_high)  # 0.5 ≤ z ≤ 1.5
        h_lateral = abs(x) - 2.0                            # |x| ≤ 2
        h_vertical_oob = abs(z) - 3.0                       # |z| ≤ 3
        h_raw = max(h_altitude, h_lateral, h_vertical_oob)

        mode = self._sdf_mode
        if mode == "baseline":
            return h_raw
        if mode == "hard1":
            return 1.0 if h_raw > 0 else h_raw
        if mode == "hard5":
            return 5.0 if h_raw > 0 else h_raw
        if mode == "hard10":
            return 10.0 if h_raw > 0 else h_raw
        if mode == "hard5_scale10":
            # RAA-style: where(h_raw > 0, 5, h_raw) * 10
            base = 5.0 if h_raw > 0 else h_raw
            return base * 10.0
        # Unreachable: validated in __init__
        raise ValueError(f"Unknown sdf_mode: {mode}")

    # ---- Obs normalization ----
    def _update_obs_running_stats(self, obs):
        if not self._update_stats:
            return
        self._obs_count += 1
        delta = obs.astype(np.float64) - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs.astype(np.float64) - self._obs_mean
        self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count

    def _norm_obs(self, obs):
        if not self._normalize_obs:
            return obs
        self._update_obs_running_stats(obs)
        std = np.sqrt(self._obs_var + 1e-8)
        normed = (obs.astype(np.float64) - self._obs_mean) / std
        return np.clip(normed, -self._obs_norm_clip, self._obs_norm_clip).astype(np.float32)

    def set_eval_mode(self):
        self._update_stats = False

    def set_train_mode(self):
        self._update_stats = True

    def get_obs_stats(self):
        return self._obs_mean.copy(), self._obs_var.copy(), self._obs_count

    def set_obs_stats(self, mean, var, count):
        self._obs_mean = mean.copy()
        self._obs_var = var.copy()
        self._obs_count = count

    # ---- Core API ----
    def reset(self, options=None):
        low = np.array([-1.5, -1.0, 0.25, -1.5, -0.2, -0.1], dtype=np.float32)
        high = np.array([1.5, 1.0, 1.75, 1.5, 0.2, 0.1], dtype=np.float32)
        self.state = self._rng.uniform(low, high).astype(np.float32)

        # Allow overriding initial state
        if options:
            if "init_x" in options: self.state[0] = float(options["init_x"])
            if "init_vx" in options: self.state[1] = float(options["init_vx"])
            if "init_z" in options: self.state[2] = float(options["init_z"])
            if "init_vz" in options: self.state[3] = float(options["init_vz"])
            if "init_theta" in options: self.state[4] = float(options["init_theta"])
            if "init_omega" in options: self.state[5] = float(options["init_omega"])

        self._t = 0
        self._waypoint_idx = self._nearest_waypoint_idx(self.state[0], self.state[2])
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0

        return self._get_obs()

    def step(self, action):
        """
        Args:
            action: [T1, T2] in [-1, 1] (agent output)

        Returns:
            obs, reward, h_val (SDF), binary_cost, done, info
            (matches our SafeGymWrapper interface)
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Rescale [-1, 1] → [0, 1] for thrust
        action_raw = (action + 1.0) / 2.0

        # Step dynamics
        self.state = quad2d_step(
            self.state, action_raw, self.dt,
            self.m, self.I, self.g, self.thrust_scale, self.torque_scale)

        # Advance waypoint
        self._waypoint_idx = (self._waypoint_idx + 1) % len(self._waypoints)

        # Reward: -quadratic tracking error
        ref = self._waypoints[self._waypoint_idx]
        state_err = self.state - ref
        action_err = action_raw - self.a_ref
        reward = -float(state_err @ self.Q @ state_err + action_err @ self.R @ action_err)

        # Safety
        z = float(self.state[2])
        x = float(self.state[0])
        h_val = self._sdf(x, z)  # continuous SDF including OOB
        out_of_bounds = (abs(x) > 2.0) or (abs(z) > 3.0)
        binary_cost = float(h_val > 0)  # now includes OOB (matches reference)

        # Termination
        terminated = bool(out_of_bounds)
        self._t += 1
        truncated = bool(self._t >= self.max_episode_steps)
        done = terminated or truncated

        # Episode stats
        self._episode_reward += reward
        self._episode_cost += binary_cost
        self._episode_length += 1

        info = {
            'h': h_val,
            'binary_cost': binary_cost,
            'z': z,
            'x': x,
            'terminated': terminated,
            'truncated': truncated,
            'episode_reward': self._episode_reward,
            'episode_cost': self._episode_cost,
            'episode_length': self._episode_length,
        }

        obs = self._get_obs()
        return obs, reward, h_val, binary_cost, done, info

    @property
    def episode_info(self):
        return {
            'reward': self._episode_reward,
            'cost': self._episode_cost,
            'length': self._episode_length,
        }
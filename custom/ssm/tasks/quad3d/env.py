"""
quad3d_env.py — 3D Quadrotor environment for SSM, ported from Dawson et al. (CoRL 2021).

Dynamics from neural_clbf/systems/quad3d.py (control-affine, forward Euler).
Safety/goal definitions faithful to Dawson's safe_mask / unsafe_mask / goal_mask.
RL components (reward, termination, reset) added for online safe RL training.

Environment:
  - State: [px, py, pz, vx, vy, vz, phi, theta, psi]  (9D)
  - Z positive DOWNWARD (Dawson code convention):
      pz < 0 = in the air (safe), pz > 0 = below ground (unsafe)
  - Observation: raw 9D state (goal is fixed origin, not appended)
  - Action: [-1, 1]^4, remapped internally:
      thrust = m*g*(1 + a[0]) in [0, 2*m*g]  (hover at a[0]=0)
      phi_dot, theta_dot, psi_dot = 5 * a[1:4] in [-5, 5] rad/s
  - Dynamics: dx/dt = f(x) + g(x)*u, forward Euler (dt=0.01)

Safety (Dawson safe_mask, using full 9D state norm):
  - Safe: pz <= 0 AND ||x_9D|| <= 3
  - SDF: h(s) = max(pz, ||x_9D|| - 3.0);  h < 0 safe, h > 0 unsafe
  - binary_cost = 1[h > 0]

Termination (Dawson unsafe_mask, deeper failure):
  - terminated: pz >= 0.3 OR ||x_9D|| >= 3.5
  - truncated: max_episode_steps (default 500 = 5 seconds)

Goal (Dawson goal_mask, kept as diagnostic only):
  - ||x_9D|| <= 0.3 AND safe(x)
  - Used for the `episode_dawson_goal_steps` diagnostic, NOT for reward.

Reward (NEW dense regulation reward, default):
  - `reward_norm='l2'` (default):
      r_t = -(x_t - x_ref)^T Q (x_t - x_ref) - a_t^T R a_t
  - `reward_norm='l1'`:
      r_t = -sum_i Q_diag[i] * |x_t[i] - x_ref[i]| - action_penalty * sum_j |a_t[j]|
  - x_ref defaults to origin (all 9 dims = 0). Only the pz dimension can be
    moved off origin via target_pz; all other 8 dims are fixed at 0.
  - Shared-Q default:
      Q_diag = [q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_ang, q_ang, q_psi]
  - Axis-split optional mode:
      Q_diag = [q_pos_xy, q_pos_xy, q_pos_z, q_vel_xy, q_vel_xy, q_vel_z, q_ang, q_ang, q_psi]
    with q_psi defaulting to q_ang when not provided.
    Defaults: q_pos=10, q_vel=1, q_ang=0.2.
  - R = action_penalty * I_4 (default action_penalty=1e-3).
  - IMPORTANT: a is the NORMALIZED agent action ∈ [-1, 1]^4, NOT physical
    control. a = 0 already corresponds to u_eq via _remap_action, so we
    penalize ||a||² directly (no `u - u_eq` subtraction).

Reward (LEGACY sparse Dawson-goal-set, opt-in via legacy_reward=True):
  - r = -(max(||x_9D|| - 0.3, 0))^2 - (u_physical - u_eq)^T R (u_physical - u_eq)
  - Kept for ablation and for backward compatibility with old checkpoints.

References:
  - Dawson et al. "Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier
    Functions" CoRL 2021, Appendix (3D Quadrotor section)
  - quad3d.py in https://github.com/MIT-REALM/neural_clbf
"""
import numpy as np
import gym
import gym.spaces

from ssm.common.sdf import shape_sdf

OBS_FEATURE_MODES = (
    "state",
    "h",
    "h_components",
    "h_components_closing",
    "h_components_closing_reach",
)


# ============================================================
# Physics constants (matching Dawson)
# ============================================================
GRAV = 9.80665  # Dawson's utils.py


# ============================================================
# Dynamics (pure numpy, no JAX/PyTorch dependency)
# ============================================================
def quad3d_xdot(state, u, m=1.0):
    """Compute dx/dt = f(x) + g(x) * u for the 3D quadrotor.

    Dawson's control-affine dynamics (Z positive downward):
        f(x) = [vx, vy, vz, 0, 0, +g, 0, 0, 0]
        g(x) = see quad3d.py _g() method

    Args:
        state: [px, py, pz, vx, vy, vz, phi, theta, psi]  (9,)
        u: [thrust, phi_dot, theta_dot, psi_dot]  (4,) in physical units
        m: mass (default 1.0)

    Returns:
        xdot: (9,) state derivative
    """
    px, py, pz, vx, vy, vz, phi, theta, psi = state
    f_thrust, phi_dot, theta_dot, psi_dot = u

    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    s_phi = np.sin(phi)
    c_phi = np.cos(phi)

    # f(x): drift
    # Positions: velocity
    # Velocities: only gravity in vz (+g because Z is down)
    # Angles: zero (directly actuated via g)

    # g(x)*u: control input
    # vx_dot += -(1/m) * sin(theta) * thrust
    # vy_dot += (1/m) * cos(theta) * sin(phi) * thrust
    # vz_dot += -(1/m) * cos(theta) * cos(phi) * thrust  (+ grav from drift)
    # phi_dot, theta_dot, psi_dot = direct from u[1:4]

    xdot = np.zeros(9, dtype=np.float64)
    xdot[0] = vx                                           # px_dot
    xdot[1] = vy                                           # py_dot
    xdot[2] = vz                                           # pz_dot
    xdot[3] = -(f_thrust / m) * s_theta                    # vx_dot
    xdot[4] = (f_thrust / m) * c_theta * s_phi             # vy_dot
    xdot[5] = GRAV - (f_thrust / m) * c_theta * c_phi      # vz_dot
    xdot[6] = phi_dot                                       # phi_dot
    xdot[7] = theta_dot                                     # theta_dot
    xdot[8] = psi_dot                                       # psi_dot

    return xdot


def quad3d_step(state, u, dt, m=1.0):
    """Forward Euler integration (matching Dawson's zero_order_hold).

    Args:
        state: (9,) current state
        u: (4,) physical control [thrust, phi_dot, theta_dot, psi_dot]
        dt: timestep
        m: mass

    Returns:
        next_state: (9,) float32
    """
    xdot = quad3d_xdot(state, u, m)
    next_state = state + dt * xdot
    return next_state.astype(np.float32)


# ============================================================
# Environment
# ============================================================
class Quad3DEnv:
    """3D Quadrotor stabilization environment with continuous SDF cost.

    Faithful to Dawson's Quad3D system definition.
    Compatible with our SSM training loop (old gym-style interface).
    """

    # State indices (matching Dawson)
    PX, PY, PZ = 0, 1, 2
    VX, VY, VZ = 3, 4, 5
    PHI, THETA, PSI = 6, 7, 8

    N_DIMS = 9
    N_CONTROLS = 4

    def __init__(self, dt=0.01, max_episode_steps=500, seed=0,
                 action_penalty=1e-3,
                 q_pos=10.0,
                 q_vel=1.0,
                 q_pos_xy=None,
                 q_pos_z=None,
                 q_vel_xy=None,
                 q_vel_z=None,
                 q_ang=0.2,
                 q_psi=None,
                 reward_norm='l2',
                 target_pz=0.0,
                 legacy_reward=False,
                 sdf_mode='raw',
                 sdf_unsafe_const=0.0,
                 sdf_unsafe_slope=0.0,
                 sdf_global_scale=0.0,
                 init_curriculum=False,
                 init_curriculum_mode='outer_xy',
                 init_curriculum_frac=0.3,
                 init_curriculum_pos_xy_min=1.4,
                 init_curriculum_schedule='constant',
                 init_curriculum_anneal_start=0,
                 init_curriculum_anneal_end=0,
                 init_curriculum_anneal_end_frac=0.0,
                 obs_feature_mode='state',
                 obs_h_scale=1.0,
                 obs_hdot_scale=1.0,
                 obs_reach_scale=1.0,
                 obs_goal_pos_radius=0.30,
                 obs_goal_vel_radius=0.30,
                 obs_goal_angle_radius=0.20):
        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # Physics (Dawson: only parameter is mass)
        self.m = 1.0

        # Action remap parameters
        # External: [-1, 1]^4
        # Internal: thrust = mg*(1 + a[0]) in [0, 2mg],  rates = 5*a[1:4]
        self.thrust_scale = self.m * GRAV   # mg
        self.rate_scale = 5.0               # rad/s

        # Equilibrium control (hover at origin)
        self.u_eq = np.array([self.m * GRAV, 0.0, 0.0, 0.0], dtype=np.float32)

        # ---- Reward configuration ----
        # Dawson goal radius — kept as diagnostic only (NOT used by reward in new mode).
        self.goal_radius = 0.3

        # Reference state for new dense quadratic reward.
        # Default: origin (all 9 dims = 0). Only pz can be configured via target_pz;
        # all other 8 dims are always 0 by design.
        self.x_ref = np.zeros(self.N_DIMS, dtype=np.float32)
        self.x_ref[self.PZ] = float(target_pz)
        self.target_pz = float(target_pz)

        # Q matrix: per-dim weighting (px, py, pz | vx, vy, vz | phi, theta, psi)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.q_ang = float(q_ang)
        self.q_psi = float(q_psi) if q_psi is not None else float(q_ang)
        split_values = {
            'q_pos_xy': q_pos_xy,
            'q_pos_z': q_pos_z,
            'q_vel_xy': q_vel_xy,
            'q_vel_z': q_vel_z,
        }
        provided_split = {k: v for k, v in split_values.items() if v is not None}
        if provided_split and len(provided_split) != 4:
            raise ValueError(
                "Axis-split Q requires q_pos_xy, q_pos_z, q_vel_xy, and q_vel_z together. "
                f"Got partial split args: {sorted(provided_split.keys())}"
            )
        self.uses_axis_split_q = len(provided_split) == 4
        if self.uses_axis_split_q:
            self.q_pos_xy = float(q_pos_xy)
            self.q_pos_z = float(q_pos_z)
            self.q_vel_xy = float(q_vel_xy)
            self.q_vel_z = float(q_vel_z)
        else:
            self.q_pos_xy = self.q_pos
            self.q_pos_z = self.q_pos
            self.q_vel_xy = self.q_vel
            self.q_vel_z = self.q_vel

        self.Q_diag = np.array([self.q_pos_xy, self.q_pos_xy, self.q_pos_z,
                                self.q_vel_xy, self.q_vel_xy, self.q_vel_z,
                                q_ang, q_ang, self.q_psi], dtype=np.float32)
        self.Q = np.diag(self.Q_diag).astype(np.float32)

        # Reward norm switch: keep default L2 behavior, allow weighted L1 probes.
        self.reward_norm = str(reward_norm).lower()
        assert self.reward_norm in ('l1', 'l2'), (
            f"reward_norm must be 'l1' or 'l2', got {reward_norm!r}"
        )

        # R matrix: action penalty (applied to NORMALIZED action a, not (u-u_eq))
        self.action_penalty = float(action_penalty)
        self.R = np.diag([action_penalty] * self.N_CONTROLS).astype(np.float32)

        # Reward mode switch
        self.legacy_reward = bool(legacy_reward)

        # Safety thresholds (Dawson)
        self.safe_pz = 0.0           # safe_mask: pz <= 0
        self.safe_radius = 3.0       # safe_mask: ||x|| <= 3
        self.unsafe_pz = 0.3         # unsafe_mask: pz >= 0.3
        self.unsafe_radius = 3.5     # unsafe_mask: ||x|| >= 3.5

        # ---- Observation safety/reach features ----
        self.obs_feature_mode = str(obs_feature_mode)
        if self.obs_feature_mode not in OBS_FEATURE_MODES:
            raise ValueError(
                f"obs_feature_mode must be one of {OBS_FEATURE_MODES}, "
                f"got {obs_feature_mode!r}"
            )
        self.obs_h_scale = float(obs_h_scale)
        self.obs_hdot_scale = float(obs_hdot_scale)
        self.obs_reach_scale = float(obs_reach_scale)
        self.obs_goal_pos_radius = float(obs_goal_pos_radius)
        self.obs_goal_vel_radius = float(obs_goal_vel_radius)
        self.obs_goal_angle_radius = float(obs_goal_angle_radius)
        if self.obs_h_scale <= 0.0:
            raise ValueError("obs_h_scale must be > 0.")
        if self.obs_hdot_scale <= 0.0:
            raise ValueError("obs_hdot_scale must be > 0.")
        if self.obs_reach_scale <= 0.0:
            raise ValueError("obs_reach_scale must be > 0.")
        if self.obs_goal_pos_radius <= 0.0:
            raise ValueError("obs_goal_pos_radius must be > 0.")
        if self.obs_goal_vel_radius <= 0.0:
            raise ValueError("obs_goal_vel_radius must be > 0.")
        if self.obs_goal_angle_radius <= 0.0:
            raise ValueError("obs_goal_angle_radius must be > 0.")
        self.obs_feature_dim = self._obs_feature_dim_for_mode(self.obs_feature_mode)

        # ---- SDF shaping ----
        # All modes preserve sign(h) and the h=0 boundary, so binary_cost and
        # termination (based on Dawson unsafe_mask, not on h) remain unaffected.
        # Only the value of h_val returned to the agent (and used to train Q_h)
        # is shaped. Defaults: raw passthrough (no-op).
        self.sdf_mode = str(sdf_mode)
        self.sdf_unsafe_const = float(sdf_unsafe_const)
        self.sdf_unsafe_slope = float(sdf_unsafe_slope)
        self.sdf_global_scale = float(sdf_global_scale)
        # Validate at construction time by doing a dry shape_sdf call on both
        # sides of the boundary; shape_sdf itself raises on invalid combos.
        _ = shape_sdf(
            -0.1, self.sdf_mode,
            unsafe_const=self.sdf_unsafe_const,
            unsafe_slope=self.sdf_unsafe_slope,
            global_scale=self.sdf_global_scale,
        )
        _ = shape_sdf(
            0.1, self.sdf_mode,
            unsafe_const=self.sdf_unsafe_const,
            unsafe_slope=self.sdf_unsafe_slope,
            global_scale=self.sdf_global_scale,
        )

        # ---- Reset curriculum (training-only by convention) ----
        # This is intentionally narrow: we only bias the initial xy position
        # toward the outer tail seen in Y20 crash diagnostics. We do NOT try to
        # construct states near the full 9D radius boundary, because under the
        # default reset bounds ||x|| never reaches the safe radius shell.
        self.init_curriculum = bool(init_curriculum)
        self.init_curriculum_mode = str(init_curriculum_mode)
        self.init_curriculum_frac = float(init_curriculum_frac)
        self.init_curriculum_pos_xy_min = float(init_curriculum_pos_xy_min)
        self.init_curriculum_schedule = str(init_curriculum_schedule)
        self.init_curriculum_anneal_start = int(init_curriculum_anneal_start)
        self.init_curriculum_anneal_end = int(init_curriculum_anneal_end)
        self.init_curriculum_anneal_end_frac = float(init_curriculum_anneal_end_frac)
        if not (0.0 <= self.init_curriculum_frac <= 1.0):
            raise ValueError(
                f"init_curriculum_frac must be in [0, 1], got {self.init_curriculum_frac}"
            )
        if not (0.0 <= self.init_curriculum_anneal_end_frac <= 1.0):
            raise ValueError(
                "init_curriculum_anneal_end_frac must be in [0, 1], got "
                f"{self.init_curriculum_anneal_end_frac}"
            )
        if self.init_curriculum_mode not in ('outer_xy',):
            raise ValueError(
                f"Unknown init_curriculum_mode: {self.init_curriculum_mode}"
            )
        if self.init_curriculum_schedule not in ('constant', 'linear_anneal'):
            raise ValueError(
                "Unknown init_curriculum_schedule: "
                f"{self.init_curriculum_schedule}"
            )
        if self.init_curriculum_pos_xy_min < 0.0:
            raise ValueError(
                "init_curriculum_pos_xy_min must be >= 0"
            )
        if self.init_curriculum_anneal_start < 0:
            raise ValueError(
                "init_curriculum_anneal_start must be >= 0"
            )
        if self.init_curriculum_anneal_end < self.init_curriculum_anneal_start:
            raise ValueError(
                "init_curriculum_anneal_end must be >= "
                "init_curriculum_anneal_start"
            )
        if (self.init_curriculum_schedule == 'linear_anneal'
                and self.init_curriculum_anneal_end
                == self.init_curriculum_anneal_start):
            raise ValueError(
                "linear_anneal requires init_curriculum_anneal_end > "
                "init_curriculum_anneal_start"
            )

        self._reset_low = np.array(
            [-1.5, -1.5, -1.0, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
            dtype=np.float32,
        )
        self._reset_high = np.array(
            [1.5, 1.5, -0.05, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2],
            dtype=np.float32,
        )
        self._reset_pos_xy_max = float(np.linalg.norm(self._reset_high[:2]))
        if (self.init_curriculum
                and self.init_curriculum_mode == 'outer_xy'
                and self.init_curriculum_pos_xy_min >= self._reset_pos_xy_max):
            raise ValueError(
                "init_curriculum_pos_xy_min must be strictly smaller than the "
                f"default reset outer-xy radius {self._reset_pos_xy_max:.3f}. "
                f"Got {self.init_curriculum_pos_xy_min:.3f}."
            )

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.N_DIMS + self.obs_feature_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.N_CONTROLS, dtype=np.float32),
            high=np.ones(self.N_CONTROLS, dtype=np.float32), dtype=np.float32)

        # State
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(self.N_DIMS, dtype=np.float32)
        self._t = 0
        self._train_step = 0

        # Episode tracking
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0
        self._episode_goal_steps = 0

        # Obs normalization (Welford, same as Quad2D)
        self._normalize_obs = True
        self._obs_norm_clip = 5.0
        obs_dim = self.observation_space.shape[0]
        self._obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self._obs_var = np.ones(obs_dim, dtype=np.float64)
        self._obs_count = 0
        self._update_stats = True

    @staticmethod
    def _obs_feature_dim_for_mode(mode):
        if mode == "state":
            return 0
        if mode == "h":
            return 1
        if mode == "h_components":
            return 3
        if mode == "h_components_closing":
            return 4
        if mode == "h_components_closing_reach":
            return 9
        raise ValueError(f"Unknown obs_feature_mode={mode!r}")

    # ---- Action remap ----
    def _remap_action(self, action_ext):
        """Map external [-1,1]^4 to physical control.

        thrust = mg * (1 + a[0])  =>  [0, 2*mg]  (hover at a[0]=0)
        phi_dot = 5 * a[1]       =>  [-5, 5] rad/s
        theta_dot = 5 * a[2]     =>  [-5, 5] rad/s
        psi_dot = 5 * a[3]       =>  [-5, 5] rad/s
        """
        a = np.clip(action_ext, -1.0, 1.0)
        u = np.zeros(self.N_CONTROLS, dtype=np.float64)
        u[0] = self.thrust_scale * (1.0 + a[0])  # [0, 2*mg]
        u[1] = self.rate_scale * a[1]
        u[2] = self.rate_scale * a[2]
        u[3] = self.rate_scale * a[3]
        return u

    # ---- Safety ----
    def _sdf(self, state):
        """Signed distance function (Dawson safe_mask boundary), optionally shaped.

        Raw:  h_raw(s) = max(pz, ||x_9D|| - 3.0)
        Returned:  h(s) = shape_sdf(h_raw, self.sdf_mode, ...)

        sign(h) and the h=0 boundary are preserved across all shaping modes,
        so binary_cost = 1{h > 0} and any 'h < 0' safety check behave the same
        whether shaping is active or not.

        Uses full 9D state norm, matching Dawson's x.norm(dim=-1).
        """
        pz = state[self.PZ]
        state_norm = np.linalg.norm(state)
        h_raw = max(pz - self.safe_pz, state_norm - self.safe_radius)
        if self.sdf_mode == 'raw':
            return float(h_raw)
        return shape_sdf(
            float(h_raw), self.sdf_mode,
            unsafe_const=self.sdf_unsafe_const,
            unsafe_slope=self.sdf_unsafe_slope,
            global_scale=self.sdf_global_scale,
        )

    def _raw_h_components(self, state):
        pz_margin = float(state[self.PZ] - self.safe_pz)
        radius_margin = float(np.linalg.norm(state) - self.safe_radius)
        h_margin = max(pz_margin, radius_margin)
        return h_margin, np.asarray([pz_margin, radius_margin], dtype=np.float32)

    def _h_dot_feature(self, state, h_margin, components):
        pz_margin, radius_margin = float(components[0]), float(components[1])
        if pz_margin >= radius_margin:
            return float(state[self.VZ])
        norm = float(np.linalg.norm(state))
        if norm <= 1e-8:
            return 0.0
        approx_state_dot = np.zeros_like(state, dtype=np.float32)
        approx_state_dot[0:3] = state[3:6]
        return float(np.dot(state, approx_state_dot) / norm)

    def _obs_features(self, state):
        mode = self.obs_feature_mode
        if mode == "state":
            return np.zeros((0,), dtype=np.float32)

        h_margin, components = self._raw_h_components(state)
        h_scaled = np.tanh(np.asarray([h_margin], dtype=np.float32) / self.obs_h_scale)
        if mode == "h":
            return h_scaled.astype(np.float32)

        comp_scaled = np.tanh(components.astype(np.float32) / self.obs_h_scale)
        features = np.concatenate([h_scaled, comp_scaled], axis=0)
        if mode in ("h_components_closing", "h_components_closing_reach"):
            hdot = self._h_dot_feature(state, h_margin, components)
            hdot_scaled = np.tanh(np.asarray([hdot], dtype=np.float32) / self.obs_hdot_scale)
            features = np.concatenate([features, hdot_scaled], axis=0)

        if mode == "h_components_closing_reach":
            pos_err = state[0:3] - self.x_ref[0:3]
            pos_norm = np.linalg.norm(pos_err)
            vel_norm = np.linalg.norm(state[3:6])
            att_norm = np.linalg.norm(state[6:9])
            v_toward_origin = -float(np.dot(pos_err, state[3:6])) / max(float(pos_norm), 1e-6)
            reach = np.asarray(
                [
                    pos_norm / (1.5 * self.obs_reach_scale),
                    (self.obs_goal_pos_radius - pos_norm) / (0.3 * self.obs_reach_scale),
                    (self.obs_goal_vel_radius - vel_norm) / (0.3 * self.obs_reach_scale),
                    (self.obs_goal_angle_radius - att_norm) / (0.1 * self.obs_reach_scale),
                    v_toward_origin / (0.5 * self.obs_reach_scale),
                ],
                dtype=np.float32,
            )
            reach_scaled = np.tanh(reach)
            features = np.concatenate([features, reach_scaled], axis=0)

        return features.astype(np.float32)

    def _is_unsafe(self, state):
        """Dawson unsafe_mask: pz >= 0.3 OR ||x_9D|| >= 3.5."""
        pz = state[self.PZ]
        state_norm = np.linalg.norm(state)
        return (pz >= self.unsafe_pz) or (state_norm >= self.unsafe_radius)

    def _is_in_goal(self, state):
        """Dawson goal_mask: ||x_9D|| <= 0.3 AND safe."""
        state_norm = np.linalg.norm(state)
        pz = state[self.PZ]
        in_goal = state_norm <= self.goal_radius
        is_safe = (pz <= self.safe_pz) and (state_norm <= self.safe_radius)
        return in_goal and is_safe

    # ---- Reward ----
    def _compute_reward(self, state, action_norm, u_physical):
        """Compute per-step reward.

        Default mode (dense regulation reward):
            If reward_norm == 'l2':
                r = -(x - x_ref)^T Q (x - x_ref) - a^T R a
            If reward_norm == 'l1':
                r = -sum_i Q_diag[i] * |x_i - x_ref_i|
                    - action_penalty * sum_j |a_j|

            where `a` is the NORMALIZED agent action ∈ [-1, 1]^4. Note that
            a = 0 already corresponds to u_eq via _remap_action, so penalizing
            ||a||² IS equivalent in spirit to penalizing (u_physical - u_eq)
            but stays in the agent's natural action space and avoids any
            scale mismatch from the thrust remap.

        Legacy mode (sparse Dawson-goal-set, opt-in via legacy_reward=True):
            r = -(max(||x|| - 0.3, 0))^2 - (u_physical - u_eq)^T R (u_physical - u_eq)
            Kept for ablation and backward compat with old checkpoints.

        Args:
            state: (9,) current state (post-step)
            action_norm: (4,) NORMALIZED agent action ∈ [-1, 1]^4 (post-clip)
            u_physical: (4,) physical control after _remap_action

        Returns:
            scalar reward (float)
        """
        if self.legacy_reward:
            state_norm = np.linalg.norm(state)
            goal_error = max(state_norm - self.goal_radius, 0.0)
            state_cost = goal_error ** 2
            u_err = u_physical - self.u_eq
            action_cost = float(u_err @ self.R @ u_err)
            return -(state_cost + action_cost)

        # New dense reward (default path)
        state_err = state.astype(np.float32) - self.x_ref
        a = action_norm.astype(np.float32)
        if self.reward_norm == 'l2':
            state_cost = float(state_err @ self.Q @ state_err)
            action_cost = float(a @ self.R @ a)
        else:  # 'l1'
            state_cost = float(np.sum(self.Q_diag * np.abs(state_err)))
            action_cost = float(self.action_penalty * np.sum(np.abs(a)))
        return -(state_cost + action_cost)

    # ---- Obs normalization (identical to Quad2D) ----
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

    def _get_obs(self):
        raw = np.concatenate(
            [self.state.copy(), self._obs_features(self.state)],
            axis=0,
        ).astype(np.float32)
        return self._norm_obs(raw)

    def set_eval_mode(self):
        self._update_stats = False

    def set_train_mode(self):
        self._update_stats = True

    def set_train_step(self, step):
        self._train_step = int(step)

    def get_obs_stats(self):
        return self._obs_mean.copy(), self._obs_var.copy(), self._obs_count

    def set_obs_stats(self, mean, var, count):
        self._obs_mean = mean.copy()
        self._obs_var = var.copy()
        self._obs_count = count

    def _passes_init_curriculum(self, candidate):
        if (not self.init_curriculum) or self.init_curriculum_mode != 'outer_xy':
            return True
        pos_xy_norm = float(np.linalg.norm(candidate[:2]))
        return pos_xy_norm >= self.init_curriculum_pos_xy_min

    def current_init_curriculum_frac(self):
        if not self.init_curriculum:
            return 0.0
        if self.init_curriculum_schedule == 'constant':
            return self.init_curriculum_frac
        if self._train_step <= self.init_curriculum_anneal_start:
            return self.init_curriculum_frac
        if self._train_step >= self.init_curriculum_anneal_end:
            return self.init_curriculum_anneal_end_frac
        alpha = (
            (self._train_step - self.init_curriculum_anneal_start)
            / (self.init_curriculum_anneal_end - self.init_curriculum_anneal_start)
        )
        return ((1.0 - alpha) * self.init_curriculum_frac
                + alpha * self.init_curriculum_anneal_end_frac)

    def _sample_safe_non_goal_start(self, require_curriculum=False):
        """Sample a safe, non-goal start from the default reset box.

        If `require_curriculum=True`, also enforce the active curriculum
        predicate (currently: outer-xy tail). On failure, returns None so the
        caller can gracefully fall back to the baseline sampler.
        """
        candidate = None
        for _ in range(1000):
            candidate = self._rng.uniform(self._reset_low, self._reset_high).astype(np.float32)
            if require_curriculum and not self._passes_init_curriculum(candidate):
                continue
            h = self._sdf(candidate)
            if h < 0 and not self._is_in_goal(candidate):
                return candidate
        return None

    # ---- Core API ----
    def reset(self, options=None):
        """Sample safe, non-goal initial state.

        Default ranges (inside safe set, away from goal):
            px, py in [-1.5, 1.5]
            pz in [-1.0, -0.05]   (in the air, Z-down)
            vx, vy, vz in [-0.5, 0.5]
            phi, theta, psi in [-0.2, 0.2]

        Rejection sampling: discard if !safe or in_goal.
        """
        cur_frac = self.current_init_curriculum_frac()
        use_curriculum = (
            self.init_curriculum
            and options is None
            and self._rng.random() < cur_frac
        )
        candidate = None
        if use_curriculum:
            candidate = self._sample_safe_non_goal_start(require_curriculum=True)
        if candidate is None:
            candidate = self._sample_safe_non_goal_start(require_curriculum=False)
        if candidate is None:
            raise RuntimeError("Failed to sample a safe, non-goal initial state.")
        self.state = candidate

        # Allow overriding initial state
        if options:
            for key, idx in [("init_px", 0), ("init_py", 1), ("init_pz", 2),
                             ("init_vx", 3), ("init_vy", 4), ("init_vz", 5),
                             ("init_phi", 6), ("init_theta", 7), ("init_psi", 8)]:
                if key in options:
                    self.state[idx] = float(options[key])

        self._t = 0
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0
        self._episode_goal_steps = 0

        return self._get_obs()

    def step(self, action):
        """Step the environment.

        Args:
            action: [-1, 1]^4 (agent output, NORMALIZED)

        Returns:
            obs, reward, h_val (SDF), binary_cost, done, info
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        action_norm = action  # already clipped, used for reward

        # Remap to physical control
        u_physical = self._remap_action(action_norm)

        # Step dynamics (forward Euler, matching Dawson)
        self.state = quad3d_step(self.state, u_physical, self.dt, self.m)

        # Safety
        h_val = self._sdf(self.state)
        binary_cost = float(h_val > 0)

        # Reward (new dense quadratic by default; legacy if flagged)
        reward = self._compute_reward(self.state, action_norm, u_physical)

        # Goal check (Dawson goal_mask, kept as diagnostic only)
        in_goal = self._is_in_goal(self.state)

        # Termination (Dawson unsafe_mask = deeper failure)
        terminated = self._is_unsafe(self.state)
        self._t += 1
        truncated = (self._t >= self.max_episode_steps)
        done = terminated or truncated

        # Episode stats
        self._episode_reward += reward
        self._episode_cost += binary_cost
        self._episode_length += 1
        if in_goal:
            self._episode_goal_steps += 1

        pz = float(self.state[self.PZ])
        state_norm = float(np.linalg.norm(self.state))
        # New dual-norm metrics: ALWAYS float, NEVER None.
        # When target_pz==0 these are equal; we still log both so the pipeline
        # is identical for failover (target_pz=-0.5) without code changes.
        state_norm_to_origin = state_norm
        state_norm_to_ref = float(np.linalg.norm(self.state - self.x_ref))

        info = {
            'h': h_val,
            'binary_cost': binary_cost,
            'pz': pz,
            'state_norm': state_norm,                    # legacy alias for _to_origin
            'state_norm_to_origin': state_norm_to_origin,
            'state_norm_to_ref': state_norm_to_ref,
            'in_goal': in_goal,                          # legacy diagnostic
            'in_dawson_goal': in_goal,                   # NEW alias for clarity
            'terminated': terminated,
            'truncated': truncated,
            'episode_reward': self._episode_reward,
            'episode_cost': self._episode_cost,
            'episode_length': self._episode_length,
            'episode_goal_steps': self._episode_goal_steps,        # legacy
            'episode_dawson_goal_steps': self._episode_goal_steps, # NEW alias
        }

        obs = self._get_obs()
        return obs, reward, h_val, binary_cost, done, info

    @property
    def episode_info(self):
        return {
            'reward': self._episode_reward,
            'cost': self._episode_cost,
            'length': self._episode_length,
            'goal_steps': self._episode_goal_steps,         # legacy
            'dawson_goal_steps': self._episode_goal_steps,  # NEW alias
        }


# ============================================================
# Sanity checks (run as script)
# ============================================================

"""Quad2D obstacle stabilize-avoid environment.

This task reuses the original Quad2D planar dynamics, but replaces circular
trajectory tracking with fixed-goal stabilization through a symmetric obstacle
layout.  Observation defaults to the raw 6D system state, matching the Quad3D
style where a fixed goal is part of the task definition rather than appended to
obs.  Optional task-feature modes append analytic safety margins to the
observation; these do not change the safety target h, reward, reset, or
termination logic.

State:  [x, xdot, z, zdot, theta, theta_dot]
Action: [T1, T2] in [-1, 1]^2, remapped internally to [0, 1] thrusts

Safety:
  h_phys(s) > 0 means the point quadrotor has entered an inflated obstacle or
  boundary safety buffer.  Nominal obstacle collision and raw workspace OOB are
  deeper terminal events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import gym
import gym.spaces
import numpy as np


STATE_DIM = 6
ACT_DIM = 2
SUCCESS_MODES = ("full_state_terminal", "dwell_box")
OBS_FEATURE_MODES = ("state", "h", "h_components", "h_components_closing", "h_components_closing_reach")


GLOBAL_RESET_INSET = 0.20
ROUTE_NEAR_GOAL_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-2.45, 2.45, 2.10, 2.90),   # wide top/red region above the goal-front bar
    (-1.20, -0.35, 1.05, 2.35),  # left approach around the goal-front bar
    (0.35, 1.20, 1.05, 2.35),    # right approach around the goal-front bar
)
ROUTE_GAP_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-1.55, -1.12, 0.02, 0.75),  # left inner gap between horizontal bar and block
    (1.12, 1.55, 0.02, 0.75),    # right inner gap between horizontal bar and block
    (-1.30, 1.30, -1.25, -0.45), # broad lower approach, not just the fixed start dot
    (-2.55, -1.55, -0.10, 1.20), # left outer detour corridor
    (1.55, 2.55, -0.10, 1.20),   # right outer detour corridor
)
UNDER_RESET_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-1.05, 1.05, -1.15, -0.15),  # broad lower approach under the horizontal bar
    (-2.25, -1.55, -1.25, -0.12), # below the left block
    (1.55, 2.25, -1.25, -0.12),   # below the right block
    (-0.75, 0.75, 0.62, 0.92),    # between horizontal bar and goal-front bar
    (-1.46, -1.18, 0.18, 0.50),   # left inner gap
    (1.18, 1.46, 0.18, 0.50),     # right inner gap
)
LOW_PAPER_JITTER_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-0.30, 0.30, -1.20, -0.85),
)
LOW_UNDER_BAR_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-1.10, 1.10, -0.95, -0.10),
)
LOW_UNDER_BLOCK_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-2.25, -1.45, -1.20, -0.10),
    (1.45, 2.25, -1.20, -0.10),
)
LOW_INNER_GAP_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-1.46, -1.18, 0.10, 0.50),
    (1.18, 1.46, 0.10, 0.50),
)
LOW_INNER_GAP_BOXES_XZ_V3: Tuple[Tuple[float, float, float, float], ...] = (
    (-1.38, -0.85, 0.10, 0.50),
    (0.85, 1.38, 0.10, 0.50),
)
SIMPLE_UNDER3_BAR_BOX_XZ: Tuple[float, float, float, float] = (-0.80, 0.80, -0.45, -0.08)
SIMPLE_UNDER3_LEFT_BLOCK_BOX_XZ: Tuple[float, float, float, float] = (-2.05, -1.40, -0.45, -0.08)
SIMPLE_UNDER3_RIGHT_BLOCK_BOX_XZ: Tuple[float, float, float, float] = (1.40, 2.05, -0.45, -0.08)
SIMPLE_UNDER3_GAP_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-1.35, -0.85, -0.20, 0.50),
    (0.85, 1.35, -0.20, 0.50),
)
SIMPLE_UNDER3_GOAL_SIDE_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-0.85, -0.45, 2.35, 2.55),
    (0.45, 0.85, 2.35, 2.55),
)
LOW_MID_BOXES_XZ: Tuple[Tuple[float, float, float, float], ...] = (
    (-1.40, 1.40, 0.55, 1.15),
)
GAP_MIX_BRANCH_WEIGHTS = np.array([0.50, 0.50], dtype=np.float64)


def quad2d_step(
    state: np.ndarray,
    action_raw: np.ndarray,
    dt: float,
    m: float = 1.0,
    I: float = 0.02,
    g: float = 9.81,
    thrust_scale: Optional[float] = None,
    torque_scale: float = 0.1,
) -> np.ndarray:
    """Semi-implicit Euler step for the planar quadrotor."""
    if thrust_scale is None:
        thrust_scale = m * g

    a = np.clip(np.asarray(action_raw, dtype=np.float32), 0.0, 1.0)
    x, xdot, z, zdot, theta, theta_dot = np.asarray(state, dtype=np.float32)

    f1 = thrust_scale * a[0]
    f2 = thrust_scale * a[1]
    total_thrust = f1 + f2
    torque = torque_scale * (a[1] - a[0])

    xddot = -(total_thrust / m) * np.sin(theta)
    zddot = (total_thrust / m) * np.cos(theta) - g
    theta_ddot = torque / I

    xdot = xdot + xddot * dt
    x = x + xdot * dt
    zdot = zdot + zddot * dt
    z = z + zdot * dt
    theta_dot = theta_dot + theta_ddot * dt
    theta = theta + theta_dot * dt

    return np.array([x, xdot, z, zdot, theta, theta_dot], dtype=np.float32)


def mirror_state(state: np.ndarray) -> np.ndarray:
    """Mirror a state across the layout centerline x=0."""
    s = np.asarray(state, dtype=np.float32).copy()
    s[0] = -s[0]
    s[1] = -s[1]
    s[4] = -s[4]
    s[5] = -s[5]
    return s


def mirror_action(action: np.ndarray) -> np.ndarray:
    """Mirror normalized [-1, 1] left/right thrust commands."""
    a = np.asarray(action, dtype=np.float32).copy()
    return a[[1, 0]]


@dataclass(frozen=True)
class RectObstacle:
    name: str
    center: Tuple[float, float]
    size: Tuple[float, float]

    @property
    def xmin(self) -> float:
        return self.center[0] - 0.5 * self.size[0]

    @property
    def xmax(self) -> float:
        return self.center[0] + 0.5 * self.size[0]

    @property
    def zmin(self) -> float:
        return self.center[1] - 0.5 * self.size[1]

    @property
    def zmax(self) -> float:
        return self.center[1] + 0.5 * self.size[1]


@dataclass(frozen=True)
class Quad2DStabLayout:
    name: str
    xlim: Tuple[float, float]
    zlim: Tuple[float, float]
    start_center: Tuple[float, float]
    start_halfwidth: Tuple[float, float]
    goal: Tuple[float, float]
    obstacles: Tuple[RectObstacle, ...]
    obstacle_margin: float = 0.10
    boundary_margin: float = 0.10

    @property
    def start_low(self) -> np.ndarray:
        return np.array(
            [
                self.start_center[0] - self.start_halfwidth[0],
                -0.30,
                self.start_center[1] - self.start_halfwidth[1],
                -0.30,
                -0.12,
                -0.08,
            ],
            dtype=np.float32,
        )

    @property
    def start_high(self) -> np.ndarray:
        return np.array(
            [
                self.start_center[0] + self.start_halfwidth[0],
                0.30,
                self.start_center[1] + self.start_halfwidth[1],
                0.30,
                0.12,
                0.08,
            ],
            dtype=np.float32,
        )


def make_corridor_layout_v2() -> Quad2DStabLayout:
    """Symmetric corridor layout with widened inner and outer passages."""
    return Quad2DStabLayout(
        name="corridor_v2",
        xlim=(-2.78, 2.78),
        zlim=(-1.55, 3.10),
        start_center=(0.0, -1.08),
        start_halfwidth=(0.15, 0.12),
        goal=(0.0, 2.45),
        obstacles=(
            RectObstacle("horizontal_bar", center=(0.0, 0.30), size=(2.00, 0.30)),
            RectObstacle("left_block", center=(-1.84, 0.30), size=(0.44, 0.44)),
            RectObstacle("right_block", center=(1.84, 0.30), size=(0.44, 0.44)),
            RectObstacle("goal_front_bar", center=(0.0, 1.55), size=(0.38, 0.86)),
        ),
        obstacle_margin=0.10,
        boundary_margin=0.10,
    )


def make_corridor_layout_v3() -> Quad2DStabLayout:
    """Easier wide-gap corridor draft for diagnosing geometry hardness."""
    return Quad2DStabLayout(
        name="corridor_v3",
        xlim=(-2.78, 2.78),
        zlim=(-1.55, 3.10),
        start_center=(0.0, -1.08),
        start_halfwidth=(0.15, 0.12),
        goal=(0.0, 2.45),
        obstacles=(
            RectObstacle("horizontal_bar", center=(0.0, 0.30), size=(1.30, 0.30)),
            RectObstacle("left_block", center=(-1.72, 0.30), size=(0.32, 0.32)),
            RectObstacle("right_block", center=(1.72, 0.30), size=(0.32, 0.32)),
            RectObstacle("goal_front_bar", center=(0.0, 1.55), size=(0.34, 0.74)),
        ),
        obstacle_margin=0.10,
        boundary_margin=0.10,
    )


LAYOUTS = {
    "corridor_v2": make_corridor_layout_v2,
    "corridor_v3": make_corridor_layout_v3,
}


def get_layout(name: str = "corridor_v2") -> Quad2DStabLayout:
    if name not in LAYOUTS:
        raise ValueError(f"Unknown Quad2D stab layout {name!r}. Valid: {sorted(LAYOUTS)}")
    return LAYOUTS[name]()


def _rect_h_value(x: float, z: float, rect: RectObstacle, margin: float) -> float:
    """Signed value for an inflated rectangle, positive inside."""
    cx, cz = rect.center
    hx = 0.5 * rect.size[0] + margin
    hz = 0.5 * rect.size[1] + margin
    qx = abs(x - cx) - hx
    qz = abs(z - cz) - hz
    outside = np.hypot(max(qx, 0.0), max(qz, 0.0))
    inside = min(max(qx, qz), 0.0)
    signed_distance = outside + inside  # negative inside inflated rect
    return float(-signed_distance)


def rect_h_values(x: float, z: float, rects: Iterable[RectObstacle], margin: float) -> List[float]:
    return [_rect_h_value(x, z, rect, margin) for rect in rects]


def safety_h_components(x: float, z: float, layout: Quad2DStabLayout) -> np.ndarray:
    """Inflated-obstacle and workspace-boundary h components.

    Ordering is layout.obstacles followed by the workspace boundary. Positive
    values are unsafe, matching physical_h_value().
    """
    values = rect_h_values(x, z, layout.obstacles, layout.obstacle_margin)
    values.append(workspace_h_value(x, z, layout, layout.boundary_margin))
    return np.asarray(values, dtype=np.float32)


def workspace_h_value(x: float, z: float, layout: Quad2DStabLayout, margin: float) -> float:
    xmin, xmax = layout.xlim
    zmin, zmax = layout.zlim
    return float(max(xmin + margin - x, x - (xmax - margin), zmin + margin - z, z - (zmax - margin)))


def raw_workspace_h_value(x: float, z: float, layout: Quad2DStabLayout) -> float:
    xmin, xmax = layout.xlim
    zmin, zmax = layout.zlim
    return float(max(xmin - x, x - xmax, zmin - z, z - zmax))


def physical_h_value(x: float, z: float, layout: Quad2DStabLayout) -> float:
    values = rect_h_values(x, z, layout.obstacles, layout.obstacle_margin)
    values.append(workspace_h_value(x, z, layout, layout.boundary_margin))
    return float(max(values))


def nominal_collision_h_value(x: float, z: float, layout: Quad2DStabLayout) -> float:
    values = rect_h_values(x, z, layout.obstacles, 0.0)
    values.append(raw_workspace_h_value(x, z, layout))
    return float(max(values))


def shape_stab_h(
    h_phys: float,
    mode: str,
    unsafe_const: float = 0.0,
    unsafe_slope: float = 0.0,
    global_scale: float = 0.0,
) -> float:
    """Shape the physical signed distance used as the Q_h target.

    All supported modes preserve the h=0 boundary.  Binary cost, reset
    rejection, and terminal collision checks always use h_phys directly.
    """
    if mode == "raw":
        return float(h_phys)
    if mode == "hard":
        if unsafe_const <= 0.0:
            raise ValueError(
                f"shape_stab_h: mode='hard' requires unsafe_const > 0, got {unsafe_const!r}"
            )
        return float(unsafe_const if h_phys > 0.0 else h_phys)
    if mode == "hard_slope":
        if unsafe_const < 0.0:
            raise ValueError(
                f"shape_stab_h: mode='hard_slope' requires unsafe_const >= 0, got {unsafe_const!r}"
            )
        if unsafe_slope < 0.0:
            raise ValueError(
                f"shape_stab_h: mode='hard_slope' requires unsafe_slope >= 0, got {unsafe_slope!r}"
            )
        if (unsafe_const + unsafe_slope) <= 0.0:
            raise ValueError(
                "shape_stab_h: mode='hard_slope' requires unsafe_const + unsafe_slope > 0"
            )
        return float(unsafe_const + unsafe_slope * h_phys if h_phys > 0.0 else h_phys)
    if mode == "hard_scale":
        if unsafe_const <= 0.0:
            raise ValueError(
                f"shape_stab_h: mode='hard_scale' requires unsafe_const > 0, got {unsafe_const!r}"
            )
        if global_scale <= 0.0:
            raise ValueError(
                f"shape_stab_h: mode='hard_scale' requires global_scale > 0, got {global_scale!r}"
            )
        base = unsafe_const if h_phys > 0.0 else h_phys
        return float(base * global_scale)
    raise ValueError(
        f"shape_stab_h: unknown mode={mode!r}. Valid: raw, hard, hard_slope, hard_scale."
    )


class Quad2DStabEnv:
    """Fixed-goal Quad2D stabilize-avoid task."""

    X = 0
    XDOT = 1
    Z = 2
    ZDOT = 3
    THETA = 4
    THETA_DOT = 5

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        max_episode_steps: int = 480,
        seed: int = 0,
        layout_name: str = "corridor_v2",
        q_pos: float = 10.0,
        q_x: Optional[float] = None,
        q_z: Optional[float] = None,
        q_vel: float = 1.0,
        q_angle: float = 0.2,
        action_penalty: float = 1e-3,
        reward_norm: str = "l2",
        goal_pos_radius: float = 0.30,
        goal_vel_radius: float = 0.45,
        goal_angle_radius: float = 0.25,
        goal_dwell_steps: int = 45,
        success_mode: str = "full_state_terminal",
        success_radius: float = 0.50,
        terminate_on_success: bool = False,
        reset_mode: str = "start_band",
        reset_mix_start_frac: float = 0.70,
        reset_mix_global_frac: float = 0.20,
        reset_mix_goal_frac: float = 0.10,
        reset_route_global_frac: float = 0.35,
        reset_route_near_frac: float = 0.30,
        reset_route_gap_frac: float = 0.35,
        reset_under_global_frac: float = 0.35,
        reset_under_frac: float = 0.65,
        reset_low_frac: float = 0.80,
        reset_low_mid_frac: float = 0.15,
        reset_low_near_frac: float = 0.05,
        reset_low_paper_jitter_frac: float = 0.30,
        reset_low_paper_pinpoint_vel_frac: float = 0.0,
        reset_low_under_bar_frac: float = 0.20,
        reset_low_under_blocks_frac: float = 0.20,
        reset_low_inner_gaps_frac: float = 0.20,
        reset_low_global_frac: float = 0.10,
        reset_simple_bar_frac: float = 0.30,
        reset_simple_left_block_frac: float = 0.10,
        reset_simple_right_block_frac: float = 0.10,
        reset_simple_gap_frac: float = 0.10,
        reset_simple_low_uniform_frac: float = 0.25,
        reset_simple_near_frac: float = 0.15,
        reset_simple_side_frac: float = 0.0,
        reset_simple_fork_frac: float = 0.0,
        reset_safety_bar_frac: float = 0.20,
        reset_safety_left_block_frac: float = 0.15,
        reset_safety_right_block_frac: float = 0.15,
        reset_safety_gap_frac: float = 0.15,
        reset_safety_low_uniform_frac: float = 0.25,
        reset_safety_mid_uniform_frac: float = 0.05,
        reset_safety_near_frac: float = 0.05,
        simple_under_vx_abs: float = 0.50,
        simple_under_vz_low: float = 0.00,
        simple_under_vz_high: float = 0.70,
        simple_bar_vz_low: Optional[float] = None,
        simple_bar_vz_high: Optional[float] = None,
        simple_block_vz_low: Optional[float] = None,
        simple_block_vz_high: Optional[float] = None,
        simple_under_xdot_zero_frac: float = 0.0,
        simple_bar_xdot_zero_frac: Optional[float] = None,
        simple_block_xdot_zero_frac: Optional[float] = None,
        simple_gap_vx_abs: float = 0.40,
        simple_gap_vz_low: float = -0.20,
        simple_gap_vz_high: float = 0.50,
        simple_low_uniform_vx_abs: float = 0.35,
        simple_low_uniform_vz_abs: float = 0.35,
        simple_fork_x_low: float = 0.05,
        simple_fork_x_high: float = 0.35,
        simple_fork_z_low: float = -1.00,
        simple_fork_z_high: float = -0.30,
        simple_fork_vx_low: float = 0.10,
        simple_fork_vx_high: float = 0.55,
        simple_fork_vz_low: float = -0.05,
        simple_fork_vz_high: float = 0.45,
        simple_fork_theta_abs: float = 0.12,
        simple_fork_omega_abs: float = 0.15,
        low_paper_jitter_vx_abs: float = 0.25,
        low_paper_jitter_vz_abs: float = 0.25,
        low_paper_pinpoint_vx_abs: float = 1.00,
        low_paper_pinpoint_vz_abs: float = 0.30,
        low_paper_pinpoint_theta_abs: float = 0.10,
        low_paper_pinpoint_omega_abs: float = 0.05,
        low_paper_pinpoint_x_abs: float = 0.05,
        low_paper_pinpoint_z_center: float = -1.08,
        low_paper_pinpoint_z_halfwidth: float = 0.02,
        low_under_bar_vx_abs: float = 0.25,
        low_under_bar_vz_abs: float = 0.25,
        low_under_blocks_vx_abs: float = 0.25,
        low_under_blocks_vz_abs: float = 0.25,
        low_inner_gaps_vx_abs: float = 0.25,
        low_inner_gaps_vz_abs: float = 0.25,
        init_h_threshold: float = 0.0,
        sdf_mode: str = "raw",
        sdf_unsafe_const: float = 0.0,
        sdf_unsafe_slope: float = 0.0,
        sdf_global_scale: float = 0.0,
        gate_lambda: float = 0.0,
        gate_x_max: float = 0.90,
        gate_z_center: float = 0.30,
        gate_z_width: float = 1.20,
        obs_feature_mode: str = "state",
        obs_h_scale: float = 1.0,
        obs_hdot_scale: float = 1.0,
        dwell_bonus: float = 0.0,
    ) -> None:
        self.dt = float(dt)
        self.max_episode_steps = int(max_episode_steps)
        self.layout = get_layout(layout_name)
        self.layout_name = layout_name

        self.m = 1.0
        self.I = 0.02
        self.g = 9.81
        self.thrust_scale = self.m * self.g
        self.torque_scale = 0.1

        self.goal = np.array(self.layout.goal, dtype=np.float32)
        self.x_ref = np.array([self.goal[0], 0.0, self.goal[1], 0.0, 0.0, 0.0], dtype=np.float32)
        self.a_ref = np.array([0.5, 0.5], dtype=np.float32)

        self.q_pos = float(q_pos)
        self.q_x = float(self.q_pos if q_x is None else q_x)
        self.q_z = float(self.q_pos if q_z is None else q_z)
        self.q_vel = float(q_vel)
        self.q_angle = float(q_angle)
        self.action_penalty = float(action_penalty)
        self.Q_diag = np.array(
            [self.q_x, self.q_vel, self.q_z, self.q_vel, self.q_angle, self.q_angle],
            dtype=np.float32,
        )
        self.R_diag = np.array([self.action_penalty, self.action_penalty], dtype=np.float32)
        self.dwell_bonus = float(dwell_bonus)
        self.reward_norm = str(reward_norm).lower()
        if self.reward_norm not in ("l1", "l2"):
            raise ValueError(f"reward_norm must be 'l1' or 'l2', got {reward_norm!r}")

        self.goal_pos_radius = float(goal_pos_radius)
        self.goal_vel_radius = float(goal_vel_radius)
        self.goal_angle_radius = float(goal_angle_radius)
        self.goal_dwell_steps = int(goal_dwell_steps)
        self.success_mode = str(success_mode)
        if self.success_mode not in SUCCESS_MODES:
            raise ValueError(f"success_mode must be one of {SUCCESS_MODES}, got {success_mode!r}")
        self.success_radius = float(success_radius)
        self.terminate_on_success = bool(terminate_on_success)

        self.reset_mode = str(reset_mode)
        if self.reset_mode not in (
            "start_band",
            "global_safe",
            "mix",
            "route_mix",
            "under_mix",
            "low_mix",
            "gap_mix",
            "simple_under3",
            "safety_under50",
        ):
            raise ValueError(
                "reset_mode must be start_band, global_safe, mix, route_mix, under_mix, "
                "low_mix, gap_mix, simple_under3, or safety_under50"
            )
        self.reset_mix_start_frac = float(reset_mix_start_frac)
        self.reset_mix_global_frac = float(reset_mix_global_frac)
        self.reset_mix_goal_frac = float(reset_mix_goal_frac)
        self.reset_route_global_frac = float(reset_route_global_frac)
        self.reset_route_near_frac = float(reset_route_near_frac)
        self.reset_route_gap_frac = float(reset_route_gap_frac)
        self.reset_under_global_frac = float(reset_under_global_frac)
        self.reset_under_frac = float(reset_under_frac)
        self.reset_low_frac = float(reset_low_frac)
        self.reset_low_mid_frac = float(reset_low_mid_frac)
        self.reset_low_near_frac = float(reset_low_near_frac)
        self.reset_low_component_weights = np.array(
            [
                reset_low_paper_jitter_frac,
                reset_low_paper_pinpoint_vel_frac,
                reset_low_under_bar_frac,
                reset_low_under_blocks_frac,
                reset_low_inner_gaps_frac,
                reset_low_global_frac,
            ],
            dtype=np.float64,
        )
        self.reset_simple_under3_weights = np.array(
            [
                reset_simple_bar_frac,
                reset_simple_left_block_frac,
                reset_simple_right_block_frac,
                reset_simple_gap_frac,
                reset_simple_low_uniform_frac,
                reset_simple_near_frac,
                reset_simple_side_frac,
                reset_simple_fork_frac,
            ],
            dtype=np.float64,
        )
        self.reset_safety_under50_weights = np.array(
            [
                reset_safety_bar_frac,
                reset_safety_left_block_frac,
                reset_safety_right_block_frac,
                reset_safety_gap_frac,
                reset_safety_low_uniform_frac,
                reset_safety_mid_uniform_frac,
                reset_safety_near_frac,
            ],
            dtype=np.float64,
        )
        self.simple_under_vx_abs = float(simple_under_vx_abs)
        self.simple_under_vz_low = float(simple_under_vz_low)
        self.simple_under_vz_high = float(simple_under_vz_high)
        self.simple_bar_vz_low = float(self.simple_under_vz_low if simple_bar_vz_low is None else simple_bar_vz_low)
        self.simple_bar_vz_high = float(self.simple_under_vz_high if simple_bar_vz_high is None else simple_bar_vz_high)
        self.simple_block_vz_low = float(
            self.simple_under_vz_low if simple_block_vz_low is None else simple_block_vz_low
        )
        self.simple_block_vz_high = float(
            self.simple_under_vz_high if simple_block_vz_high is None else simple_block_vz_high
        )
        self.simple_under_xdot_zero_frac = float(np.clip(simple_under_xdot_zero_frac, 0.0, 1.0))
        self.simple_bar_xdot_zero_frac = float(
            np.clip(
                self.simple_under_xdot_zero_frac
                if simple_bar_xdot_zero_frac is None
                else simple_bar_xdot_zero_frac,
                0.0,
                1.0,
            )
        )
        self.simple_block_xdot_zero_frac = float(
            np.clip(
                self.simple_under_xdot_zero_frac
                if simple_block_xdot_zero_frac is None
                else simple_block_xdot_zero_frac,
                0.0,
                1.0,
            )
        )
        self.simple_gap_vx_abs = float(simple_gap_vx_abs)
        self.simple_gap_vz_low = float(simple_gap_vz_low)
        self.simple_gap_vz_high = float(simple_gap_vz_high)
        self.simple_low_uniform_vx_abs = float(simple_low_uniform_vx_abs)
        self.simple_low_uniform_vz_abs = float(simple_low_uniform_vz_abs)
        self.simple_fork_x_low = float(simple_fork_x_low)
        self.simple_fork_x_high = float(simple_fork_x_high)
        self.simple_fork_z_low = float(simple_fork_z_low)
        self.simple_fork_z_high = float(simple_fork_z_high)
        self.simple_fork_vx_low = float(simple_fork_vx_low)
        self.simple_fork_vx_high = float(simple_fork_vx_high)
        self.simple_fork_vz_low = float(simple_fork_vz_low)
        self.simple_fork_vz_high = float(simple_fork_vz_high)
        self.simple_fork_theta_abs = float(simple_fork_theta_abs)
        self.simple_fork_omega_abs = float(simple_fork_omega_abs)
        self.low_paper_jitter_vx_abs = float(low_paper_jitter_vx_abs)
        self.low_paper_jitter_vz_abs = float(low_paper_jitter_vz_abs)
        self.low_paper_pinpoint_vx_abs = float(low_paper_pinpoint_vx_abs)
        self.low_paper_pinpoint_vz_abs = float(low_paper_pinpoint_vz_abs)
        self.low_paper_pinpoint_theta_abs = float(low_paper_pinpoint_theta_abs)
        self.low_paper_pinpoint_omega_abs = float(low_paper_pinpoint_omega_abs)
        self.low_paper_pinpoint_x_abs = float(low_paper_pinpoint_x_abs)
        self.low_paper_pinpoint_z_center = float(low_paper_pinpoint_z_center)
        self.low_paper_pinpoint_z_halfwidth = float(low_paper_pinpoint_z_halfwidth)
        self.low_under_bar_vx_abs = float(low_under_bar_vx_abs)
        self.low_under_bar_vz_abs = float(low_under_bar_vz_abs)
        self.low_under_blocks_vx_abs = float(low_under_blocks_vx_abs)
        self.low_under_blocks_vz_abs = float(low_under_blocks_vz_abs)
        self.low_inner_gaps_vx_abs = float(low_inner_gaps_vx_abs)
        self.low_inner_gaps_vz_abs = float(low_inner_gaps_vz_abs)
        self.init_h_threshold = float(init_h_threshold)

        self.sdf_mode = str(sdf_mode)
        self.sdf_unsafe_const = float(sdf_unsafe_const)
        self.sdf_unsafe_slope = float(sdf_unsafe_slope)
        self.sdf_global_scale = float(sdf_global_scale)
        self.gate_lambda = float(gate_lambda)
        self.gate_x_max = float(gate_x_max)
        self.gate_z_center = float(gate_z_center)
        self.gate_z_width = float(gate_z_width)
        _ = self._shape_h(-0.1)
        _ = self._shape_h(0.1)

        self.obs_feature_mode = str(obs_feature_mode)
        if self.obs_feature_mode not in OBS_FEATURE_MODES:
            raise ValueError(f"obs_feature_mode must be one of {OBS_FEATURE_MODES}, got {obs_feature_mode!r}")
        self.obs_h_scale = float(obs_h_scale)
        self.obs_hdot_scale = float(obs_hdot_scale)
        if self.obs_h_scale <= 0.0:
            raise ValueError("obs_h_scale must be > 0.")
        if self.obs_hdot_scale <= 0.0:
            raise ValueError("obs_hdot_scale must be > 0.")
        self.obs_feature_dim = self._obs_feature_dim_for_mode(self.obs_feature_mode, self.layout)
        self.obs_dim = STATE_DIM + self.obs_feature_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(ACT_DIM, dtype=np.float32),
            high=np.ones(ACT_DIM, dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self._t = 0
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0
        self._episode_goal_steps = 0
        self._episode_total_goal_steps = 0
        self._episode_max_goal_streak = 0
        self._episode_success = False
        self._episode_success_step: Optional[int] = None
        self._episode_last_action_error_norm = float("nan")

        self._normalize_obs = True
        self._obs_norm_clip = 5.0
        self._obs_mean = np.zeros(self.obs_dim, dtype=np.float64)
        self._obs_var = np.ones(self.obs_dim, dtype=np.float64)
        self._obs_count = 0
        self._update_stats = True

    @staticmethod
    def _obs_feature_dim_for_mode(mode: str, layout: Quad2DStabLayout) -> int:
        component_count = len(layout.obstacles) + 1
        if mode == "state":
            return 0
        if mode == "h":
            return 1
        if mode == "h_components":
            return 1 + component_count
        if mode == "h_components_closing":
            return 2 + component_count
        if mode == "h_components_closing_reach":
            return 2 + component_count + 5
        raise ValueError(f"Unknown obs feature mode {mode!r}")

    def _shape_h(self, h_phys: float) -> float:
        return shape_stab_h(
            h_phys,
            self.sdf_mode,
            self.sdf_unsafe_const,
            self.sdf_unsafe_slope,
            self.sdf_global_scale,
        )

    def h_phys(self, state: Optional[np.ndarray] = None) -> float:
        s = self.state if state is None else state
        return physical_h_value(float(s[self.X]), float(s[self.Z]), self.layout)

    def h_components(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        s = self.state if state is None else state
        return safety_h_components(float(s[self.X]), float(s[self.Z]), self.layout)

    def _h_dot_feature(self, state: np.ndarray, h_now: float) -> float:
        x_next = float(state[self.X] + state[self.XDOT] * self.dt)
        z_next = float(state[self.Z] + state[self.ZDOT] * self.dt)
        h_next = physical_h_value(x_next, z_next, self.layout)
        return float((h_next - h_now) / max(self.dt, 1e-8))

    def _reach_features(self, state: np.ndarray) -> np.ndarray:
        dx = float(state[self.X] - self.goal[0])
        dz = float(state[self.Z] - self.goal[1])
        dist = float(np.hypot(dx, dz))
        vx = float(state[self.XDOT])
        vz = float(state[self.ZDOT])
        vel_norm = float(np.hypot(vx, vz))
        v_toward_goal = -(dx * vx + dz * vz) / max(dist, 1e-6)
        angle_margin = self.goal_angle_radius - abs(float(state[self.THETA]))
        return np.tanh(
            np.asarray(
                [
                    dist / 1.50,
                    (self.goal_pos_radius - dist) / 0.30,
                    (self.goal_vel_radius - vel_norm) / 0.30,
                    angle_margin / 0.10,
                    v_toward_goal / 0.75,
                ],
                dtype=np.float32,
            )
        ).astype(np.float32)

    def _obs_features(self, state: np.ndarray) -> np.ndarray:
        mode = self.obs_feature_mode
        if mode == "state":
            return np.zeros((0,), dtype=np.float32)
        components = self.h_components(state)
        h_margin = float(np.max(components))
        h_scaled = np.tanh(np.asarray([h_margin], dtype=np.float32) / self.obs_h_scale)
        if mode == "h":
            return h_scaled.astype(np.float32)
        comp_scaled = np.tanh(components.astype(np.float32) / self.obs_h_scale)
        features = np.concatenate([h_scaled, comp_scaled], axis=0)
        if mode in ("h_components_closing", "h_components_closing_reach"):
            hdot = self._h_dot_feature(state, h_margin)
            hdot_scaled = np.tanh(np.asarray([hdot], dtype=np.float32) / self.obs_hdot_scale)
            features = np.concatenate([features, hdot_scaled], axis=0)
        if mode == "h_components_closing_reach":
            features = np.concatenate([features, self._reach_features(state)], axis=0)
        return features.astype(np.float32)

    def _raw_obs(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        s = self.state if state is None else state
        base = np.asarray(s, dtype=np.float32)
        if self.obs_feature_dim <= 0:
            return base.copy()
        return np.concatenate([base, self._obs_features(base)], axis=0).astype(np.float32)

    def h_nominal_collision(self, state: Optional[np.ndarray] = None) -> float:
        s = self.state if state is None else state
        return nominal_collision_h_value(float(s[self.X]), float(s[self.Z]), self.layout)

    def _is_terminal_collision(self, state: np.ndarray) -> bool:
        return self.h_nominal_collision(state) > 0.0

    def _is_in_goal(self, state: np.ndarray) -> bool:
        pos_err = np.linalg.norm(state[[self.X, self.Z]] - self.goal)
        vel_norm = np.linalg.norm(state[[self.XDOT, self.ZDOT]])
        angle_ok = abs(float(state[self.THETA])) <= self.goal_angle_radius
        return (
            pos_err <= self.goal_pos_radius
            and vel_norm <= self.goal_vel_radius
            and angle_ok
            and self.h_phys(state) < 0.0
        )

    def _terminal_state_norm_to_ref(self, state: Optional[np.ndarray] = None) -> float:
        s = self.state if state is None else state
        return float(np.linalg.norm(s.astype(np.float32) - self.x_ref))

    def _full_state_success(self, state: Optional[np.ndarray] = None) -> bool:
        s = self.state if state is None else state
        return self._terminal_state_norm_to_ref(s) < self.success_radius and self.h_nominal_collision(s) <= 0.0

    def _compute_reward(self, state: np.ndarray, action_raw: np.ndarray) -> float:
        err = state.astype(np.float32) - self.x_ref
        action_err = action_raw.astype(np.float32) - self.a_ref
        if self.reward_norm == "l2":
            state_cost = float(np.sum(np.square(err) * self.Q_diag))
            action_cost = float(np.sum(np.square(action_err) * self.R_diag))
        else:
            state_cost = float(np.sum(np.abs(err) * self.Q_diag))
            action_cost = float(np.sum(np.abs(action_err) * self.R_diag))
        gate_cost = 0.0
        if self.gate_lambda > 0.0 and self.gate_z_width > 0.0:
            x = float(state[self.X])
            z = float(state[self.Z])
            gate_profile = max(0.0, 1.0 - abs(z - self.gate_z_center) / self.gate_z_width)
            x_req = self.gate_x_max * gate_profile
            gate_cost = self.gate_lambda * max(0.0, x_req - abs(x)) ** 2
        bonus = self.dwell_bonus if self.dwell_bonus > 0.0 and self._is_in_goal(state) else 0.0
        return -(state_cost + action_cost + gate_cost) + bonus

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
        return np.clip(normed, -self._obs_norm_clip, self._obs_norm_clip).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._norm_obs(self._raw_obs())

    def set_eval_mode(self) -> None:
        self._update_stats = False

    def set_train_mode(self) -> None:
        self._update_stats = True

    def get_obs_stats(self) -> Tuple[np.ndarray, np.ndarray, int]:
        return self._obs_mean.copy(), self._obs_var.copy(), self._obs_count

    def set_obs_stats(self, mean: np.ndarray, var: np.ndarray, count: int) -> None:
        mean_arr = np.asarray(mean, dtype=np.float64)
        var_arr = np.asarray(var, dtype=np.float64)
        if mean_arr.shape != (self.obs_dim,) or var_arr.shape != (self.obs_dim,):
            raise ValueError(
                f"obs stats shape mismatch for obs_dim={self.obs_dim}: "
                f"mean={mean_arr.shape}, var={var_arr.shape}"
            )
        self._obs_mean = mean_arr.copy()
        self._obs_var = var_arr.copy()
        self._obs_count = int(count)

    def _sample_start_band_state(self) -> np.ndarray:
        return self._rng.uniform(self.layout.start_low, self.layout.start_high).astype(np.float32)

    def _sample_goal_near_state(self) -> np.ndarray:
        low = np.array([-0.45, -0.25, 2.10, -0.25, -0.12, -0.08], dtype=np.float32)
        high = np.array([0.45, 0.25, 2.70, 0.25, 0.12, 0.08], dtype=np.float32)
        return self._rng.uniform(low, high).astype(np.float32)

    def _sample_box_state(
        self,
        box_xz: Tuple[float, float, float, float],
        *,
        vel_abs: float = 0.25,
        vx_abs: Optional[float] = None,
        vz_abs: Optional[float] = None,
        theta_abs: float = 0.12,
        omega_abs: float = 0.08,
    ) -> np.ndarray:
        xmin, xmax, zmin, zmax = box_xz
        vx = float(vel_abs if vx_abs is None else vx_abs)
        vz = float(vel_abs if vz_abs is None else vz_abs)
        low = np.array([xmin, -vx, zmin, -vz, -theta_abs, -omega_abs], dtype=np.float32)
        high = np.array([xmax, vx, zmax, vz, theta_abs, omega_abs], dtype=np.float32)
        return self._rng.uniform(low, high).astype(np.float32)

    def _sample_box_state_vz_range(
        self,
        box_xz: Tuple[float, float, float, float],
        *,
        vx_abs: float,
        vz_low: float,
        vz_high: float,
        theta_abs: float = 0.12,
        omega_abs: float = 0.08,
    ) -> np.ndarray:
        xmin, xmax, zmin, zmax = box_xz
        low = np.array([xmin, -vx_abs, zmin, vz_low, -theta_abs, -omega_abs], dtype=np.float32)
        high = np.array([xmax, vx_abs, zmax, vz_high, theta_abs, omega_abs], dtype=np.float32)
        return self._rng.uniform(low, high).astype(np.float32)

    def _sample_route_near_goal_state(self) -> np.ndarray:
        idx = int(self._rng.integers(0, len(ROUTE_NEAR_GOAL_BOXES_XZ)))
        return self._sample_box_state(ROUTE_NEAR_GOAL_BOXES_XZ[idx])

    def _sample_route_gap_state(self) -> np.ndarray:
        idx = int(self._rng.integers(0, len(ROUTE_GAP_BOXES_XZ)))
        return self._sample_box_state(ROUTE_GAP_BOXES_XZ[idx])

    def _sample_under_state(self) -> np.ndarray:
        idx = int(self._rng.integers(0, len(UNDER_RESET_BOXES_XZ)))
        return self._sample_box_state(UNDER_RESET_BOXES_XZ[idx])

    def _sample_from_boxes(self, boxes: Tuple[Tuple[float, float, float, float], ...], **kwargs) -> np.ndarray:
        idx = int(self._rng.integers(0, len(boxes)))
        return self._sample_box_state(boxes[idx], **kwargs)

    def _sample_paper_pinpoint_vel_state(self) -> np.ndarray:
        x_abs = max(self.low_paper_pinpoint_x_abs, 0.0)
        z_halfwidth = max(self.low_paper_pinpoint_z_halfwidth, 0.0)
        box = (
            -x_abs,
            x_abs,
            self.low_paper_pinpoint_z_center - z_halfwidth,
            self.low_paper_pinpoint_z_center + z_halfwidth,
        )
        return self._sample_box_state(
            box,
            vx_abs=self.low_paper_pinpoint_vx_abs,
            vz_abs=self.low_paper_pinpoint_vz_abs,
            theta_abs=self.low_paper_pinpoint_theta_abs,
            omega_abs=self.low_paper_pinpoint_omega_abs,
        )

    def _sample_low_global_state(self) -> np.ndarray:
        xmin, xmax = self.layout.xlim
        zmin, zmax = self.layout.zlim
        low_z = max(zmin + GLOBAL_RESET_INSET, -1.25)
        high_z = min(zmax - GLOBAL_RESET_INSET, 0.50)
        return self._sample_box_state(
            (xmin + GLOBAL_RESET_INSET, xmax - GLOBAL_RESET_INSET, low_z, high_z),
            vel_abs=0.35,
            theta_abs=0.14,
            omega_abs=0.10,
        )

    def _sample_simple_low_uniform_state(self) -> np.ndarray:
        xmin, xmax = self.layout.xlim
        zmin, _zmax = self.layout.zlim
        low_z = max(zmin + GLOBAL_RESET_INSET, -1.25)
        high_z = 0.50
        return self._sample_box_state(
            (xmin + GLOBAL_RESET_INSET, xmax - GLOBAL_RESET_INSET, low_z, high_z),
            vx_abs=self.simple_low_uniform_vx_abs,
            vz_abs=self.simple_low_uniform_vz_abs,
            theta_abs=0.14,
            omega_abs=0.10,
        )

    def _sample_simple_under_box_state(
        self,
        box_xz: Tuple[float, float, float, float],
        *,
        vz_low: float,
        vz_high: float,
        xdot_zero_frac: float,
    ) -> np.ndarray:
        vx_abs = 0.0 if self._rng.random() < xdot_zero_frac else self.simple_under_vx_abs
        return self._sample_box_state_vz_range(
            box_xz,
            vx_abs=vx_abs,
            vz_low=vz_low,
            vz_high=vz_high,
        )

    def _sample_simple_gap_state(self) -> np.ndarray:
        idx = int(self._rng.integers(0, len(SIMPLE_UNDER3_GAP_BOXES_XZ)))
        return self._sample_box_state_vz_range(
            SIMPLE_UNDER3_GAP_BOXES_XZ[idx],
            vx_abs=self.simple_gap_vx_abs,
            vz_low=self.simple_gap_vz_low,
            vz_high=self.simple_gap_vz_high,
        )

    def _sample_simple_goal_side_state(self) -> np.ndarray:
        idx = int(self._rng.integers(0, len(SIMPLE_UNDER3_GOAL_SIDE_BOXES_XZ)))
        return self._sample_box_state(SIMPLE_UNDER3_GOAL_SIDE_BOXES_XZ[idx])

    def _sample_simple_fork_state(self) -> np.ndarray:
        side = -1.0 if self._rng.random() < 0.5 else 1.0
        x_abs = float(self._rng.uniform(self.simple_fork_x_low, self.simple_fork_x_high))
        vx_abs = float(self._rng.uniform(self.simple_fork_vx_low, self.simple_fork_vx_high))
        return np.asarray(
            [
                side * x_abs,
                side * vx_abs,
                float(self._rng.uniform(self.simple_fork_z_low, self.simple_fork_z_high)),
                float(self._rng.uniform(self.simple_fork_vz_low, self.simple_fork_vz_high)),
                float(self._rng.uniform(-self.simple_fork_theta_abs, self.simple_fork_theta_abs)),
                float(self._rng.uniform(-self.simple_fork_omega_abs, self.simple_fork_omega_abs)),
            ],
            dtype=np.float32,
        )

    def _sample_simple_under3_state(self) -> np.ndarray:
        weights = self._normalized_choice_weights(self.reset_simple_under3_weights, "simple_under3 reset")
        branch = int(self._rng.choice(len(weights), p=weights))
        if branch == 0:
            return self._sample_simple_under_box_state(
                SIMPLE_UNDER3_BAR_BOX_XZ,
                vz_low=self.simple_bar_vz_low,
                vz_high=self.simple_bar_vz_high,
                xdot_zero_frac=self.simple_bar_xdot_zero_frac,
            )
        if branch == 1:
            return self._sample_simple_under_box_state(
                SIMPLE_UNDER3_LEFT_BLOCK_BOX_XZ,
                vz_low=self.simple_block_vz_low,
                vz_high=self.simple_block_vz_high,
                xdot_zero_frac=self.simple_block_xdot_zero_frac,
            )
        if branch == 2:
            return self._sample_simple_under_box_state(
                SIMPLE_UNDER3_RIGHT_BLOCK_BOX_XZ,
                vz_low=self.simple_block_vz_low,
                vz_high=self.simple_block_vz_high,
                xdot_zero_frac=self.simple_block_xdot_zero_frac,
            )
        if branch == 3:
            return self._sample_simple_gap_state()
        if branch == 4:
            return self._sample_simple_low_uniform_state()
        if branch == 5:
            return self._sample_goal_near_state()
        if branch == 6:
            return self._sample_simple_goal_side_state()
        return self._sample_simple_fork_state()

    def _sample_safety_under50_state(self) -> np.ndarray:
        weights = self._normalized_choice_weights(self.reset_safety_under50_weights, "safety_under50 reset")
        branch = int(self._rng.choice(7, p=weights))
        if branch == 0:
            return self._sample_simple_under_box_state(
                SIMPLE_UNDER3_BAR_BOX_XZ,
                vz_low=self.simple_bar_vz_low,
                vz_high=self.simple_bar_vz_high,
                xdot_zero_frac=self.simple_bar_xdot_zero_frac,
            )
        if branch == 1:
            return self._sample_simple_under_box_state(
                SIMPLE_UNDER3_LEFT_BLOCK_BOX_XZ,
                vz_low=self.simple_block_vz_low,
                vz_high=self.simple_block_vz_high,
                xdot_zero_frac=self.simple_block_xdot_zero_frac,
            )
        if branch == 2:
            return self._sample_simple_under_box_state(
                SIMPLE_UNDER3_RIGHT_BLOCK_BOX_XZ,
                vz_low=self.simple_block_vz_low,
                vz_high=self.simple_block_vz_high,
                xdot_zero_frac=self.simple_block_xdot_zero_frac,
            )
        if branch == 3:
            return self._sample_simple_gap_state()
        if branch == 4:
            return self._sample_simple_low_uniform_state()
        if branch == 5:
            return self._sample_low_mid_state()
        return self._sample_goal_near_state()

    def _sample_low_state(self) -> np.ndarray:
        weights = self._normalized_choice_weights(self.reset_low_component_weights, "low component reset")
        branch = int(self._rng.choice(6, p=weights))
        if branch == 0:
            return self._sample_from_boxes(
                LOW_PAPER_JITTER_BOXES_XZ,
                vx_abs=self.low_paper_jitter_vx_abs,
                vz_abs=self.low_paper_jitter_vz_abs,
            )
        if branch == 1:
            return self._sample_paper_pinpoint_vel_state()
        if branch == 2:
            return self._sample_from_boxes(
                LOW_UNDER_BAR_BOXES_XZ,
                vx_abs=self.low_under_bar_vx_abs,
                vz_abs=self.low_under_bar_vz_abs,
            )
        if branch == 3:
            return self._sample_from_boxes(
                LOW_UNDER_BLOCK_BOXES_XZ,
                vx_abs=self.low_under_blocks_vx_abs,
                vz_abs=self.low_under_blocks_vz_abs,
            )
        if branch == 4:
            return self._sample_from_boxes(
                self._low_inner_gap_boxes(),
                vx_abs=self.low_inner_gaps_vx_abs,
                vz_abs=self.low_inner_gaps_vz_abs,
            )
        return self._sample_low_global_state()

    def _sample_low_mid_state(self) -> np.ndarray:
        return self._sample_from_boxes(LOW_MID_BOXES_XZ)

    def _sample_gap_mix_state(self) -> np.ndarray:
        weights = self._normalized_choice_weights(GAP_MIX_BRANCH_WEIGHTS, "gap_mix reset")
        branch = int(self._rng.choice(2, p=weights))
        if branch == 0:
            return self._sample_from_boxes(self._low_inner_gap_boxes())
        return self._sample_from_boxes(LOW_UNDER_BLOCK_BOXES_XZ)

    def _low_inner_gap_boxes(self) -> Tuple[Tuple[float, float, float, float], ...]:
        if self.layout_name == "corridor_v3":
            return LOW_INNER_GAP_BOXES_XZ_V3
        return LOW_INNER_GAP_BOXES_XZ

    def _sample_global_state(self) -> np.ndarray:
        xmin, xmax = self.layout.xlim
        zmin, zmax = self.layout.zlim
        low = np.array(
            [
                xmin + GLOBAL_RESET_INSET,
                -0.45,
                zmin + GLOBAL_RESET_INSET,
                -0.45,
                -0.18,
                -0.12,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                xmax - GLOBAL_RESET_INSET,
                0.45,
                zmax - GLOBAL_RESET_INSET,
                0.45,
                0.18,
                0.12,
            ],
            dtype=np.float32,
        )
        return self._rng.uniform(low, high).astype(np.float32)

    @staticmethod
    def _normalized_choice_weights(weights: np.ndarray, name: str) -> np.ndarray:
        if np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError(f"{name} fractions must be non-negative and not all zero")
        return weights / weights.sum()

    def _sample_candidate_by_mode(self) -> np.ndarray:
        if self.reset_mode == "start_band":
            return self._sample_start_band_state()
        if self.reset_mode == "global_safe":
            return self._sample_global_state()
        if self.reset_mode == "route_mix":
            weights = self._normalized_choice_weights(
                np.array(
                    [
                        self.reset_route_global_frac,
                        self.reset_route_near_frac,
                        self.reset_route_gap_frac,
                    ],
                    dtype=np.float64,
                ),
                "route_mix reset",
            )
            branch = self._rng.choice(3, p=weights)
            if branch == 0:
                return self._sample_global_state()
            if branch == 1:
                return self._sample_route_near_goal_state()
            return self._sample_route_gap_state()

        if self.reset_mode == "under_mix":
            weights = self._normalized_choice_weights(
                np.array([self.reset_under_global_frac, self.reset_under_frac], dtype=np.float64),
                "under_mix reset",
            )
            branch = self._rng.choice(2, p=weights)
            if branch == 0:
                return self._sample_global_state()
            return self._sample_under_state()

        if self.reset_mode == "low_mix":
            weights = self._normalized_choice_weights(
                np.array([self.reset_low_frac, self.reset_low_mid_frac, self.reset_low_near_frac], dtype=np.float64),
                "low_mix reset",
            )
            branch = int(self._rng.choice(3, p=weights))
            if branch == 0:
                return self._sample_low_state()
            if branch == 1:
                return self._sample_low_mid_state()
            return self._sample_goal_near_state()

        if self.reset_mode == "gap_mix":
            return self._sample_gap_mix_state()

        if self.reset_mode == "simple_under3":
            return self._sample_simple_under3_state()

        if self.reset_mode == "safety_under50":
            return self._sample_safety_under50_state()

        weights = self._normalized_choice_weights(
            np.array(
                [self.reset_mix_start_frac, self.reset_mix_global_frac, self.reset_mix_goal_frac],
                dtype=np.float64,
            ),
            "mix reset",
        )
        branch = self._rng.choice(3, p=weights)
        if branch == 0:
            return self._sample_start_band_state()
        if branch == 1:
            return self._sample_global_state()
        return self._sample_goal_near_state()

    def reset(self, options: Optional[Dict[str, float]] = None) -> np.ndarray:
        if options:
            candidate = self._sample_start_band_state()
            for key, idx in [
                ("init_x", self.X),
                ("init_vx", self.XDOT),
                ("init_z", self.Z),
                ("init_vz", self.ZDOT),
                ("init_theta", self.THETA),
                ("init_omega", self.THETA_DOT),
                ("init_theta_dot", self.THETA_DOT),
            ]:
                if key in options:
                    candidate[idx] = float(options[key])
            self.state = candidate.astype(np.float32)
        else:
            candidate = None
            for _ in range(2000):
                proposal = self._sample_candidate_by_mode()
                if self.h_phys(proposal) < self.init_h_threshold and not self._is_in_goal(proposal):
                    candidate = proposal
                    break
            if candidate is None:
                raise RuntimeError("Failed to sample a safe non-goal Quad2D-stab start.")
            self.state = candidate

        self._t = 0
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0
        self._episode_goal_steps = 0
        self._episode_total_goal_steps = 0
        self._episode_max_goal_streak = 0
        self._episode_success = False
        self._episode_success_step = None
        self._episode_last_action_error_norm = float("nan")
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        action_raw = (action + 1.0) / 2.0

        self.state = quad2d_step(
            self.state,
            action_raw,
            self.dt,
            self.m,
            self.I,
            self.g,
            self.thrust_scale,
            self.torque_scale,
        )

        h_phys = self.h_phys(self.state)
        h_val = self._shape_h(h_phys)
        binary_cost = float(h_phys > 0.0)
        reward = self._compute_reward(self.state, action_raw)

        in_goal = self._is_in_goal(self.state)
        if in_goal:
            self._episode_goal_steps += 1
            self._episode_total_goal_steps += 1
            self._episode_max_goal_streak = max(self._episode_max_goal_streak, self._episode_goal_steps)
        else:
            self._episode_goal_steps = 0

        success_now = False
        if (
            self.success_mode == "dwell_box"
            and (not self._episode_success)
            and self._episode_goal_steps >= self.goal_dwell_steps
        ):
            self._episode_success = True
            self._episode_success_step = self._t + 1
            success_now = True

        collision = self._is_terminal_collision(self.state)
        terminated = collision
        self._t += 1
        truncated = self._t >= self.max_episode_steps
        if self.success_mode == "dwell_box" and self.terminate_on_success and self._episode_success:
            terminated = True
        done = terminated or truncated

        self._episode_reward += reward
        self._episode_cost += binary_cost
        self._episode_length += 1

        pos_err = float(np.linalg.norm(self.state[[self.X, self.Z]] - self.goal))
        vel_norm = float(np.linalg.norm(self.state[[self.XDOT, self.ZDOT]]))
        h_nominal = self.h_nominal_collision(self.state)
        terminal_state_norm = self._terminal_state_norm_to_ref(self.state)
        terminal_action_error_norm = float(np.linalg.norm(action_raw - self.a_ref))
        self._episode_last_action_error_norm = terminal_action_error_norm
        dwell_success = self._episode_max_goal_streak >= self.goal_dwell_steps and h_nominal <= 0.0
        info = {
            "h": float(h_val),
            "h_phys": float(h_phys),
            "h_nominal": float(h_nominal),
            "binary_cost": binary_cost,
            "x": float(self.state[self.X]),
            "z": float(self.state[self.Z]),
            "pos_error": pos_err,
            "vel_norm": vel_norm,
            "in_goal": bool(in_goal),
            "success": bool(self._full_state_success(self.state) if self.success_mode == "full_state_terminal" else (self._episode_success and h_nominal <= 0.0)),
            "dwell_success": bool(dwell_success),
            "terminal_state_norm_to_ref": float(terminal_state_norm),
            "terminal_state_error_norm": float(terminal_state_norm),
            "terminal_action_error_norm": terminal_action_error_norm,
            "success_now": bool(success_now),
            "collision": bool(collision),
            "success_terminal": bool(self.success_mode == "dwell_box" and self.terminate_on_success and self._episode_success),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_reward": self._episode_reward,
            "episode_cost": self._episode_cost,
            "episode_length": self._episode_length,
            "episode_goal_steps": self._episode_goal_steps,
            "episode_total_goal_steps": self._episode_total_goal_steps,
            "episode_max_goal_streak": self._episode_max_goal_streak,
            "success_step": -1 if self._episode_success_step is None else self._episode_success_step,
        }
        return self._get_obs(), reward, h_val, binary_cost, done, info

    @property
    def episode_info(self) -> Dict[str, float]:
        terminal_state_norm = self._terminal_state_norm_to_ref(self.state)
        terminal_action_error_norm = float(self._episode_last_action_error_norm)
        dwell_success = (
            self._episode_max_goal_streak >= self.goal_dwell_steps
            and self.h_nominal_collision(self.state) <= 0.0
        )
        if self.success_mode == "full_state_terminal":
            episode_success = terminal_state_norm < self.success_radius and self.h_nominal_collision(self.state) <= 0.0
            success_step = float(self._episode_length) if episode_success else -1.0
        else:
            episode_success = dwell_success
            success_step = -1.0 if self._episode_success_step is None else float(self._episode_success_step)
        return {
            "reward": float(self._episode_reward),
            "cost": float(self._episode_cost),
            "length": float(self._episode_length),
            "success": float(episode_success),
            "dwell_success": float(dwell_success),
            "goal_steps": float(self._episode_goal_steps),
            "total_goal_steps": float(self._episode_total_goal_steps),
            "max_goal_streak": float(self._episode_max_goal_streak),
            "goal_dwell_frac": float(self._episode_total_goal_steps / max(self._episode_length, 1)),
            "terminal_state_norm_to_ref": float(terminal_state_norm),
            "terminal_state_error_norm": float(terminal_state_norm),
            "terminal_action_error_norm": terminal_action_error_norm,
            "success_step": success_step,
        }


FIXED_PAPER_START = {
    "init_x": 0.0,
    "init_vx": 0.0,
    "init_z": -1.08,
    "init_vz": 0.0,
    "init_theta": 0.0,
    "init_omega": 0.0,
}

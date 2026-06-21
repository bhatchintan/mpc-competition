from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


# --- CONFIGURATION ---
MPC_CONFIG = {
    "dt": 0.1,
    "horizon": 30,  # Increased horizon to better predict moving obstacles
    "near_goal_distance": 1.2,
}

# Expanded grid for better precision at high speeds
CONTROL_GRID = {
    "slow_speeds": [0.0, 0.4, 0.8, 1.2],
    "cruise_speeds": [0.8, 1.2, 1.8, 2.4, 3.0, 3.5], 
    "turn_rates": [-1.2, -0.8, -0.4, -0.1, 0.0, 0.1, 0.4, 0.8, 1.2],
}

# Tuned for "Time to Goal" (minimized travel time)
WEIGHTS = {
    "goal_position": 1.2,    # Slightly lower to favor speed over millimeter precision
    "goal_heading": 0.05,
    "turn_rate": 0.02,       # Heavily reduced to allow aggressive maneuvering
    "terminal_goal": 40.0,
    "collision": 10000.0,    # Increased to ensure we don't hit moving targets
    "collision_depth": 2000.0,
    "near_obstacle": 25.0,   # Increased to push the robot further away from obstacles
    "far_obstacle": 1.0,
    "stopping": 5.0,         # Reduced to prevent premature slowing
}

SAFETY = {
    "near_obstacle_distance": 0.5,
    "far_obstacle_distance": 1.5,
}


class Agent:
    def __init__(self) -> None:
        self.previous_control = np.array([0.0, 0.0], dtype=float)

    def act(self, observation: dict[str, Any]) -> list[float]:
        start_wall_time = time.time()
        
        state = np.array(observation["state"], dtype=float)
        goal = np.array(observation["goal"], dtype=float)
        active_obstacles = get_active_obstacles(observation)
        control_limits = observation["control_limits"]
        robot_radius = float(observation["robot_radius"])

        # Generate all candidates as a matrix (N, 2)
        candidates = get_candidate_matrix(state, goal, control_limits)
        
        # Vectorized evaluation: Simulate all candidates at once
        costs = evaluate_all_rollouts(
            state,
            goal,
            candidates,
            self.previous_control,
            active_obstacles,
            control_limits,
            robot_radius,
        )

        best_idx = np.argmin(costs)
        best_control = candidates[best_idx]

        self.previous_control = apply_control_limits(
            best_control, self.previous_control, control_limits
        )
        
        print(f"act() wall-clock: {time.time() - start_wall_time:.4f}s")
        return [float(best_control[0]), float(best_control[1])]


def get_candidate_matrix(state: np.ndarray, goal: np.ndarray, control_limits: dict[str, float]) -> np.ndarray:
    distance_to_goal = np.linalg.norm(goal[:2] - state[:2])
    speeds = CONTROL_GRID["slow_speeds"] if distance_to_goal < MPC_CONFIG["near_goal_distance"] else CONTROL_GRID["cruise_speeds"]
    
    target_heading = math.atan2(goal[1] - state[1], goal[0] - state[0])
    heading_error = normalize_angle(target_heading - state[2])
    proportional_turn = np.clip(2.0 * heading_error, -1.0, 1.0)

    # Combine grid with proportional heuristics
    turn_rates = np.array(CONTROL_GRID["turn_rates"])
    extra_turns = np.array([proportional_turn, proportional_turn - 0.3, proportional_turn + 0.3])
    all_turns = np.unique(np.concatenate([turn_rates, extra_turns]))
    all_speeds = np.array(speeds)

    # Meshgrid to create (N, 2) matrix
    s, t = np.meshgrid(all_speeds, all_turns)
    candidates = np.stack([s.flatten(), t.flatten()], axis=1)
    
    # Clip to hard limits
    candidates[:, 0] = np.clip(candidates[:, 0], control_limits["v_min"], control_limits["v_max"])
    candidates[:, 1] = np.clip(candidates[:, 1], -control_limits["omega_max"], control_limits["omega_max"])
    
    return candidates


def evaluate_all_rollouts(
    state: np.ndarray,
    goal: np.ndarray,
    candidates: np.ndarray,
    previous_control: np.ndarray,
    active_obstacles: list[dict[str, float]],
    control_limits: dict[str, float],
    robot_radius: float,
) -> np.ndarray:
    num_candidates = candidates.shape[0]
    horizon = MPC_CONFIG["horizon"]
    dt = MPC_CONFIG["dt"]
    
    # State for all candidates: (num_candidates, 3) -> [x, y, theta]
    curr_states = np.tile(state, (num_candidates, 1))
    # Control for all candidates: (num_candidates, 2) -> [v, omega]
    curr_controls = np.tile(previous_control, (num_candidates, 1))
    
    total_costs = np.zeros(num_candidates)
    
    # Pre-calculate obstacle properties for speed
    obs_data = []
    for ob in active_obstacles:
        obs_data.append({
            "cx": ob["cx"], "cy": ob["cy"], "v": ob["v"], 
            "angle": ob["angle"], "omega": ob["omega"],
            "w": ob["width"], "h": ob["height"]
        })

    for step in range(1, horizon + 1):
        t_pred = step * dt
        
        # 1. Update Controls (ramp limits)
        curr_controls = apply_control_limits_vectorized(candidates, curr_controls, control_limits)
        
        # 2. Update States (unicycle model)
        v = curr_controls[:, 0]
        omega = curr_controls[:, 1]
        theta = curr_states[:, 2]
        
        curr_states[:, 0] += v * np.cos(theta) * dt
        curr_states[:, 1] += v * np.sin(theta) * dt
        curr_states[:, 2] = np.arctan2(np.sin(theta + omega * dt), np.cos(theta + omega * dt))
        
        # 3. Goal Costs (Vectorized)
        dist_to_goal = np.linalg.norm(curr_states[:, :2] - goal[:2], axis=1)
        heading_err = np.abs(normalize_angle(goal[2] - curr_states[:, 2]))
        
        total_costs += WEIGHTS["goal_position"] * dist_to_goal
        total_costs += WEIGHTS["goal_heading"] * heading_err
        total_costs += WEIGHTS["turn_rate"] * np.abs(omega)
        
        # 4. Obstacle Costs (Vectorized over candidates)
        for ob in obs_data:
            # Predict obstacle position at time t_pred
            p_cx = ob["cx"] + ob["v"] * math.cos(ob["angle"]) * t_pred
            p_cy = ob["cy"] + ob["v"] * math.sin(ob["angle"]) * t_pred
            p_angle = normalize_angle(ob["angle"] + ob["omega"] * t_pred)
            
            # Compute clearance for all candidates against this obstacle
            clearances = rectangle_clearance_vectorized(curr_states, p_cx, p_cy, p_angle, ob["w"], ob["h"], robot_radius)
            total_costs += compute_collision_cost_vectorized(clearances)

    # Terminal costs
    final_dist = np.linalg.norm(curr_states[:, :2] - goal[:2], axis=1)
    total_costs += WEIGHTS["terminal_goal"] * final_dist
    
    # Stopping penalty
    stop_mask = (candidates[:, 0] < 0.2) & (final_dist > 0.7)
    total_costs[stop_mask] += WEIGHTS["stopping"]
    
    return total_costs


def apply_control_limits_vectorized(requested: np.ndarray, previous: np.ndarray, limits: dict[str, float]) -> np.ndarray:
    dt = MPC_CONFIG["dt"]
    max_v_change = limits["a_max"] * dt
    max_w_change = limits["omega_dot_max"] * dt
    
    # Requested is already clipped in get_candidate_matrix
    v_min = np.maximum(previous[:, 0] - max_v_change, limits["v_min"])
    v_max = np.minimum(previous[:, 0] + max_v_change, limits["v_max"])
    omega_min = np.maximum(previous[:, 1] - max_w_change, -limits["omega_max"])
    omega_max = np.minimum(previous[:, 1] + max_w_change, limits["omega_max"])
    
    return np.stack([
        np.clip(requested[:, 0], v_min, v_max),
        np.clip(requested[:, 1], omega_min, omega_max)
    ], axis=1)


def rectangle_clearance_vectorized(states: np.ndarray, cx: float, cy: float, angle: float, w: float, h: float, robot_radius: float) -> np.ndarray:
    # Vectorized transform to obstacle local frame
    tx = states[:, 0] - cx
    ty = states[:, 1] - cy
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    
    lx = tx * cos_a - ty * sin_a
    ly = tx * sin_a + ty * cos_a
    
    ox = np.abs(lx) - w / 2.0
    oy = np.abs(ly) - h / 2.0
    
    # Signed distance to rectangle
    outside_dist = np.hypot(np.maximum(ox, 0.0), np.maximum(oy, 0.0))
    inside_dist = np.minimum(np.maximum(ox, oy), 0.0)
    
    return outside_dist + inside_dist - robot_radius


def compute_collision_cost_vectorized(clearance: np.ndarray) -> np.ndarray:
    costs = np.zeros_like(clearance)
    
    # Case 1: Collision
    mask_coll = clearance < 0.0
    costs[mask_coll] = WEIGHTS["collision"] + WEIGHTS["collision_depth"] * np.abs(clearance[mask_coll])
    
    # Case 2: Near
    mask_near = (clearance >= 0.0) & (clearance < SAFETY["near_obstacle_distance"])
    costs[mask_near] = WEIGHTS["near_obstacle"] * (SAFETY["near_obstacle_distance"] - clearance[mask_near])**2
    
    # Case 3: Far
    mask_far = (clearance >= SAFETY["near_obstacle_distance"]) & (clearance < SAFETY["far_obstacle_distance"])
    costs[mask_far] = WEIGHTS["far_obstacle"] * (SAFETY["far_obstacle_distance"] - clearance[mask_far])**2
    
    return costs


def get_active_obstacles(observation: dict[str, Any]) -> list[dict[str, float]]:
    active_indices = set(observation["active_obstacle_indices"])
    return [ob for i, ob in enumerate(observation["obstacles"]) if i in active_indices]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def apply_control_limits(req: np.ndarray, prev: np.ndarray, limits: dict[str, float]) -> np.ndarray:
    # Scalar version for the final state update
    dt = MPC_CONFIG["dt"]
    v = np.clip(req[0], prev[0] - limits["a_max"]*dt, prev[0] + limits["a_max"]*dt)
    v = np.clip(v, limits["v_min"], limits["v_max"])
    w = np.clip(req[1], prev[1] - limits["omega_dot_max"]*dt, prev[1] + limits["omega_dot_max"]*dt)
    w = np.clip(w, -limits["omega_max"], limits["omega_max"])
    return np.array([v, w])

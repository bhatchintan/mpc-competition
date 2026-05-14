from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np


class AgentProtocol(Protocol):
    def act(self, observation: dict[str, Any]) -> list[float] | tuple[float, float]:
        pass


@dataclass(frozen=True)
class ControlLimits:
    v_min: float = 0.0
    v_max: float = 3.0
    omega_max: float = 1.0
    a_max: float = 1.5
    omega_dot_max: float = 1.5


@dataclass(frozen=True)
class Scenario:
    name: str
    scene: str
    geometry: str
    max_obstacles: int
    n_circles: int


@dataclass(frozen=True)
class Obstacle:
    cx: float
    cy: float
    width: float
    height: float
    angle: float
    v: float = 0.0
    omega: float = 0.0


@dataclass(frozen=True)
class ProblemConfig:
    dt: float = 0.1
    max_steps: int = 500
    robot_radius: float = 0.4
    goal_threshold: float = 0.5
    action_time_limit_s: float = 0.2


WAYPOINTS: tuple[tuple[float, float, float], ...] = (
    (5.9, 2.7, 1.57),
    (10.0, 7.0, 1.57),
    (5.0, 12.0, 3.14),
    (13.6, 13.2, 0.0),
)
START_POSE: tuple[float, float, float] = (1.0, 1.0, 0.0)
CONTROL_LIMITS = ControlLimits()
PROBLEM_CONFIG = ProblemConfig()


def scenarios() -> list[Scenario]:
    return [
        Scenario("static_multi", "static", "multi_circle", 13, 3),
        Scenario("dynamic_multi", "dynamic", "multi_circle", 8, 3),
    ]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def step_unicycle(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    return np.array(
        [
            state[0] + control[0] * math.cos(state[2]) * dt,
            state[1] + control[0] * math.sin(state[2]) * dt,
            normalize_angle(state[2] + control[1] * dt),
        ],
        dtype=float,
    )


def obstacle_state(obstacle: Obstacle, elapsed_time: float) -> Obstacle:
    return Obstacle(
        cx=obstacle.cx + obstacle.v * math.cos(obstacle.angle) * elapsed_time,
        cy=obstacle.cy + obstacle.v * math.sin(obstacle.angle) * elapsed_time,
        width=obstacle.width,
        height=obstacle.height,
        angle=normalize_angle(obstacle.angle + obstacle.omega * elapsed_time),
        v=obstacle.v,
        omega=obstacle.omega,
    )


def static_obstacles() -> list[Obstacle]:
    return [
        Obstacle(3.3, 1.8, 0.7, 2.1, 0.15),
        Obstacle(6.1, 1.4, 2.4, 0.6, -0.2),
        Obstacle(7.5, 3.6, 0.9, 2.2, 0.25),
        Obstacle(2.1, 5.6, 2.2, 0.7, 0.45),
        Obstacle(5.1, 5.1, 0.8, 2.7, -0.1),
        Obstacle(9.0, 5.3, 2.6, 0.8, 0.3),
        Obstacle(12.2, 5.1, 0.8, 2.4, -0.25),
        Obstacle(3.1, 8.7, 0.7, 2.2, 0.1),
        Obstacle(6.9, 8.8, 2.7, 0.7, -0.25),
        Obstacle(10.6, 9.2, 0.8, 2.5, 0.2),
        Obstacle(2.6, 12.0, 2.1, 0.8, 0.15),
        Obstacle(8.2, 12.0, 0.9, 2.2, -0.35),
        Obstacle(11.6, 12.4, 2.2, 0.7, 0.1),
    ]


def dynamic_obstacles() -> list[Obstacle]:
    base_obstacles = static_obstacles()
    movers = [
        Obstacle(1.8, 3.6, 0.7, 0.9, 0.0, 0.45, 0.0),
        Obstacle(4.7, 3.7, 1.1, 0.7, 1.57, 0.25, 0.0),
        Obstacle(8.4, 2.1, 0.7, 1.2, 2.6, 0.35, 0.0),
        Obstacle(11.4, 3.1, 1.0, 0.8, 2.85, 0.28, 0.0),
        Obstacle(13.1, 7.1, 0.7, 1.1, -1.57, 0.22, 0.0),
        Obstacle(7.5, 7.4, 0.9, 0.9, 0.6, 0.18, 0.12),
        Obstacle(4.1, 10.5, 1.0, 0.8, -0.5, 0.24, 0.0),
    ]
    return base_obstacles + movers


def load_obstacles(scene: str) -> list[Obstacle]:
    if scene == "static":
        return static_obstacles()
    if scene == "dynamic":
        return dynamic_obstacles()
    raise ValueError(f"Unknown scene: {scene}")


def rect_to_single_circle(width: float, height: float) -> float:
    return 0.5 * math.hypot(width, height)


def rect_to_multi_circles(obstacle: Obstacle, n_circles: int) -> list[dict[str, float]]:
    if n_circles < 1:
        raise ValueError("n_circles must be positive")

    if obstacle.width >= obstacle.height:
        major_length = obstacle.width
        minor_length = obstacle.height
        major_angle = obstacle.angle
    else:
        major_length = obstacle.height
        minor_length = obstacle.width
        major_angle = obstacle.angle + math.pi / 2.0

    segment_length = major_length / n_circles
    radius = math.hypot(segment_length / 2.0, minor_length / 2.0)
    start = -major_length / 2.0 + segment_length / 2.0
    circles = []
    for index in range(n_circles):
        local_x = start + index * segment_length
        circles.append(
            {
                "cx": obstacle.cx + local_x * math.cos(major_angle),
                "cy": obstacle.cy + local_x * math.sin(major_angle),
                "radius": radius,
                "v": obstacle.v,
                "angle": obstacle.angle,
            }
        )
    return circles


def obstacle_circles(
    obstacle: Obstacle, geometry: str, n_circles: int
) -> list[dict[str, float]]:
    if geometry == "single_circle":
        return [
            {
                "cx": obstacle.cx,
                "cy": obstacle.cy,
                "radius": rect_to_single_circle(obstacle.width, obstacle.height),
                "v": obstacle.v,
                "angle": obstacle.angle,
            }
        ]
    if geometry == "multi_circle":
        return rect_to_multi_circles(obstacle, n_circles)
    raise ValueError(f"Unknown obstacle geometry: {geometry}")


def nearest_obstacles(
    position: np.ndarray, obstacles: list[Obstacle], max_obstacles: int
) -> tuple[list[Obstacle], list[int]]:
    distances = [
        (math.hypot(position[0] - obstacle.cx, position[1] - obstacle.cy), index)
        for index, obstacle in enumerate(obstacles)
    ]
    distances.sort(key=lambda item: item[0])
    selected = distances[:max_obstacles]
    indices = [index for _, index in selected]
    return [obstacles[index] for index in indices], indices


def rectangle_clearance(
    point: np.ndarray, obstacle: Obstacle, robot_radius: float
) -> float:
    translated_x = point[0] - obstacle.cx
    translated_y = point[1] - obstacle.cy
    cos_angle = math.cos(-obstacle.angle)
    sin_angle = math.sin(-obstacle.angle)
    local_x = translated_x * cos_angle - translated_y * sin_angle
    local_y = translated_x * sin_angle + translated_y * cos_angle

    dx = abs(local_x) - obstacle.width / 2.0
    dy = abs(local_y) - obstacle.height / 2.0
    outside_distance = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside_distance = min(max(dx, dy), 0.0)
    return outside_distance + inside_distance - robot_radius


def validate_control(action: Any) -> np.ndarray:
    if not isinstance(action, (list, tuple, np.ndarray)):
        raise TypeError("Agent.act must return a two-item sequence")
    if len(action) != 2:
        raise ValueError("Agent.act must return [v, omega]")

    control = np.array([float(action[0]), float(action[1])], dtype=float)
    if not np.all(np.isfinite(control)):
        raise ValueError("Control values must be finite")
    return control


def apply_control_limits(
    requested_control: np.ndarray, previous_control: np.ndarray, dt: float
) -> np.ndarray:
    limited = requested_control.copy()
    limited[0] = float(np.clip(limited[0], CONTROL_LIMITS.v_min, CONTROL_LIMITS.v_max))
    limited[1] = float(
        np.clip(limited[1], -CONTROL_LIMITS.omega_max, CONTROL_LIMITS.omega_max)
    )

    v_delta = CONTROL_LIMITS.a_max * dt
    omega_delta = CONTROL_LIMITS.omega_dot_max * dt
    limited[0] = float(
        np.clip(
            limited[0], previous_control[0] - v_delta, previous_control[0] + v_delta
        )
    )
    limited[1] = float(
        np.clip(
            limited[1],
            previous_control[1] - omega_delta,
            previous_control[1] + omega_delta,
        )
    )
    return limited


def build_observation(
    step: int,
    state: np.ndarray,
    goal: np.ndarray,
    waypoint_index: int,
    current_obstacles: list[Obstacle],
    active_indices: list[int],
    active_obstacles: list[Obstacle],
    scenario: Scenario,
) -> dict[str, Any]:
    circles = []
    for obstacle in active_obstacles:
        circles.extend(
            obstacle_circles(obstacle, scenario.geometry, scenario.n_circles)
        )

    return {
        "step": step,
        "time": step * PROBLEM_CONFIG.dt,
        "dt": PROBLEM_CONFIG.dt,
        "state": state.tolist(),
        "goal": goal.tolist(),
        "waypoint_index": waypoint_index,
        "waypoints": [list(waypoint) for waypoint in WAYPOINTS],
        "obstacles": [asdict(obstacle) for obstacle in current_obstacles],
        "active_obstacle_indices": active_indices,
        "active_obstacle_circles": circles,
        "geometry": scenario.geometry,
        "scene": scenario.scene,
        "robot_radius": PROBLEM_CONFIG.robot_radius,
        "control_limits": asdict(CONTROL_LIMITS),
    }


def score_result(result: dict[str, Any]) -> float:
    controls = np.array(result["controls"], dtype=float)
    trajectory = np.array(result["trajectory"], dtype=float)
    clearances = np.array(result["min_clearances"], dtype=float)
    action_times = np.array(result["action_times"], dtype=float)

    waypoints_reached = int(result["waypoints_reached"])
    target_index = min(waypoints_reached, len(WAYPOINTS) - 1)
    target = np.array(WAYPOINTS[target_index], dtype=float)
    initial_distance = float(np.linalg.norm(trajectory[0, :2] - target[:2]))
    final_distance = float(np.linalg.norm(trajectory[-1, :2] - target[:2]))
    progress = max(0.0, initial_distance - final_distance)

    waypoint_score = 250.0 * waypoints_reached
    finish_bonus = 250.0 if waypoints_reached == len(WAYPOINTS) else 0.0
    progress_score = 60.0 * progress
    proximity_score = max(0.0, 80.0 - 20.0 * final_distance)

    collision_depth = np.maximum(0.0, -clearances)
    collision_penalty = 200.0 * float(np.sum(collision_depth))
    collision_penalty += 25.0 * int(np.count_nonzero(collision_depth > 0.0))

    if len(controls) > 1:
        smoothness_penalty = 4.0 * float(
            np.sum(np.linalg.norm(np.diff(controls, axis=0), axis=1))
        )
    else:
        smoothness_penalty = 0.0
    effort_penalty = 0.25 * float(np.sum(np.linalg.norm(controls, axis=1)))

    slow_steps = int(
        np.count_nonzero(action_times > PROBLEM_CONFIG.action_time_limit_s)
    )
    speed_penalty = 15.0 * slow_steps

    score = (
        waypoint_score
        + finish_bonus
        + progress_score
        + proximity_score
        - collision_penalty
        - smoothness_penalty
        - effort_penalty
        - speed_penalty
    )
    return round(score, 3)


def run_simulation(agent: AgentProtocol, scenario: Scenario) -> dict[str, Any]:
    obstacles = load_obstacles(scenario.scene)
    state = np.array(START_POSE, dtype=float)
    previous_control = np.array([0.0, 0.0], dtype=float)
    waypoint_index = 0

    trajectory = [state.tolist()]
    controls: list[list[float]] = []
    requested_controls: list[list[float]] = []
    action_times: list[float] = []
    min_clearances: list[float] = []
    obstacle_history: list[list[dict[str, float]]] = []
    active_obstacle_history: list[list[int]] = []

    for step in range(PROBLEM_CONFIG.max_steps):
        if waypoint_index >= len(WAYPOINTS):
            break

        goal = np.array(WAYPOINTS[waypoint_index], dtype=float)
        elapsed_time = step * PROBLEM_CONFIG.dt
        current_obstacles = [
            obstacle_state(obstacle, elapsed_time) for obstacle in obstacles
        ]
        active_obstacles, active_indices = nearest_obstacles(
            state[:2], current_obstacles, scenario.max_obstacles
        )

        observation = build_observation(
            step,
            state,
            goal,
            waypoint_index,
            current_obstacles,
            active_indices,
            active_obstacles,
            scenario,
        )

        start_time = time.perf_counter()
        action = agent.act(observation)
        action_time = time.perf_counter() - start_time

        requested_control = validate_control(action)
        control = apply_control_limits(
            requested_control, previous_control, PROBLEM_CONFIG.dt
        )
        state = step_unicycle(state, control, PROBLEM_CONFIG.dt)
        previous_control = control

        clearances = [
            rectangle_clearance(state[:2], obstacle, PROBLEM_CONFIG.robot_radius)
            for obstacle in current_obstacles
        ]
        min_clearance = min(clearances) if clearances else float("inf")

        trajectory.append(state.tolist())
        controls.append(control.tolist())
        requested_controls.append(requested_control.tolist())
        action_times.append(action_time)
        min_clearances.append(min_clearance)
        obstacle_history.append([asdict(obstacle) for obstacle in current_obstacles])
        active_obstacle_history.append(active_indices)

        while waypoint_index < len(WAYPOINTS):
            active_goal = np.array(WAYPOINTS[waypoint_index], dtype=float)
            if (
                np.linalg.norm(state[:2] - active_goal[:2])
                >= PROBLEM_CONFIG.goal_threshold
            ):
                break
            waypoint_index += 1

    result: dict[str, Any] = {
        "scenario": asdict(scenario),
        "start_pose": list(START_POSE),
        "waypoints": [list(waypoint) for waypoint in WAYPOINTS],
        "trajectory": trajectory,
        "controls": controls,
        "requested_controls": requested_controls,
        "action_times": action_times,
        "min_clearances": min_clearances,
        "obstacles_history": obstacle_history,
        "active_obstacle_history": active_obstacle_history,
        "waypoints_reached": waypoint_index,
        "steps": len(controls),
        "completed": waypoint_index == len(WAYPOINTS),
    }
    result["score"] = score_result(result)
    result["metrics"] = summarize_result(result)
    return result


def summarize_result(result: dict[str, Any]) -> dict[str, float | int | bool]:
    action_times = np.array(result["action_times"], dtype=float)
    clearances = np.array(result["min_clearances"], dtype=float)
    controls = np.array(result["controls"], dtype=float)
    trajectory = np.array(result["trajectory"], dtype=float)

    path_length = 0.0
    if len(trajectory) > 1:
        path_length = float(
            np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1))
        )

    average_action_time = float(np.mean(action_times)) if len(action_times) else 0.0
    average_frequency = 1.0 / average_action_time if average_action_time > 0.0 else 0.0
    minimum_clearance = float(np.min(clearances)) if len(clearances) else float("inf")
    collision_steps = int(np.count_nonzero(clearances < 0.0)) if len(clearances) else 0
    average_speed = float(np.mean(controls[:, 0])) if len(controls) else 0.0

    return {
        "waypoints_reached": int(result["waypoints_reached"]),
        "completed": bool(result["completed"]),
        "steps": int(result["steps"]),
        "path_length": round(path_length, 3),
        "minimum_clearance": round(minimum_clearance, 3),
        "collision_steps": collision_steps,
        "average_action_time_s": round(average_action_time, 5),
        "average_frequency_hz": round(average_frequency, 2),
        "average_speed": round(average_speed, 3),
    }

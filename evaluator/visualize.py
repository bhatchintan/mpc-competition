from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D


def draw_rectangle(
    axis: plt.Axes,
    obstacle: dict[str, float],
    facecolor: str,
    edgecolor: str,
    alpha: float,
) -> None:
    rectangle = patches.Rectangle(
        (-obstacle["width"] / 2.0, -obstacle["height"] / 2.0),
        obstacle["width"],
        obstacle["height"],
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=1.5,
    )
    transform = (
        Affine2D().rotate(obstacle["angle"]).translate(obstacle["cx"], obstacle["cy"])
        + axis.transData
    )
    rectangle.set_transform(transform)
    axis.add_patch(rectangle)


def draw_robot(axis: plt.Axes, state: np.ndarray, radius: float) -> None:
    circle = patches.Circle(
        (state[0], state[1]),
        radius,
        facecolor="dodgerblue",
        edgecolor="navy",
        linewidth=2,
        alpha=0.9,
    )
    axis.add_patch(circle)
    axis.arrow(
        state[0],
        state[1],
        radius * math.cos(state[2]),
        radius * math.sin(state[2]),
        head_width=radius * 0.3,
        head_length=radius * 0.45,
        facecolor="navy",
        edgecolor="navy",
    )


def render_replay(
    result: dict[str, Any], output_path: Path, max_frames: int = 120
) -> None:
    trajectory = np.array(result["trajectory"], dtype=float)
    obstacles_history = result["obstacles_history"]
    active_history = result["active_obstacle_history"]
    waypoints = np.array(result["waypoints"], dtype=float)

    frame_count = max(len(trajectory) - 1, 1)
    if frame_count > max_frames:
        frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(frame_count)

    figure, axis = plt.subplots(figsize=(6, 6))

    def animate(frame_index: int) -> None:
        axis.clear()
        step = int(frame_indices[frame_index])
        obstacles = obstacles_history[min(step, len(obstacles_history) - 1)]
        active = set(active_history[min(step, len(active_history) - 1)])

        for index, obstacle in enumerate(obstacles):
            draw_rectangle(
                axis,
                obstacle,
                "salmon" if index in active else "lightgray",
                "darkred" if index in active else "gray",
                0.75 if index in active else 0.35,
            )

        axis.plot(waypoints[:, 0], waypoints[:, 1], "rx", markersize=9)
        axis.plot(
            trajectory[: step + 1, 0], trajectory[: step + 1, 1], "b-", linewidth=2
        )
        draw_robot(axis, trajectory[step], 0.4)
        axis.set_xlim(-1, 15)
        axis.set_ylim(-1, 15)
        axis.set_aspect("equal")
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")

    animation = FuncAnimation(
        figure, animate, frames=len(frame_indices), interval=60, blit=False
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer="pillow", fps=15, dpi=80)
    plt.close(figure)


def plot_replay(result: dict[str, Any], output_path: Path) -> None:
    trajectory = np.array(result["trajectory"], dtype=float)
    controls = np.array(result["controls"], dtype=float)
    clearances = np.array(result["min_clearances"], dtype=float)
    waypoints = np.array(result["waypoints"], dtype=float)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2)
    axes[0].plot(waypoints[:, 0], waypoints[:, 1], "rx", markersize=9)
    axes[0].set_xlim(-1, 15)
    axes[0].set_ylim(-1, 15)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("trajectory")

    if len(controls):
        times = np.arange(len(controls)) * 0.1
        axes[1].plot(times, controls[:, 0], label="v")
        axes[1].plot(times, controls[:, 1], label="omega")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title("controls")

    if len(clearances):
        axes[2].plot(clearances, color="black")
        axes[2].axhline(0.0, color="red", linestyle="--")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("clearance")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=100)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an MPC replay JSON file")
    parser.add_argument("replay", type=Path)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    with args.replay.open("r", encoding="utf-8") as input_file:
        result: dict[str, Any] = json.load(input_file)

    if args.gif is not None:
        render_replay(result, args.gif)
    if args.plot is not None:
        plot_replay(result, args.plot)


if __name__ == "__main__":
    main()

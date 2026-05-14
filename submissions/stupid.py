from __future__ import annotations

import math
from typing import Any


class Agent:
    def act(self, observation: dict[str, Any]) -> list[float]:
        state = observation["state"]
        goal = observation["goal"]

        dx = float(goal[0]) - float(state[0])
        dy = float(goal[1]) - float(state[1])
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(target_heading - float(state[2])),
            math.cos(target_heading - float(state[2])),
        )

        v = min(2.0, 0.8 * distance)
        if abs(heading_error) > 0.8:
            v *= 0.35

        omega = max(-1.0, min(1.0, 2.0 * heading_error))
        return [v, omega]

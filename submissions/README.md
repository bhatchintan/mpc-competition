# Submissions

Add one Python file in this directory. The file name is the leaderboard name.

Each file must define an `Agent` class:

```python
class Agent:
    def act(self, observation):
        return [0.0, 0.0]
```

`observation` is a dictionary with:

- `state`: `[x, y, theta]`
- `goal`: active waypoint `[x, y, theta]`
- `waypoints`: all waypoints
- `obstacles`: current rectangular obstacles
- `active_obstacle_circles`: nearest obstacle circles for the selected geometry
- `control_limits`: velocity and turn-rate limits
- `dt`: simulator time step

The evaluator clips controls to the platform limits. Invalid return values give the submission a large negative score.

`baseline.py` is the readable starter. `gpt.py` is the intentionally hard-to-read LLM competitor. `stupid.py` is the simple waypoint tracker that collides.

For LLM use rules, see the `LLMs` section at the end of `README.md`.

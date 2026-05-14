# Unicycle MPC obstacle course

## Task

Build a controller for a unicycle robot with state:

```text
x = [x, y, theta]
```

and control input:

```text
u = [v, omega]
```

The robot must move through four waypoints while avoiding rectangular obstacles. Some scenes are static. Some scenes include moving obstacles with a constant-velocity motion model.

Submissions do not need to expose an optimizer. The only required API is:

```python
class Agent:
    def act(self, observation):
        return [0.0, 0.0]
```

`act` receives the current observation and returns one control input for the next simulator step.

## Dynamics

The evaluator uses Euler integration:

```text
x_next     = x + v cos(theta) dt
y_next     = y + v sin(theta) dt
theta_next = theta + omega dt
```

The simulator time step is `0.1 s`.

Each scene runs for at most `500` simulator steps.

## Controls

The evaluator clips controls to:

```text
0.0 <= v <= 3.0
-1.0 <= omega <= 1.0
```

The evaluator also applies rate limits:

```text
|delta v| <= 1.5 dt
|delta omega| <= 1.5 dt
```

## Observation

Each call to `Agent.act(observation)` receives a dictionary with:

- `state`: current `[x, y, theta]`
- `goal`: active waypoint `[x, y, theta]`
- `waypoints`: all waypoints
- `obstacles`: current rectangular obstacles
- `active_obstacle_indices`: nearest obstacles used for the current case
- `active_obstacle_circles`: circle approximation for the active obstacles
- `geometry`: `multi_circle`
- `scene`: `static` or `dynamic`
- `robot_radius`: robot radius
- `control_limits`: velocity and turn-rate limits
- `dt`: simulator time step
- `time`: current simulator time

Rectangular obstacles are scored with rectangle geometry. Multi-circle approximations are given to the agent because most MPC formulations use circles for collision costs.

## Cases

The public evaluator runs two cases:

- static scene with multi-circle obstacle models
- dynamic scene with multi-circle obstacle models

The dynamic scene predicts obstacle motion with:

```text
cx(t) = cx(0) + v cos(angle) t
cy(t) = cy(0) + v sin(angle) t
```

## Scoring

The score rewards:

- waypoint progress
- finishing all waypoints
- ending near the active waypoint

The score penalizes:

- collision depth
- collision steps
- control effort
- control jumps
- controller calls slower than `0.2 s`

Collision penalties can push a score below zero.

The final leaderboard score is the mean score across the two cases.

## Starter agents

`submissions/stupid.py` is a simple waypoint tracker. It reaches goals by driving through obstacles, so it receives negative collision scores.

`submissions/baseline.py` is the readable starter. It uses a short rollout search with clear variable names and obstacle costs.

`submissions/gpt.py` is the LLM competitor. Its code is intentionally hard to read.

For LLM use rules, see the `LLMs` section at the end of `README.md`.

## Visualization

The repo includes a demo GIF at:

```text
assets/demo/unicycle-mpc.gif
```

Generated replay JSON files can be rendered with:

```bash
python -m evaluator.visualize outputs/replays/gpt_dynamic_multi.json --gif outputs/gpt.gif --plot outputs/gpt.png
```

The evaluator writes one replay per public case. With the current multi-circle-only setup, each submission gets:

- `outputs/replays/<name>_static_multi.json`
- `outputs/replays/<name>_dynamic_multi.json`

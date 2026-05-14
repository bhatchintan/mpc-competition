from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from evaluator.problem import run_simulation, scenarios


def load_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Submission does not exist: {path}")
    if path.suffix != ".py":
        raise ValueError("Submission must be a Python file")

    module_name = f"submission_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load submission: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_agent_class(module: ModuleType) -> type[Any]:
    if not hasattr(module, "Agent"):
        raise AttributeError("Submission must define an Agent class")

    agent_class = module.Agent
    if not callable(agent_class):
        raise TypeError("Agent must be callable")
    return agent_class


def evaluate_submission(path: Path) -> dict[str, Any]:
    module = load_module(path)
    agent_class = get_agent_class(module)

    scenario_results = []
    for scenario in scenarios():
        agent = agent_class()
        if not hasattr(agent, "act") or not callable(agent.act):
            raise TypeError("Agent instance must provide an act method")
        scenario_results.append(run_simulation(agent, scenario))

    score = round(
        sum(float(result["score"]) for result in scenario_results)
        / max(len(scenario_results), 1),
        3,
    )

    return {
        "name": path.stem,
        "score": score,
        "scenarios": {
            result["scenario"]["name"]: {
                "score": result["score"],
                "metrics": result["metrics"],
            }
            for result in scenario_results
        },
        "replays": {result["scenario"]["name"]: result for result in scenario_results},
    }

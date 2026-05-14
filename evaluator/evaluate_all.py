from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
OUTPUTS_DIR = REPO_ROOT / "outputs"
LEADERBOARD_PATH = REPO_ROOT / "leaderboard.json"
WORKER_TIMEOUT_S = 120


def submission_paths() -> list[Path]:
    if not SUBMISSIONS_DIR.exists():
        return []
    return sorted(
        path for path in SUBMISSIONS_DIR.glob("*.py") if not path.name.startswith("_")
    )


def run_worker(path: Path) -> dict[str, Any]:
    output_path = OUTPUTS_DIR / "raw" / f"{path.stem}.json"
    command = [
        sys.executable,
        "-m",
        "evaluator.worker",
        str(path),
        "--output",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": path.stem,
            "score": -10000.0,
            "status": "timeout",
            "error": f"Submission exceeded {WORKER_TIMEOUT_S} seconds",
            "scenarios": {},
        }

    if not output_path.exists():
        return {
            "name": path.stem,
            "score": -10000.0,
            "status": "error",
            "error": completed.stderr
            or completed.stdout
            or "Worker produced no output",
            "scenarios": {},
        }

    with output_path.open("r", encoding="utf-8") as input_file:
        result: dict[str, Any] = json.load(input_file)

    for scenario_name, replay in result.get("replays", {}).items():
        replay_path = OUTPUTS_DIR / "replays" / f"{path.stem}_{scenario_name}.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        with replay_path.open("w", encoding="utf-8") as output_file:
            json.dump(replay, output_file, indent=2)
            output_file.write("\n")

    result.pop("replays", None)
    if completed.returncode != 0 and result.get("status") == "ok":
        result["status"] = "error"
        result["error"] = completed.stderr or "Worker exited with a failure code"
    return result


def merge_leaderboard(results: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for result in results:
        name = str(result["name"])
        entries.append(
            {
                "name": name,
                "score": float(result["score"]),
                "status": result.get("status", "ok"),
                "scenarios": result.get("scenarios", {}),
                "error": result.get("error"),
            }
        )

    sorted_entries = sorted(
        entries,
        key=lambda entry: (-float(entry["score"]), str(entry["name"])),
    )
    return {
        "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "entries": sorted_entries,
    }


def write_leaderboard(data: dict[str, Any]) -> None:
    with LEADERBOARD_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


def main() -> None:
    paths = submission_paths()
    results = [run_worker(path) for path in paths]
    leaderboard = merge_leaderboard(results)
    write_leaderboard(leaderboard)

    print(json.dumps(leaderboard, indent=2))

    failed = [result for result in results if result.get("status") != "ok"]
    if failed:
        names = ", ".join(str(result["name"]) for result in failed)
        raise SystemExit(f"Some submissions failed: {names}")


if __name__ == "__main__":
    main()

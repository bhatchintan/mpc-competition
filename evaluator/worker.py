from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from evaluator.evaluate import evaluate_submission


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one MPC submission")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = evaluate_submission(args.submission)
        result["status"] = "ok"
    except Exception as error:
        result = {
            "name": args.submission.stem,
            "score": -10000.0,
            "status": "error",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "scenarios": {},
            "replays": {},
        }
    write_json(args.output, result)


if __name__ == "__main__":
    main()

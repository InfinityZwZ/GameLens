"""Evaluate a saved Agent run using a deterministic case definition."""

from __future__ import annotations

import argparse
from pathlib import Path

from gamelens.evaluation.contracts import evaluate_run, load_evaluation_case, load_run_state


def main() -> None:
    parser = argparse.ArgumentParser(description="评测Agent运行记录")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_run(
        load_run_state(args.run),
        load_evaluation_case(args.case),
    )
    print(result.model_dump_json(indent=2))
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()

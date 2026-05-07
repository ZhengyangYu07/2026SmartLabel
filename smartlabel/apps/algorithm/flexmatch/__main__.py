"""Allow running the flexmatch package directly with python -m."""

from __future__ import annotations

import argparse

from .train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()
    run_training(args.config)
    print("结束")


if __name__ == "__main__":
    main()
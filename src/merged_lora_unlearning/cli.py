from __future__ import annotations

import argparse
import json
from typing import Callable

from merged_lora_unlearning.artifacts import RunArtifacts, read_json
from merged_lora_unlearning.config import Config, load_config
from merged_lora_unlearning.unlearning.objectives import METHOD_SPECS


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", required=True, help="Experiment YAML file")


def _with_config(args, action: Callable[[Config], object]) -> None:
    result = action(load_config(args.config))
    if result is not None:
        print(json.dumps(result, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlu", description="Merged LoRA unlearning experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "filter", "acquire", "oracle", "report", "run", "status"):
        command = subparsers.add_parser(name)
        _config_argument(command)
    unlearn_parser = subparsers.add_parser("unlearn")
    unlearn_parser.add_argument("method", choices=tuple(METHOD_SPECS))
    _config_argument(unlearn_parser)
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model", required=True, help="base, acquisition_adapter, target, oracle, or method")
    _config_argument(eval_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "data":
        from merged_lora_unlearning.data.pipeline import run_data_pipeline

        _with_config(args, run_data_pipeline)
    elif args.command == "filter":
        from merged_lora_unlearning.data.filtering import filter_base_knowledge

        _with_config(args, filter_base_knowledge)
    elif args.command in {"acquire", "oracle"}:
        from merged_lora_unlearning.training.acquisition import train_lora
        from merged_lora_unlearning.training.merging import merge_lora

        role = "acquisition" if args.command == "acquire" else "oracle"
        train_lora(config, role)
        print(merge_lora(config, role))
    elif args.command == "unlearn":
        from merged_lora_unlearning.unlearning.trainer import unlearn

        print(unlearn(config, args.method))
    elif args.command == "eval":
        from merged_lora_unlearning.evaluation.runner import evaluate_model

        _with_config(args, lambda cfg: evaluate_model(cfg, args.model))
    elif args.command == "report":
        from merged_lora_unlearning.reporting.report import generate_report

        print(generate_report(config))
    elif args.command == "run":
        from merged_lora_unlearning.workflow import run_full_experiment

        run_full_experiment(config)
    elif args.command == "status":
        artifacts = RunArtifacts(config)
        if not artifacts.status_path.exists():
            print("No run exists.")
        else:
            print(json.dumps(read_json(artifacts.status_path), indent=2))


if __name__ == "__main__":
    main()

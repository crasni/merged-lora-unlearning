from __future__ import annotations

import argparse
import json
from typing import Callable

from merged_lora_unlearning.artifacts import RunArtifacts, read_json
from merged_lora_unlearning.config import Config, load_config
from merged_lora_unlearning.unlearning.objectives import METHOD_SPECS


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        metavar="PATH",
        help="experiment YAML that defines the run, model, data, and settings",
    )


def _with_config(args, action: Callable[[Config], object]) -> None:
    result = action(load_config(args.config))
    if result is not None:
        print(json.dumps(result, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlu",
        description=(
            "Run reproducible merged-LoRA factual unlearning experiments and evaluate them "
            "with MUSE metrics."
        ),
        epilog=(
            "Typical workflow:\n"
            "  mlu run -c configs/experiments/1_5b.yaml\n"
            "  mlu run -c configs/experiments/1_5b_unlearning.yaml\n\n"
            "Inspect a run:\n"
            "  mlu status -c configs/experiments/1_5b_unlearning.yaml\n"
            "  mlu eval --model npo -c configs/experiments/1_5b_unlearning.yaml\n\n"
            "Reuse existing unlearning checkpoints:\n"
            "  mlu finalize all -c configs/experiments/1_5b_paper.yaml\n\n"
            "Use 'mlu COMMAND -h' for command-specific help."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="COMMAND",
    )
    commands = {
        "run": "run the complete experiment workflow defined by a config",
        "status": "show completed, running, and failed stages for a run",
        "data": "generate deterministic facts and create experiment splits",
        "filter": "filter facts the base model already knows",
        "acquire": "train and merge the factual acquisition LoRA",
        "oracle": "train and merge the retain-only oracle baseline",
        "unlearn": "train one unlearning method and select its checkpoint",
        "finalize": "finalize existing unlearning checkpoints without retraining",
        "eval": "evaluate a finalized model role with MUSE metrics",
        "report": "build the comparison report from saved evaluations",
    }
    for name in ("run", "status", "data", "filter", "acquire", "oracle", "report"):
        command = subparsers.add_parser(name, help=commands[name], description=commands[name])
        _config_argument(command)
    unlearn_parser = subparsers.add_parser(
        "unlearn",
        help=commands["unlearn"],
        description=(
            "Train one unlearning method from the configured target model, evaluate the saved "
            "validation checkpoints, and finalize the best eligible checkpoint."
        ),
    )
    unlearn_parser.add_argument(
        "method",
        choices=tuple(METHOD_SPECS),
        help="forget objective and optional retain regularizer to train",
    )
    _config_argument(unlearn_parser)
    finalize_parser = subparsers.add_parser(
        "finalize",
        help=commands["finalize"],
        description=(
            "Reuse already-trained checkpoint-* directories for a method, apply the configured "
            "checkpoint_selection strategy, and rebuild outputs/runs/<run>/models/<method>."
        ),
    )
    finalize_parser.add_argument(
        "method",
        choices=(*tuple(METHOD_SPECS), "all"),
        help="unlearning method to finalize, or all configured methods",
    )
    _config_argument(finalize_parser)
    eval_parser = subparsers.add_parser(
        "eval",
        help=commands["eval"],
        description=(
            "Evaluate a finalized model role from the configured run. Method names resolve to "
            "outputs/runs/<run>/models/<method>."
        ),
    )
    eval_parser.add_argument(
        "--model",
        required=True,
        metavar="ROLE",
        help="base, acquisition_adapter, target, oracle, or an unlearning method name",
    )
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
    elif args.command == "finalize":
        from merged_lora_unlearning.unlearning.trainer import finalize_unlearned_checkpoint

        methods = config.unlearning.methods if args.method == "all" else [args.method]
        for method in methods:
            print(finalize_unlearned_checkpoint(config, method))
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

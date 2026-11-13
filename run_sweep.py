"""Create and run a W&B sweep for a selected model and dataset."""

import argparse
from pathlib import Path

import yaml

from run_compare import run_one


ROOT = Path(__file__).resolve().parent
SWEEP_CONFIGS = {
    "deepconn": ROOT / "sweep.yaml",
    "ncf": ROOT / "sweep_ncf.yaml",
}


def main():
    parser = argparse.ArgumentParser(
        description="Start a W&B hyperparameter sweep for a model and dataset."
    )
    parser.add_argument("--model", choices=tuple(SWEEP_CONFIGS), required=True)
    parser.add_argument("--dataset", choices=("amazon", "yelp"), required=True)
    parser.add_argument(
        "--sweep_id",
        help="Reuse an existing sweep ID instead of creating a new sweep.",
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--project")
    parser.add_argument("--entity")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be a positive integer")

    import wandb

    if args.sweep_id:
        sweep_id = args.sweep_id
        project = args.project
    else:
        sweep_path = SWEEP_CONFIGS[args.model]
        with sweep_path.open("r", encoding="utf-8") as file:
            sweep_config = yaml.safe_load(file)
        sweep_config.pop("program", None)
        sweep_config.pop("command", None)
        sweep_config["name"] = f"{args.model}-{args.dataset}-tuning"
        sweep_config["parameters"]["dataset"] = {"value": args.dataset}
        project = args.project or sweep_config["project"]
        sweep_id = wandb.sweep(
            sweep_config,
            project=project,
            entity=args.entity,
        )
        print(f"Created sweep: {sweep_id}")

    wandb.agent(
        sweep_id,
        function=lambda: run_one(
            str(ROOT / "configs" / f"{args.model}.yaml"),
            sweep=True,
        ),
        project=project,
        entity=args.entity,
        count=args.count,
    )


if __name__ == "__main__":
    main()

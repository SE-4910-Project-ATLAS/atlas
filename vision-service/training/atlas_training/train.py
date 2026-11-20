"""Fine-tune a COCO-pretrained YOLO model on the prepared dataset (transfer learning).

    python -m atlas_training.train configs/ambulance_v1.yaml

Uses the train and val splits only; the best epoch is chosen on val, never on test.
The run folder runs/train/<experiment>/ gets a copy of the config and of the dataset
manifest, so that every weight file can be traced to its data and settings.
"""

from __future__ import annotations

import argparse
import shutil

from . import config as cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--epochs", type=int, help="override the config, e.g. 1 for a smoke test")
    parser.add_argument("--name", help="run name (default: the experiment name)")
    args = parser.parse_args()

    from ultralytics import YOLO

    exp = cfg.load(args.config)
    if not exp.data_yaml.exists():
        raise SystemExit("Prepare the dataset first: python -m atlas_training.prepare <config>")
    settings = dict(exp.train)
    base = settings.pop("base_model")
    if args.epochs:
        settings["epochs"] = args.epochs

    cfg.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(cfg.WEIGHTS_DIR / base))   # downloaded there on first use
    result = model.train(
        data=str(exp.data_yaml),
        seed=exp.seed,
        deterministic=True,
        project=str(cfg.RUNS_DIR / "train"),
        name=args.name or exp.name,
        exist_ok=False,
        **settings,
    )
    run_dir = result.save_dir if result is not None else model.trainer.save_dir
    shutil.copy2(args.config, run_dir / "experiment.yaml")
    shutil.copy2(exp.dataset_dir / "manifest.json", run_dir / "dataset_manifest.json")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

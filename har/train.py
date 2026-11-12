import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import torch
from torch import nn

from har.engine import device_for, load_split_metadata, make_loader, run_epoch, seed_everything
from har.models import build_model


def train(config_path, root, splits, output, smoke=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config["primary_metric"] != "macro_f1":
        raise ValueError("The preselected primary metric must be macro_f1")
    seed_everything(config["seed"])
    # Required for deterministic CUDA matrix multiplication.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    classes, summary = load_split_metadata(splits, root)
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(Path("har").glob("*.py")))).hexdigest()
    runtime = {"torch": str(torch.__version__), "torchvision": importlib.metadata.version("torchvision")}
    if smoke:
        config.update(epochs=1, batch_size=2, pretrained=False, device="cpu")
    device = device_for(config["device"])
    model = build_model(config["variant"], len(classes), config["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader("train", root, splits, classes, config, True)
    val_loader = make_loader("val", root, splits, classes, config)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("HAR-smoke" if smoke else "HAR-coursework")
    with mlflow.start_run(run_name=config["variant"]) as run:
        directory = Path(output) / run.info.run_id
        directory.mkdir(parents=True, exist_ok=False)
        mlflow.log_params({**config, "optimizer": "AdamW", "loss": "CrossEntropyLoss",
                           "actual_device": str(device), "smoke": smoke,
                           "trainable_parameters": sum(p.numel() for p in model.parameters()),
                           "torch_version": torch.__version__})
        mlflow.set_tags({"mlflow.note.content": config["description"],
                         "dataset": summary["source"], "split_seed": str(summary["seed"]),
                         "train_manifest_sha256": summary["manifest_sha256"]["train"]})
        mlflow.set_tag("source_sha256", source_hash)
        for name in ("summary.json", "classes.json", "train.csv", "val.csv", "test.csv",
                     "excluded.json", "duplicates.json"):
            mlflow.log_artifact(str(Path(splits) / name), artifact_path="data")
        mlflow.log_artifact(str(config_path), artifact_path="configuration")
        # Retain the exact implementation alongside every run.
        for source in sorted(Path("har").glob("*.py")):
            mlflow.log_artifact(str(source), artifact_path="source/har")
        mlflow.log_artifact("requirements.txt", artifact_path="source")
        history, best = [], -1.0
        for epoch in range(config["epochs"]):
            train_metrics, _ = run_epoch(model, train_loader, device, criterion, optimizer,
                                         max_batches=2 if smoke else None)
            val_metrics, _ = run_epoch(model, val_loader, device, criterion,
                                       max_batches=2 if smoke else None)
            metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append({"epoch": epoch + 1, **metrics})
            mlflow.log_metrics(metrics, step=epoch + 1)
            if val_metrics["macro_f1"] > best:
                best = val_metrics["macro_f1"]
                cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                torch.save({"state_dict": cpu_state, "classes": classes,
                            "config": config, "run_id": run.info.run_id,
                            "best_epoch": epoch + 1, "best_val_macro_f1": best,
                            "best_val_accuracy": val_metrics["accuracy"],
                            "best_val_loss": val_metrics["loss"],
                            "manifest_sha256": summary["manifest_sha256"],
                            "runtime": runtime, "actual_device": str(device),
                            "source_sha256": source_hash,
                            "smoke": smoke}, directory / "best.pt")
            print(json.dumps(history[-1]), flush=True)
        (directory / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (directory / "run.json").write_text(json.dumps({"run_id": run.info.run_id,
             "variant": config["variant"], "best_val_macro_f1": best,
             "checkpoint": "best.pt", "smoke": smoke}, indent=2), encoding="utf-8")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric in zip(axes, ("loss", "macro_f1", "accuracy")):
            for split in ("train", "val"):
                ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
            ax.set(title=metric, xlabel="Epoch")
            ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "learning_curves.png", dpi=150)
        plt.close(fig)
        mlflow.log_metric("best_val_macro_f1", best)
        mlflow.log_artifacts(str(directory), artifact_path="training")
        print(f"Checkpoint: {directory / 'best.pt'}", flush=True)
        return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--data-root", default="Human Action Recognition")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--output", default="outputs/runs")
    parser.add_argument("--smoke", action="store_true", help="Pipeline check; not a scientific result")
    args = parser.parse_args()
    train(args.config, args.data_root, args.splits, args.output, args.smoke)


if __name__ == "__main__":
    main()

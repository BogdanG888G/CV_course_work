import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn

from har.engine import device_for, load_split_metadata, make_loader, run_epoch, seed_everything
from har.models import build_model


def evaluate(final, root, splits):
    final = Path(final)
    if (final / "evaluation_started.json").exists():
        raise FileExistsError("Test evaluation already started; use existing results and do not tune on test")
    selection = json.loads((final / "selection.json").read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256((final / "model.pt").read_bytes()).hexdigest()
    if actual_sha != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint has changed")
    checkpoint = torch.load(final / "model.pt", map_location="cpu", weights_only=True)
    classes, summary = load_split_metadata(splits, root)
    if checkpoint["smoke"] or checkpoint["manifest_sha256"] != summary["manifest_sha256"] or checkpoint["classes"] != classes:
        raise ValueError("Smoke checkpoint or mismatched split/classes")
    config = checkpoint["config"]
    seed_everything(config["seed"])
    device = device_for(config["device"])
    model = build_model(config["variant"], len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_id=checkpoint["run_id"]):
        (final / "evaluation_started.json").write_text(json.dumps({"run_id": checkpoint["run_id"]}), encoding="utf-8")
        metrics, results = run_epoch(model, make_loader("test", root, splits, classes, config), device, nn.CrossEntropyLoss())
        (final / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report = classification_report(results["actual"], results["predicted"], labels=list(range(len(classes))),
                                        target_names=classes, output_dict=True, zero_division=0)
        (final / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        pd.DataFrame({"path": results["paths"],
                      "actual": [classes[i] for i in results["actual"]],
                      "predicted": [classes[i] for i in results["predicted"]],
                      "confidence": np.max(results["probabilities"], axis=1)}).to_csv(final / "predictions.csv", index=False)
        matrix = confusion_matrix(results["actual"], results["predicted"], labels=list(range(len(classes))))
        fig, ax = plt.subplots(figsize=(13, 11))
        ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax, xticks_rotation=90, colorbar=False)
        fig.tight_layout()
        fig.savefig(final / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        correct = [i for i, (a, p) in enumerate(zip(results["actual"], results["predicted"])) if a == p]
        errors = [i for i, (a, p) in enumerate(zip(results["actual"], results["predicted"])) if a != p]
        for name, indices in (("correct_examples", correct), ("error_examples", errors)):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for ax in axes.flat:
                ax.axis("off")
            for ax, i in zip(axes.flat, indices[:8]):
                with Image.open(Path(root) / results["paths"][i]) as image:
                    ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
                ax.set_title(f"True: {classes[results['actual'][i]]}\nPred: {classes[results['predicted'][i]]}", fontsize=9)
            fig.tight_layout()
            fig.savefig(final / f"{name}.png", dpi=150)
            plt.close(fig)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifacts(str(final), artifact_path="final_evaluation")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", default="outputs/final")
    parser.add_argument("--data-root", default="Human Action Recognition")
    parser.add_argument("--splits", default="data/splits")
    args = parser.parse_args()
    evaluate(args.final, args.data_root, args.splits)

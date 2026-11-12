"""Select on validation only; freeze a deployable checkpoint before test use."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import torch


def select(run_dirs, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Selection exists: {output}; do not reselect after inspecting test results")
    records, checkpoints = [], []
    for directory in run_dirs:
        directory = Path(directory)
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["smoke"]:
            raise ValueError("Smoke runs cannot be selected as coursework results")
        checkpoints.append(checkpoint)
        records.append({"variant": checkpoint["config"]["variant"], "run_id": checkpoint["run_id"],
                        "best_val_macro_f1": checkpoint["best_val_macro_f1"],
                        "val_accuracy": checkpoint["best_val_accuracy"],
                        "val_loss": checkpoint["best_val_loss"],
                        "best_epoch": checkpoint["best_epoch"], "directory": str(directory)})
    reference = checkpoints[0]
    for checkpoint in checkpoints[1:]:
        if checkpoint["manifest_sha256"] != reference["manifest_sha256"] or checkpoint["classes"] != reference["classes"]:
            raise ValueError("Experiments use different datasets or splits")
        for field in ("actual_device", "runtime", "source_sha256"):
            if checkpoint[field] != reference[field]:
                raise ValueError(f"Experiments differ in {field}; use the same environment and code")
        for key in ("seed", "epochs", "batch_size", "image_size", "learning_rate", "weight_decay", "pretrained", "device", "workers"):
            if checkpoint["config"][key] != reference["config"][key]:
                raise ValueError(f"Experiments differ in {key}; run a controlled comparison")
    baselines = [r for r in records if r["variant"] == "baseline"]
    modifications = [r for r in records if r["variant"] != "baseline"]
    if len(baselines) != 1 or not modifications:
        raise ValueError("Provide exactly one baseline and at least one modified run")
    if len({r["variant"] for r in records}) != len(records):
        raise ValueError("Provide one run per variant; retain other attempts in MLflow and report")
    improved = max(r["best_val_macro_f1"] for r in modifications) > baselines[0]["best_val_macro_f1"]
    if not improved and len(modifications) < 3:
        raise ValueError("No validation improvement: requirements demand three distinct trained modifications")
    winner = max(records, key=lambda r: r["best_val_macro_f1"])
    output.mkdir(parents=True)
    shutil.copyfile(Path(winner["directory"]) / "best.pt", output / "model.pt")
    selection = {"selected": winner, "primary_metric": "macro_f1", "modified_improved": improved,
                 "experiments": records,
                 "checkpoint_sha256": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest()}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(output / "comparison.csv", index=False)
    print(json.dumps(selection, indent=2))
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/final")
    args = parser.parse_args()
    select(args.runs, args.output)

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from har.data import ActionDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def device_for(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_split_metadata(directory, root=None):
    directory = Path(directory)
    classes = json.loads((directory / "classes.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        actual = hashlib.sha256((directory / f"{split}.csv").read_bytes()).hexdigest()
        if summary["manifest_sha256"][split] != actual:
            raise ValueError(f"Manifest changed: {split}")
        if root is not None:
            import csv
            from har.prepare import file_sha256
            with (directory / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if file_sha256(Path(root) / row["path"]) != row["sha256"]:
                        raise ValueError(f"Source image changed: {row['path']}")
    return classes, summary


def make_loader(split, root, directory, classes, config, training=False):
    dataset = ActionDataset(Path(directory) / f"{split}.csv", root, classes,
                            config["image_size"], training)
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training,
                      num_workers=config["workers"], generator=generator)


def run_epoch(model, loader, device, criterion, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss, count, actual, predictions, probabilities, paths = 0.0, 0, [], [], [], []
    with torch.set_grad_enabled(training):
        for batch, (images, targets, names) in enumerate(loader):
            if max_batches is not None and batch >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            count += len(targets)
            actual.extend(targets.detach().cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
            probabilities.extend(logits.detach().softmax(1).cpu().tolist())
            paths.extend(names)
    if count == 0:
        raise ValueError("Empty loader")
    metrics = {"loss": total_loss / count, "accuracy": accuracy_score(actual, predictions),
               "macro_f1": f1_score(actual, predictions,
                                    labels=list(range(len(probabilities[0]))),
                                    average="macro", zero_division=0)}
    return metrics, {"actual": actual, "predicted": predictions,
                     "probabilities": probabilities, "paths": paths}

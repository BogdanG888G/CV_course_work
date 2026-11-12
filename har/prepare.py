"""Validate images and keep exact/perceptual duplicate groups in one split."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class BKTree:
    """Hamming-distance index for 64-bit perceptual hashes."""
    def __init__(self):
        self.root = None

    def add(self, value, index):
        node = (value, index, {})
        if self.root is None:
            self.root = node
            return
        current = self.root
        while True:
            distance = (value ^ current[0]).bit_count()
            if distance in current[2]:
                current = current[2][distance]
            else:
                current[2][distance] = node
                return

    def find(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield node[1]
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)


def prepare(root, output, seed=42, radius=4):
    root, output = Path(root), Path(output)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Split already exists: {output}. Use another output directory.")
    output.mkdir(parents=True, exist_ok=True)
    with (root / "Training_set.csv").open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    if not source or not {"filename", "label"}.issubset(source[0]):
        raise ValueError("Training_set.csv must have filename,label columns")
    filenames = [row["filename"] for row in source]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Repeated filenames in label table")
    rows, excluded = [], []
    for number, row in enumerate(source, 1):
        name = row["filename"]
        if Path(name).name != name or "\\" in name:
            raise ValueError(f"Unsafe filename in source: {name}")
        path = root / "train" / name
        try:
            with Image.open(path) as original:
                original.load()
                image = ImageOps.exif_transpose(original).convert("RGB")
                pixel_hash = hashlib.sha256(
                    f"{image.size}".encode() + image.tobytes()
                ).hexdigest()
                rows.append({"path": f"train/{name}", "label": row["label"],
                             "sha256": file_sha256(path), "pixel_sha256": pixel_hash,
                             "phash": str(imagehash.phash(image)),
                             "width": image.width, "height": image.height})
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            excluded.append({"path": f"train/{name}", "reason": str(error)})
        if number % 1000 == 0:
            print(f"Validated {number}/{len(source)} images", flush=True)
    if not rows:
        raise ValueError("No readable labelled images")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    tree, exact, pixels = BKTree(), {}, {}
    for i, row in enumerate(rows):
        for field, mapping in (("sha256", exact), ("pixel_sha256", pixels)):
            if row[field] in mapping:
                union(i, mapping[row[field]])
            mapping[row[field]] = i
        value = int(row["phash"], 16)
        for match in tree.find(value, radius):
            union(i, match)
        # Index only unique pHashes: same-hash images have already been joined.
        if next(tree.find(value, 0), None) is None:
            tree.add(value, i)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    clean_groups, duplicate_groups, conflicts = [], [], []
    for indices in groups.values():
        members = [rows[i] for i in indices]
        labels = sorted({row["label"] for row in members})
        if len(members) > 1:
            duplicate_groups.append({"paths": [r["path"] for r in members], "labels": labels})
        if len(labels) > 1:
            conflicts.extend({"path": r["path"], "reason": "conflicting labels in duplicate group"}
                             for r in members)
        else:
            clean_groups.append(members)
    ids = list(range(len(clean_groups)))
    label_for = lambda selected: [clean_groups[i][0]["label"] for i in selected]
    train_ids, rest_ids = train_test_split(ids, test_size=0.30, random_state=seed,
                                         stratify=label_for(ids))
    val_ids, test_ids = train_test_split(rest_ids, test_size=0.50, random_state=seed,
                                       stratify=label_for(rest_ids))
    counts, fingerprints = {}, {}
    classes = sorted({r["label"] for g in clean_groups for r in g})
    for split, selected in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        selected_rows = sorted((r for i in selected for r in clean_groups[i]), key=lambda r: r["path"])
        path = output / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected_rows)
        counts[split] = {label: sum(r["label"] == label for r in selected_rows) for label in classes}
        fingerprints[split] = file_sha256(path)
    (output / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    (output / "duplicates.json").write_text(json.dumps(duplicate_groups, indent=2), encoding="utf-8")
    (output / "excluded.json").write_text(json.dumps(excluded + conflicts, indent=2), encoding="utf-8")
    summary = {"source": "meetnagadia/human-action-recognition-har-dataset",
               "source_labels_sha256": file_sha256(root / "Training_set.csv"),
               "seed": seed, "phash_hamming_radius": radius,
               "split_policy": "stratified duplicate groups, approximately 70/15/15",
               "source_rows": len(source), "readable_images": len(rows),
               "excluded_images": len(excluded) + len(conflicts),
               "duplicate_groups": len(duplicate_groups),
               "counts": counts, "manifest_sha256": fingerprints,
               "limitations": "pHash is heuristic; semantic near-duplicates can remain. Review duplicate groups visually."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Human Action Recognition")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-radius", type=int, default=4, choices=range(0, 9))
    args = parser.parse_args()
    prepare(args.data_root, args.output, args.seed, args.phash_radius)


if __name__ == "__main__":
    main()

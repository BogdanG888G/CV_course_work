"""Independent cross-split check using the saved hashes and pHashes."""
import argparse
import csv
import json
from pathlib import Path

from har.prepare import BKTree, file_sha256


def verify(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    all_rows, counts = [], {}
    for split in ("train", "val", "test"):
        path = directory / f"{split}.csv"
        if file_sha256(path) != summary["manifest_sha256"][split]:
            raise ValueError(f"Changed manifest: {split}")
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        counts[split] = len(rows)
        all_rows.extend((split, row) for row in rows)
    paths = [row["path"] for _, row in all_rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Repeated path within/across splits")
    seen = {field: {} for field in ("sha256", "pixel_sha256")}
    tree = BKTree()
    for index, (split, row) in enumerate(all_rows):
        for field, mapping in seen.items():
            prior = mapping.setdefault(row[field], split)
            if prior != split:
                raise ValueError(f"Cross-split {field}: {row['path']}")
        value = int(row["phash"], 16)
        for match in tree.find(value, summary["phash_hamming_radius"]):
            if all_rows[match][0] != split:
                raise ValueError(f"Cross-split pHash neighbour: {row['path']}")
        if next(tree.find(value, 0), None) is None:
            tree.add(value, index)
    result = {"counts": counts, "total": len(all_rows),
              "path_overlap": False, "file_hash_overlap": False,
              "pixel_hash_overlap": False, "phash_neighbour_overlap": False,
              "phash_radius": summary["phash_hamming_radius"],
              "note": "Checks saved hashes; does not re-read or semantically compare all images."}
    (directory / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/splits")
    verify(parser.parse_args().splits)

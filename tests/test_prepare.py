import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from har.prepare import BKTree, prepare


class PreparationTests(unittest.TestCase):
    def test_bktree_matches_bruteforce(self):
        values = [0, 3, 15, 255, 1024, 511, 1 << 63]
        tree = BKTree()
        for index, value in enumerate(values):
            tree.add(value, index)
        for query in [0, 7, 256, (1 << 63) + 3]:
            for radius in range(5):
                self.assertEqual(set(tree.find(query, radius)),
                                 {i for i, v in enumerate(values) if (query ^ v).bit_count() <= radius})

    def test_duplicate_groups_conflicts_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train").mkdir()
            rng = np.random.default_rng(123)
            rows = []
            for label in ("a", "b"):
                for i in range(30):
                    filename = f"{label}{i}.png"
                    Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(root / "train" / filename)
                    rows.append({"filename": filename, "label": label})
            duplicate = root / "train" / "duplicate.png"
            duplicate.write_bytes((root / "train" / "a0.png").read_bytes())
            rows.append({"filename": duplicate.name, "label": "a"})
            conflict = root / "train" / "conflict.png"
            conflict.write_bytes((root / "train" / "b1.png").read_bytes())
            rows.append({"filename": conflict.name, "label": "a"})
            (root / "train" / "broken.jpg").write_bytes(b"broken")
            rows.append({"filename": "broken.jpg", "label": "a"})
            with (root / "Training_set.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["filename", "label"])
                writer.writeheader()
                writer.writerows(rows)
            first = prepare(root, root / "first", radius=0)
            second = prepare(root, root / "second", radius=0)
            self.assertEqual(first, second)
            self.assertEqual(first["excluded_images"], 3)
            memberships = {}
            hashes = []
            for split in ("train", "val", "test"):
                with (root / "first" / f"{split}.csv").open() as stream:
                    records = list(csv.DictReader(stream))
                memberships.update({r["path"]: split for r in records})
                hashes.append({r["sha256"] for r in records})
            self.assertEqual(memberships["train/a0.png"], memberships["train/duplicate.png"])
            self.assertNotIn("train/b1.png", memberships)
            for i in range(3):
                for j in range(i):
                    self.assertFalse(hashes[i] & hashes[j])
            with self.assertRaises(FileExistsError):
                prepare(root, root / "first")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import torch

from har.select import select


class SelectionTests(unittest.TestCase):
    def create_run(self, root, variant, score, smoke=False, source="same"):
        directory = root / variant
        directory.mkdir()
        config = json.loads(Path(f"configs/{variant}.json").read_text())
        torch.save({"config": config, "classes": ["a", "b"], "run_id": variant,
                    "best_val_macro_f1": score, "best_val_accuracy": score,
                    "best_val_loss": 1.0, "best_epoch": 1,
                    "manifest_sha256": {"train": "a", "val": "b", "test": "c"},
                    "actual_device": "cpu", "runtime": {"torch": "test"},
                    "source_sha256": source, "smoke": smoke}, directory / "best.pt")
        return directory

    def test_no_improvement_requires_three_modifications(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = self.create_run(root, "baseline", 0.8)
            gem = self.create_run(root, "gem", 0.7)
            output = root / "final"
            with self.assertRaisesRegex(ValueError, "three distinct"):
                select([baseline, gem], output)
            self.assertFalse(output.exists())
            avgmax = self.create_run(root, "avgmax", 0.6)
            attention = self.create_run(root, "attention", 0.75)
            result = select([baseline, gem, avgmax, attention], output)
            self.assertEqual(result["selected"]["variant"], "baseline")
            self.assertFalse(result["modified_improved"])
            with self.assertRaises(FileExistsError):
                select([baseline, gem], output)

    def test_improvement_suffices_and_different_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = self.create_run(root, "baseline", 0.7)
            gem = self.create_run(root, "gem", 0.8)
            self.assertTrue(select([baseline, gem], root / "final")["modified_improved"])
            attention = self.create_run(root, "attention", 0.9, source="different")
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                select([baseline, attention], root / "another")

    def test_smoke_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            smoke = self.create_run(root, "baseline", 1.0, smoke=True)
            with self.assertRaisesRegex(ValueError, "Smoke"):
                select([smoke], root / "final")


if __name__ == "__main__":
    unittest.main()

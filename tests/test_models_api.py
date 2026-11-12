import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from har.api import create_app
from har.models import AttentionPool, GeM, build_model


class ModelAndAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_variants_forward_and_backward(self):
        for variant in ("baseline", "gem", "avgmax", "attention"):
            with self.subTest(variant=variant):
                model = build_model(variant, 15)
                logits = model(torch.rand(2, 3, 64, 64))
                self.assertEqual(tuple(logits.shape), (2, 15))
                logits.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_pooling_properties(self):
        tensor = torch.rand(2, 512, 3, 4)
        self.assertTrue(torch.allclose(AttentionPool()(tensor).flatten(1), tensor.mean((-2, -1)), atol=1e-6))
        self.assertTrue(torch.allclose(GeM()(torch.ones_like(tensor)), torch.ones(2, 512, 1, 1)))

    def test_api_contract_and_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save({"state_dict": build_model("baseline", 2).state_dict(),
                        "classes": ["a", "b"], "config": {"variant": "baseline", "image_size": 64},
                        "run_id": "synthetic-test-only", "smoke": False}, path)
            (path.parent / "selection.json").write_text(json.dumps({"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
            app = create_app(path)
            # Without lifespan initialization /health must not claim readiness.
            self.assertEqual(TestClient(app).get("/health").status_code, 503)
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                buffer = io.BytesIO()
                Image.new("RGB", (80, 80), "red").save(buffer, format="PNG")
                response = client.post("/predict", files={"file": ("image.png", buffer.getvalue(), "image/png")})
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertAlmostEqual(sum(body["probabilities"].values()), 1.0, places=5)
                self.assertIn(body["class"], ["a", "b"])
                self.assertEqual(client.post("/predict", files={"file": ("bad.jpg", b"bad")}).status_code, 422)
                self.assertEqual(client.post("/predict").status_code, 422)
                self.assertEqual(client.post("/predict", files={"file": ("large.jpg", b"x" * (10 * 1024 * 1024 + 1))}).status_code, 413)
            path.write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                with TestClient(create_app(path)):
                    pass


if __name__ == "__main__":
    unittest.main()

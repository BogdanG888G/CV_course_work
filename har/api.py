import hashlib
import io
import json
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from har.data import image_transforms
from har.models import build_model


MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path = Path(model_path or os.getenv("MODEL_PATH", "outputs/final/model.pt"))
        selection = json.loads((path.parent / "selection.json").read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint differs from selection.json")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("smoke"):
            raise RuntimeError("A smoke checkpoint cannot be served as the final model")
        classes = checkpoint["classes"]
        model = build_model(checkpoint["config"]["variant"], len(classes))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        # Readiness requires successful inference, not merely opening a file.
        size = checkpoint["config"]["image_size"]
        with torch.inference_mode():
            model(torch.zeros(1, 3, size, size))
        app.state.model = model
        app.state.classes = classes
        app.state.transform = image_transforms(size)
        app.state.run_id = checkpoint["run_id"]
        yield
        app.state.model = None

    app = FastAPI(title="Human Action Recognition", lifespan=lifespan)

    @app.get("/health")
    def health():
        if getattr(app.state, "model", None) is None:
            raise HTTPException(503, "Model is not ready")
        return {"status": "ready", "run_id": app.state.run_id}

    @app.post("/predict")
    def predict(file: UploadFile):
        if getattr(app.state, "model", None) is None:
            raise HTTPException(503, "Model is not ready")
        content = file.file.read(MAX_BYTES + 1)
        if len(content) > MAX_BYTES:
            raise HTTPException(413, "Image exceeds 10 MiB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as image:
                    if image.format not in {"JPEG", "PNG", "WEBP"}:
                        raise HTTPException(415, "Supported formats: JPEG, PNG, WEBP")
                    image.load()
                    tensor = app.state.transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(422, "Invalid image or excessive pixel count") from None
        with torch.inference_mode():
            probabilities = app.state.model(tensor).softmax(dim=1)[0].tolist()
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {"class": app.state.classes[best], "confidence": probabilities[best],
                "probabilities": dict(zip(app.state.classes, probabilities)), "run_id": app.state.run_id}

    return app


app = create_app()

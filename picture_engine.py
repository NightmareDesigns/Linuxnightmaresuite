"""Dedicated picture-generation runtime for the Nightmare Suite.

This module is intentionally separate from chat model loading.  It owns the
.ntmre compression diffusion pipeline, chooses CPU or GPU execution, and keeps a small
procedural fallback available when no diffusion checkpoint is installed.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np


class PictureEngine:
    """Owns one image pipeline and reports the backend actually in use."""

    def __init__(self, model_path: Optional[str] = None, backend: str = "auto"):
        self.backend_requested = backend.lower().strip() or "auto"
        self.backend = "cpu"
        self.model_path: Optional[Path] = None
        self.pipeline = None
        self.status = "CPU kernel (procedural fallback)"
        if model_path:
            self.load(model_path)

    @staticmethod
    def find_default_model() -> Optional[Path]:
        """Return an explicitly named image checkpoint, never a chat model."""
        configured = os.environ.get("NIGHTMARE_IMAGE_MODEL", "").strip()
        if configured:
            return Path(configured)

        model_dir = Path(__file__).parent / "models"
        candidates = sorted(
            path for path in model_dir.glob("*")
            if path.suffix.lower() in {".ntmre", ".safetensors", ".gguf"}
            and any(tag in path.name.lower() for tag in ("image", "diffusion", "sd", "svd", "pornmaster", "flux"))
        )
        return candidates[0] if candidates else None

    def load(self, model_path: str) -> None:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Picture model not found: {path}")
        if path.suffix.lower() not in {".ntmre", ".safetensors", ".gguf"}:
            raise ValueError("Picture Engine accepts an NTMRE, SafeTensors, or GGUF diffusion checkpoint.")

        from nightmare_imagegen import NightmareImageGen

        pipeline = NightmareImageGen(str(path))
        if not pipeline.loaded:
            raise RuntimeError(f"Could not load picture model: {path.name}")

        gpu = getattr(getattr(pipeline, "unet", None), "_gpu", None)
        gpu_ready = bool(gpu is not None and getattr(gpu, "available", False))
        if self.backend_requested == "gpu" and not gpu_ready:
            raise RuntimeError("GPU picture backend requested, but the CUDA image kernel is unavailable.")

        self.pipeline = pipeline
        self.model_path = path
        if gpu_ready and self.backend_requested != "cpu":
            self.backend = "gpu"
            self.status = f"GPU .ntmre compression kernel ({path.name})"
        else:
            self.backend = "cpu"
            self.status = f"CPU .ntmre compression kernel ({path.name})"

    def generate(self, prompt: str, negative_prompt: str = "", width: int = 512,
                 height: int = 512, steps: int = 20, guidance: float = 7.5,
                 seed: int = -1) -> np.ndarray:
        """Generate a CHW float image from the loaded diffusion checkpoint."""
        if self.pipeline is None:
            raise RuntimeError(
                "No diffusion picture model is installed. Set NIGHTMARE_IMAGE_MODEL "
                "to an .ntmre compression image checkpoint."
            )
        return self.pipeline.txt2img(
            prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            cfg_scale=float(guidance),
            seed=int(seed),
        )

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

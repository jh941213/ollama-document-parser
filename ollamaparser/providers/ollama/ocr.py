"""Ollama OCR provider using DeepSeek-OCR model."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ollamaparser.providers.base import OCRModel
from ollamaparser.providers.ollama.client import OllamaClient


@dataclass
class OllamaOCR(OCRModel):
    """
    Ollama-based OCR provider.

    Uses DeepSeek-OCR or similar OCR models available via Ollama.
    Model name can be configured via OCR_MODEL environment variable.

    Example:
        OCR_MODEL=deepseek-ocr:latest python script.py
        OCR_MODEL=custom-ocr:v2 python script.py
    """

    model: str = ""  # Will be loaded from env or defaults to deepseek-ocr
    timeout_sec: int = 600
    host: str = ""  # e.g. http://localhost:11434 (defaults from OLLAMA_HOST)
    transport: str = "api"  # api|cli

    def __post_init__(self) -> None:
        """Initialize model from environment variable if not set."""
        if not self.model:
            self.model = os.environ.get("OCR_MODEL", "deepseek-ocr:latest")

    def ocr(self, *, image: Image.Image, mode: str = "markdown") -> str:
        """Extract text from image using OCR model."""
        if self.transport == "api":
            prompt = (
                "<|grounding|>Convert the document to markdown."
                if mode == "markdown"
                else ("Free OCR." if mode == "free" else mode)
            )
            client = OllamaClient(host=self.host, timeout_sec=self.timeout_sec)
            res = client.generate(model=self.model, prompt=prompt, images=[image])
            return (res.get("response") or "").strip()

        # CLI transport
        with tempfile.TemporaryDirectory(prefix="ollamaparser_ocr_") as td:
            img_path = Path(td) / "input.png"
            image.convert("RGB").save(img_path)
            return self.ocr_path(image_path=str(img_path), mode=mode)

    def ocr_path(self, *, image_path: str, mode: str = "markdown") -> str:
        """Extract text from image file using Ollama CLI."""
        image_path = os.path.abspath(image_path)

        if mode == "markdown":
            prompt = "<|grounding|>Convert the document to markdown."
        elif mode == "free":
            prompt = "Free OCR."
        else:
            # allow custom prompt passthrough
            prompt = mode

        # Ollama model expects: "/path/to/image\nPROMPT"
        full = f"{image_path}\n{prompt}"

        proc = subprocess.run(
            ["ollama", "run", self.model, full],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.timeout_sec,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "ollama run failed\n"
                f"model={self.model}\n"
                f"stderr:\n{proc.stderr.strip()}\n"
            )
        return (proc.stdout or "").strip()

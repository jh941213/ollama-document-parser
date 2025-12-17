"""Ollama HTTP API client."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image


def _default_host() -> str:
    """Get default Ollama host from environment or use localhost."""
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _img_to_b64_png(img: Image.Image) -> str:
    """Convert PIL Image to base64-encoded PNG string."""
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class OllamaClient:
    """HTTP client for Ollama API."""

    host: str = ""
    timeout_sec: int = 600

    def __post_init__(self) -> None:
        """Initialize host from environment if not provided."""
        if not self.host:
            self.host = _default_host()
        self.host = self.host.rstrip("/")

    def tags(self) -> dict[str, Any]:
        """Get available models from Ollama server."""
        r = requests.get(f"{self.host}/api/tags", timeout=self.timeout_sec)
        r.raise_for_status()
        return r.json()

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        images: list[Image.Image] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate text using /api/generate endpoint."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if images:
            payload["images"] = [_img_to_b64_png(im) for im in images]

        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        return r.json()

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        """Generate chat response using /api/chat endpoint."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if think is not None:
            payload["think"] = think
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        return r.json()

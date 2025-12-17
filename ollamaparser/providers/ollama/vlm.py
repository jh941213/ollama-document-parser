"""Ollama VLM provider for vision-language model tasks."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ollamaparser.providers.base import VisionLanguageModel
from ollamaparser.providers.ollama.client import OllamaClient

logger = logging.getLogger(__name__)


def _extract_content_from_response(res: dict[str, Any]) -> str:
    """
    Extract text content from Ollama response.

    Different models return different response formats:
    - /api/chat: {"message": {"content": "..."}}
    - /api/generate: {"response": "..."}
    - qwen3-vl thinking: may have different structure
    """
    # Try multiple possible locations
    candidates = []

    # 1. /api/chat standard: message.content
    if isinstance(res.get("message"), dict):
        msg = res["message"]
        if msg.get("content"):
            candidates.append(msg["content"])
        # Some models put thinking in a separate field
        if msg.get("thinking"):
            candidates.append(msg["thinking"])

    # 2. /api/generate standard: response
    if res.get("response"):
        candidates.append(res["response"])

    # 3. Direct content field
    if res.get("content"):
        candidates.append(res["content"])

    # 4. OpenAI-style: choices[0].message.content
    if isinstance(res.get("choices"), list) and res["choices"]:
        choice = res["choices"][0]
        if isinstance(choice.get("message"), dict) and choice["message"].get("content"):
            candidates.append(choice["message"]["content"])

    # Return the longest non-empty candidate
    result = ""
    for c in candidates:
        if c and len(c) > len(result):
            result = c

    if not result:
        # Debug: log the keys we got
        logger.warning(f"[OllamaVLM] Empty response. Keys received: {list(res.keys())}")
        if "message" in res:
            logger.warning(
                f"[OllamaVLM] message keys: {list(res['message'].keys()) if isinstance(res['message'], dict) else type(res['message'])}"
            )
        # Log full response for debugging (truncated)
        import json

        try:
            full = json.dumps(res, ensure_ascii=False, indent=2)
            logger.warning(f"[OllamaVLM] Full response (first 2000 chars):\n{full[:2000]}")
        except Exception:
            pass

    return result.strip()


@dataclass
class OllamaVLM(VisionLanguageModel):
    """
    Ollama-based Vision Language Model provider.

    Model name can be configured via VLM_MODEL environment variable.

    Example model tags:
    - gemma3:27b-it-qat
    - qwen3-vl:30b (recommended)

    Usage:
        VLM_MODEL=gemma3:27b-it-qat python script.py
        VLM_MODEL=qwen3-vl:30b python script.py
    """

    model: str = ""  # Will be loaded from env or defaults to gemma3:27b
    timeout_sec: int = 600
    host: str = ""  # e.g. http://localhost:11434 (defaults from OLLAMA_HOST)
    transport: str = "api"  # api|cli
    options: dict | None = None
    think: bool = False  # Disable thinking mode by default (for qwen3-vl etc.)

    def __post_init__(self) -> None:
        """Initialize model from environment variable if not set."""
        if not self.model:
            self.model = os.environ.get("VLM_MODEL", "gemma3:27b")

    def generate(self, *, prompt: str, images: list[Image.Image] | None = None) -> str:
        """Generate text response using VLM model."""
        images = images or []
        if len(images) > 1:
            # keep it simple for now
            images = images[:1]

        if self.transport == "api":
            client = OllamaClient(host=self.host, timeout_sec=self.timeout_sec)
            # Prefer /api/chat (more stable formatting) for LLM/VLM
            if images:
                import base64

                import io

                buf = io.BytesIO()
                images[0].convert("RGB").save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                res = client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt, "images": [b64]}],
                    options=self.options,
                    think=self.think,
                )
                return _extract_content_from_response(res)

            res = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=self.options,
                think=self.think,
            )
            return _extract_content_from_response(res)

        if not images:
            proc = subprocess.run(
                ["ollama", "run", self.model, prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_sec,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ollama run failed\nmodel={self.model}\nstderr:\n{proc.stderr.strip()}\n")
            return (proc.stdout or "").strip()

        with tempfile.TemporaryDirectory(prefix="ollamaparser_vlm_") as td:
            img_path = Path(td) / "input.png"
            images[0].convert("RGB").save(img_path)

            # Pattern used by deepseek-ocr: "/path/to/image\nPROMPT"
            full = f"{os.path.abspath(str(img_path))}\n{prompt}"
            proc = subprocess.run(
                ["ollama", "run", self.model, full],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_sec,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ollama run failed\nmodel={self.model}\nstderr:\n{proc.stderr.strip()}\n")
            return (proc.stdout or "").strip()

"""Base protocols for OCR and VLM models."""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class VisionLanguageModel(Protocol):
    """
    Minimal VLM interface for:
    - chart understanding (image -> text)
    - metadata generation (text -> json; optionally with images)
    """

    def generate(self, *, prompt: str, images: list[Image.Image] | None = None) -> str: ...


class OCRModel(Protocol):
    def ocr(self, *, image: Image.Image) -> str: ...

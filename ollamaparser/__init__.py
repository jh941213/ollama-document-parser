"""
Ollama Parser - Document parsing with layout detection, OCR, and VLM.

This package provides tools for extracting structured content from PDF documents
using layout detection (DocLayout-YOLO), OCR (DeepSeek-OCR via Ollama), and
VLM analysis (Gemma3/Qwen via Ollama).

Modules:
    cli: Command-line interface for document parsing
    composer: Markdown composition utilities
    detection_filter: Detection filtering and deduplication
    detectors: Layout element detection strategies
    layout_order: Reading order inference
    markdown_writer: Markdown output generation
    ocr_vlm: OCR and VLM task execution
    pdf_render: PDF page rendering
    text_clean: Text post-processing utilities
"""

from ollamaparser import (
    cli,
    composer,
    detection_filter,
    detectors,
    layout_order,
    markdown_writer,
    ocr_vlm,
    pdf_render,
    text_clean,
)

__version__ = "0.1.0"

__all__ = [
    "cli",
    "composer",
    "detection_filter",
    "detectors",
    "layout_order",
    "markdown_writer",
    "ocr_vlm",
    "pdf_render",
    "text_clean",
]

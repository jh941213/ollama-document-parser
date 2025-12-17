"""Markdown output generation for parsed documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollamaparser.layout_types import LayoutElement
    from ollamaparser.pdf_render import PageAsset


@dataclass
class MarkdownSection:
    """A section of markdown content with metadata."""
    
    content: str
    element_type: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    source: str | None = None  # "ocr" | "vlm" | "pdfium"


@dataclass
class MarkdownWriter:
    """Builder for markdown document output."""
    
    style: str = "final"  # "final" | "debug"
    lines: list[str] = field(default_factory=list)
    last_written: str | None = None
    
    def __post_init__(self):
        self.lines = ["# Parsed Document", ""]
    
    def add_page_header(self, page_num: int, image_path: str | None = None, out_dir: str = "") -> None:
        """Add page header."""
        self.lines.append(f"## Page {page_num}")
        if self.style == "debug" and image_path:
            import os
            rel_path = os.path.relpath(image_path, out_dir) if out_dir else image_path
            self.lines.append(f"![]({rel_path})")
            self.lines.append("")
    
    def add_debug_elements_list(self, elements: list["LayoutElement"]) -> None:
        """Add element listing for debug mode."""
        if self.style != "debug":
            return
        
        self.lines.append("### Detected Elements")
        self.lines.append("")
        
        if elements:
            for e in elements:
                line = f"- {e.type} bbox=({int(e.bbox.x0)},{int(e.bbox.y0)},{int(e.bbox.x1)},{int(e.bbox.y1)})"
                if e.score is not None:
                    line += f" score={e.score:.3f}"
                if e.label:
                    line += f" label={e.label}"
                self.lines.append(line)
        else:
            self.lines.append("- (none)")
        self.lines.append("")
    
    def add_debug_section_header(self, title: str) -> None:
        """Add section header for debug mode."""
        if self.style != "debug":
            return
        self.lines.append(f"### {title}")
        self.lines.append("")
    
    def add_ocr_text(
        self,
        text: str,
        element: "LayoutElement | None" = None,
        is_figure: bool = False,
    ) -> bool:
        """
        Add OCR text output.
        
        Returns True if text was actually added.
        """
        text = (text or "").strip()
        
        # Skip empty in final mode
        if self.style == "final" and not text:
            return False
        
        # In final mode, skip figure OCR unless it's a structured table
        if self.style == "final" and is_figure:
            if text.count("|") < 6 or "\n| ---" not in text:
                return False
        
        # Skip duplicates
        if text and self.last_written and text == self.last_written.strip():
            return False
        
        # Debug mode: add element info
        if self.style == "debug" and element:
            self.lines.append(
                f"#### OCR ({element.type}) bbox=({int(element.bbox.x0)},{int(element.bbox.y0)},"
                f"{int(element.bbox.x1)},{int(element.bbox.y1)})"
            )
            self.lines.append("")
        
        self.lines.append(text or "(empty)")
        self.lines.append("")
        self.last_written = text
        return True
    
    def add_vlm_interpretation(
        self,
        text: str,
        element: "LayoutElement | None" = None,
        image_path: str | None = None,
    ) -> bool:
        """
        Add VLM interpretation output.
        
        Returns True if text was actually added.
        """
        text = (text or "").strip()
        
        # Skip empty in final mode
        if self.style == "final" and not text:
            return False
        
        # Skip duplicates
        if text and self.last_written and text == self.last_written.strip():
            return False
        
        # Add image reference
        if image_path:
            self.lines.append(f"![figure]({image_path})")
            self.lines.append("")
        
        # Debug mode: add element info
        if self.style == "debug" and element:
            self.lines.append(
                f"#### VLM ({element.type}) bbox=({int(element.bbox.x0)},{int(element.bbox.y0)},"
                f"{int(element.bbox.x1)},{int(element.bbox.y1)})"
            )
            self.lines.append("")
        
        self.lines.append("**이미지 해석:**")
        self.lines.append("")
        self.lines.append(text or "(empty)")
        self.lines.append("")
        self.last_written = text
        return True
    
    def add_pdfium_text(self, text: str, ocr_enabled: bool = False) -> None:
        """Add PDFium extracted text."""
        if not text:
            return
        
        if self.style == "debug":
            self.lines.append("")
            self.lines.append("### Extracted Text (PDFium)")
            self.lines.append("")
            self.lines.append("```")
            self.lines.append(text)
            self.lines.append("```")
        elif self.style == "final" and not ocr_enabled:
            # In final mode, only show PDFium text if OCR is disabled
            self.lines.append(text)
        
        self.lines.append("")
    
    def add_raw_text(self, text: str) -> None:
        """Add raw text without any formatting."""
        if text:
            self.lines.append(text.strip())
            self.lines.append("")
    
    def end_page(self) -> None:
        """End current page."""
        self.lines.append("")
        self.last_written = None
    
    def to_string(self) -> str:
        """Get the complete markdown content."""
        return "\n".join(self.lines)
    
    def write(self, path: str) -> None:
        """Write markdown to file."""
        from pathlib import Path
        Path(path).write_text(self.to_string(), encoding="utf-8")





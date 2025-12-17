"""Document Composer - Assembles parsed elements into final output formats."""

from __future__ import annotations

from dataclasses import dataclass

from ollamaparser.layout_types import DetectedElement
from ollamaparser.types import PageAsset


@dataclass(frozen=True)
class ParsedElement:
    """Represents a single parsed element from a document page."""
    page_index: int
    type: str
    text: str
    bbox: tuple[float, float, float, float] | None = None
    source: str | None = None


def compose_markdown(pages: list[PageAsset], elements: list[ParsedElement]) -> str:
    """Compose parsed elements into a Markdown document."""
    by_page: dict[int, list[ParsedElement]] = {}
    for e in elements:
        by_page.setdefault(e.page_index, []).append(e)

    lines: list[str] = ["# Parsed Document", ""]
    for p in pages:
        lines.append(f"## Page {p.page_index+1}")
        lines.append(f"![](pages/page_{p.page_index+1:04d}.png)")
        lines.append("")

        page_elements = by_page.get(p.page_index, [])
        if page_elements:
            for e in page_elements:
                lines.append(f"### {e.type.upper()}")
                if e.bbox:
                    lines.append(f"- bbox: {tuple(int(x) for x in e.bbox)}")
                if e.source:
                    lines.append(f"- source: {e.source}")
                lines.append("")
                lines.append(e.text.strip() or "(empty)")
                lines.append("")
        elif p.text:
            lines.append("### TEXT (PDFium)")
            lines.append("")
            lines.append(p.text.strip())
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def create_elements(
    page: PageAsset, detected: list[DetectedElement], *, source: str
) -> list[ParsedElement]:
    """Create ParsedElement instances from detected layout elements."""
    return [
        ParsedElement(
            page_index=page.page_index,
            type=e.type,
            text="",
            bbox=(e.bbox.x0, e.bbox.y0, e.bbox.x1, e.bbox.y1),
            source=source,
        )
        for e in detected
    ]


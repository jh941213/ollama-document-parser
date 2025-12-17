from __future__ import annotations

import os
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from ollamaparser.types import PageAsset


def _safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def render_pdf_pages(
    pdf_path: str,
    out_pages_dir: str,
    *,
    scale: float = 2.0,
    extract_text: bool = True,
    page_start: int = 1,
    page_end: int | None = None,
) -> list[PageAsset]:
    """
    Render PDF to per-page PNG files using PDFium.
    """
    pdf_path = os.path.abspath(pdf_path)
    out_pages_dir = os.path.abspath(out_pages_dir)
    _safe_mkdir(out_pages_dir)

    pdf = pdfium.PdfDocument(pdf_path)
    assets: list[PageAsset] = []

    total_pages = len(pdf)
    if page_start < 1:
        page_start = 1
    if page_end is None or page_end > total_pages:
        page_end = total_pages
    if page_end < page_start:
        page_end = page_start

    for page_index in range(page_start - 1, page_end):
        page = pdf.get_page(page_index)

        # Render
        bitmap = page.render(scale=scale)
        pil_img: Image.Image = bitmap.to_pil()

        w, h = pil_img.size
        img_path = os.path.join(out_pages_dir, f"page_{page_index+1:04d}.png")
        pil_img.save(img_path)

        # Text extraction (best-effort)
        text: str | None = None
        if extract_text:
            try:
                textpage = page.get_textpage()
                text = (textpage.get_text_range() or "").strip() or None
            except Exception:
                text = None

        assets.append(
            PageAsset(
                page_index=page_index,
                image_path=img_path,
                width=w,
                height=h,
                text=text,
            )
        )

        page.close()

    pdf.close()
    return assets



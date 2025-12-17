from __future__ import annotations

from typing import Protocol

from PIL import Image

from ollamaparser.layout_types import BBox, DetectedElement, ElementType
from ollamaparser.types import PageAsset


class ElementDetector(Protocol):
    def detect(self, page: PageAsset) -> list[DetectedElement]: ...


class FullPageFallbackDetector:
    """
    Model-free fallback: mark the entire page as a single `text` region.
    This keeps the pipeline runnable even without DocLayout-YOLO.
    """

    def __init__(self, element_type: ElementType = "text"):
        self.element_type = element_type

    def detect(self, page: PageAsset) -> list[DetectedElement]:
        return [
            DetectedElement(
                page_index=page.page_index,
                type=self.element_type,
                bbox=BBox(x0=0.0, y0=0.0, x1=float(page.width), y1=float(page.height)),
                score=None,
                label="full_page",
            )
        ]


class DocLayoutYOLODetector:
    """
    DocLayout-YOLO integration (optional).

    Upstream: https://github.com/opendatalab/DocLayout-YOLO
    Uses the official `doclayout-yolo` SDK (YOLOv10) when available.
    - weights_path: local .pt file
    """

    def __init__(
        self,
        *,
        weights_path: str,
        from_pretrained: bool = False,
        pretrained_filename: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
        imgsz: int = 1024,
        conf: float = 0.2,
        device: str | None = None,
        table_keywords: tuple[str, ...] = ("table",),
        chart_keywords: tuple[str, ...] = ("chart", "plot", "graph"),
        image_keywords: tuple[str, ...] = ("figure", "image", "pic", "photo"),
    ):
        self.weights_path = weights_path
        self.from_pretrained = from_pretrained
        self.pretrained_filename = pretrained_filename
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.table_keywords = table_keywords
        self.chart_keywords = chart_keywords
        self.image_keywords = image_keywords
        self._resolved_weights_path: str | None = None
        self._last_pred = None  # last ultralytics Result for debugging/visualization

        # DocLayout-YOLO (DocStructBench) label patterns we want to treat as TEXT, even if they contain "figure"/"table".
        # Examples: "figure_caption", "table_caption", "table_footnote", "formula_caption"
        self._text_override_keywords: tuple[str, ...] = ("_caption", "footnote", "abandon")

    def detect(self, page: PageAsset) -> list[DetectedElement]:
        # Prefer official SDK
        try:
            from doclayout_yolo import YOLOv10  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "doclayout-yolo is not installed. Install with: `uv sync --extra detector`"
            ) from e

        weights_path = self.weights_path
        if self.from_pretrained:
            # Prefer explicit HF download (matches upstream README) over YOLOv10.from_pretrained(),
            # which may fallback to default yolov10n.pt if not wired correctly.
            if self._resolved_weights_path is None:
                try:
                    from huggingface_hub import hf_hub_download
                except Exception as e:
                    raise RuntimeError(
                        "huggingface-hub is required for --yolo-from-pretrained. "
                        "Install with: `uv sync --extra detector`"
                    ) from e
                self._resolved_weights_path = hf_hub_download(
                    repo_id=weights_path, filename=self.pretrained_filename
                )
            weights_path = self._resolved_weights_path

        model = YOLOv10(weights_path)

        det_res = model.predict(
            page.image_path,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device or "cpu",
        )
        if not det_res:
            return []

        pred0 = det_res[0]
        self._last_pred = pred0
        names = getattr(pred0, "names", {}) or {}
        boxes = getattr(pred0, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().tolist()  # type: ignore[attr-defined]
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(xyxy)  # type: ignore[attr-defined]
        clss = boxes.cls.cpu().tolist() if boxes.cls is not None else [None] * len(xyxy)  # type: ignore[attr-defined]

        out: list[DetectedElement] = []
        for (x0, y0, x1, y1), c, cls_id in zip(xyxy, confs, clss):
            label = None
            if cls_id is not None:
                try:
                    label = names.get(int(cls_id), str(int(cls_id)))
                except Exception:
                    label = str(cls_id)

            etype: ElementType = "text"
            low = (label or "").lower()
            # Priority: explicit text overrides first (so "figure_caption" doesn't become image)
            if any(k in low for k in self._text_override_keywords):
                etype = "text"
            elif "plain text" in low or low == "title":
                etype = "text"
            elif low == "figure":
                etype = "image"
            elif low == "table":
                etype = "table"
            else:
                if any(k in low for k in self.table_keywords):
                    etype = "table"
                elif any(k in low for k in self.image_keywords):
                    etype = "image"
                elif any(k in low for k in self.chart_keywords):
                    etype = "chart"

            out.append(
                DetectedElement(
                    page_index=page.page_index,
                    type=etype,
                    bbox=BBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)).clamp(page.width, page.height),
                    score=float(c) if c is not None else None,
                    label=label,
                )
            )

        out.sort(key=lambda e: (e.bbox.y0, e.bbox.x0))
        return out


def crop_element_image(page: PageAsset, el: DetectedElement) -> Image.Image:
    img = Image.open(page.image_path).convert("RGB")
    b = el.bbox.clamp(page.width, page.height)
    return img.crop((int(b.x0), int(b.y0), int(b.x1), int(b.y1)))


class PdfiumTextRectDetector:
    """
    Lightweight detector using PDFium text rectangles.
    Produces many `text` elements with bounding boxes, useful for column inference
    without heavy ML models.
    """

    def __init__(self, *, pdf_path: str, min_area: int = 200, max_rects: int = 2000):
        self.pdf_path = pdf_path
        self.min_area = min_area
        self.max_rects = max_rects
        self._doc = None

    def detect(self, page: PageAsset) -> list[DetectedElement]:
        try:
            import pypdfium2 as pdfium
        except Exception:
            return FullPageFallbackDetector().detect(page)

        if self._doc is None:
            self._doc = pdfium.PdfDocument(self.pdf_path)

        pdf_page = self._doc.get_page(page.page_index)
        try:
            pdf_w, pdf_h = pdf_page.get_size()
            textpage = pdf_page.get_textpage()
            n = int(textpage.count_rects())
            if n <= 0:
                return FullPageFallbackDetector().detect(page)

            x_scale = float(page.width) / float(pdf_w)
            y_scale = float(page.height) / float(pdf_h)

            out: list[DetectedElement] = []
            for i in range(min(n, self.max_rects)):
                l, b, r, t = textpage.get_rect(i)
                # PDF coords: origin bottom-left; image coords: origin top-left.
                x0 = float(l) * x_scale
                x1 = float(r) * x_scale
                y0 = (float(pdf_h) - float(t)) * y_scale
                y1 = (float(pdf_h) - float(b)) * y_scale

                bb = BBox(
                    x0=min(x0, x1),
                    y0=min(y0, y1),
                    x1=max(x0, x1),
                    y1=max(y0, y1),
                ).clamp(page.width, page.height)
                area = max(0.0, (bb.x1 - bb.x0)) * max(0.0, (bb.y1 - bb.y0))
                if area < float(self.min_area):
                    continue
                out.append(
                    DetectedElement(
                        page_index=page.page_index,
                        type="text",
                        bbox=bb,
                        score=None,
                        label="pdfium_text_rect",
                    )
                )

            if not out:
                return FullPageFallbackDetector().detect(page)
            out.sort(key=lambda e: (e.bbox.y0, e.bbox.x0))
            return out
        finally:
            try:
                pdf_page.close()
            except Exception:
                pass

    def __del__(self):
        try:
            if self._doc is not None:
                self._doc.close()
        except Exception:
            pass



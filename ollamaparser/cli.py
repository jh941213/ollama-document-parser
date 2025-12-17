"""
Ollama Parser CLI - Document parsing with layout detection, OCR, and VLM.

Usage:
    ollamaparser-parse --pdf input.pdf --out_dir ./output --detector doclayout-yolo ...
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from ollamaparser.detection_filter import FilteredElement, filter_elements, find_nearby_context_elements
from ollamaparser.detectors import DocLayoutYOLODetector, FullPageFallbackDetector, PdfiumTextRectDetector, crop_element_image
from ollamaparser.layout_order import infer_reading_order
from ollamaparser.markdown_writer import MarkdownWriter
from ollamaparser.metadata import generate_metadata_json
from ollamaparser.ocr_vlm import DEFAULT_VLM_PROMPT, OCRTask, TaskExecutor, VLMTask
from ollamaparser.pdf_render import PageAsset, render_pdf_pages
from ollamaparser.providers.ollama.ocr import OllamaOCR
from ollamaparser.providers.ollama.vlm import OllamaVLM
from ollamaparser.quality_check import page_quality
from ollamaparser.text_clean import clean_deepseek_ocr_text, is_likely_ocr_hallucination

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()


# =============================================================================
# CLI Options Dataclass (for cleaner function signature)
# =============================================================================

def _create_providers(
    ocr_provider: str,
    vlm_provider: str,
    ollama_model: str,
    ollama_vlm_model: str,
    ollama_host: str,
    ollama_transport: str,
    ollama_timeout_sec: int,
    vlm_think: bool,
    vlm_options: dict,
):
    """Create OCR and VLM provider instances."""
    ocr = None
    vlm = None
    
    if ocr_provider == "deepseek-ollama":
        ocr = OllamaOCR(
            model=ollama_model,
            host=ollama_host,
            transport=ollama_transport,
            timeout_sec=ollama_timeout_sec,
        )
    
    if vlm_provider in ("ollama-vlm", "gemma-ollama"):
        vlm = OllamaVLM(
            model=ollama_vlm_model,
            host=ollama_host,
            transport=ollama_transport,
            timeout_sec=ollama_timeout_sec,
            think=vlm_think,
            options=vlm_options,
        )
    
    return ocr, vlm


def _create_detector(
    detector_type: str,
    yolo_weights: str | None,
    yolo_from_pretrained: bool,
    yolo_pretrained_filename: str,
    pdf_path: str,
):
    """Create element detector instance."""
    if detector_type == "doclayout-yolo":
        if not yolo_weights:
            raise typer.BadParameter("--yolo-weights is required when --detector doclayout-yolo")
        return DocLayoutYOLODetector(
            weights_path=yolo_weights if yolo_from_pretrained else os.path.abspath(yolo_weights),
            from_pretrained=yolo_from_pretrained,
            pretrained_filename=yolo_pretrained_filename,
        )
    elif detector_type == "pdfium-text":
        return PdfiumTextRectDetector(pdf_path=pdf_path)
    else:
        return FullPageFallbackDetector()


def _save_detections(
    det,
    elements: list,
    asset: PageAsset,
    out_dir: str,
    detector_type: str,
    image_format: str,
    jpeg_quality: int,
    log_fn,
):
    """Save detection outputs (JSON + annotated image)."""
    det_dir = Path(out_dir) / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    det_json = [
        {
            "type": e.type,
            "label": e.label,
            "score": e.score,
            "bbox": [e.bbox.x0, e.bbox.y0, e.bbox.x1, e.bbox.y1],
        }
        for e in elements
    ]
    (det_dir / f"page_{asset.page_index + 1:04d}.json").write_text(
        json.dumps(det_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    
    # Annotated image for YOLO
    if detector_type == "doclayout-yolo" and hasattr(det, "_last_pred") and det._last_pred is not None:
        pred = det._last_pred
        fmt = image_format.lower().strip()
        if fmt not in ("jpg", "png"):
            fmt = "jpg"
        annotated_path = det_dir / f"page_{asset.page_index + 1:04d}_yolo.{fmt}"
        
        try:
            if hasattr(pred, "plot"):
                annotated = pred.plot(pil=True)
                if fmt == "jpg":
                    annotated.convert("RGB").save(annotated_path, format="JPEG", quality=jpeg_quality)
                else:
                    annotated.save(annotated_path)
            else:
                from PIL import Image
                arr = pred.plot()
                im = Image.fromarray(arr)
                if fmt == "jpg":
                    im.convert("RGB").save(annotated_path, format="JPEG", quality=jpeg_quality)
                else:
                    im.save(annotated_path)
            log_fn(f"Saved YOLO annotated image: {annotated_path}")
        except Exception as e:
            log_fn(f"WARNING: failed to save YOLO annotated image: {e}")
    
    log_fn(f"Saved detections for page {asset.page_index + 1} → {det_dir}")


def _build_tasks(
    filtered: list[FilteredElement],
    ocr,
    vlm,
    ocr_scope: str,
    vlm_scope: str,
    ocr_elements_mode: str,
    figure_ocr: bool,
    figure_vlm: bool,
) -> tuple[list[OCRTask], list[VLMTask]]:
    """Build OCR and VLM task lists from filtered elements."""
    ocr_tasks: list[OCRTask] = []
    vlm_tasks: list[VLMTask] = []
    
    for fe in filtered:
        idx = fe.index
        e = fe.element
        label = (getattr(e, "label", None) or "").lower()
        is_title = "title" in label and "caption" not in label
        
        # OCR targets
        if ocr and ocr_scope == "elements" and (e.type in ("text", "table") or is_title):
            if ocr_elements_mode == "markdown":
                mode = "markdown"
            elif ocr_elements_mode == "free":
                mode = "free"
            else:
                mode = "<|grounding|>Convert the table to markdown." if e.type == "table" else "free"
            ocr_tasks.append(OCRTask(index=idx, element=e, mode=mode))
        
        # VLM targets
        if vlm and vlm_scope == "visuals":
            if e.type == "image" and "figure" in label and "caption" not in label:
                if figure_ocr and ocr and ocr_scope == "elements":
                    mode = ocr_elements_mode if ocr_elements_mode in ("markdown", "free") else "free"
                    ocr_tasks.append(OCRTask(index=idx, element=e, mode=mode))
                if figure_vlm:
                    vlm_tasks.append(VLMTask(index=idx, element=e, prompt=DEFAULT_VLM_PROMPT))
    
    return ocr_tasks, vlm_tasks


def _collect_figure_context(
    vlm_tasks: list[VLMTask],
    all_elements: list,
    ocr,
    asset: PageAsset,
) -> dict[int, str]:
    """Collect nearby caption/title context for VLM figures."""
    context_map: dict[int, str] = {}
    
    if not ocr:
        return context_map
    
    for task in vlm_tasks:
        nearby = find_nearby_context_elements(task.element, all_elements)
        context_parts = []
        
        for elem in nearby:
            try:
                img = crop_element_image(asset, elem)
                txt = ocr.ocr(image=img, mode="free")
                txt = clean_deepseek_ocr_text(txt)
                if txt and len(txt) > 3 and not is_likely_ocr_hallucination(txt):
                    context_parts.append(txt)
            except Exception:
                pass
        
        if context_parts:
            context_map[task.index] = "\n".join(context_parts)
            task.context = context_map[task.index]
    
    return context_map


# =============================================================================
# Main CLI Command
# =============================================================================

@app.command()
def parse(
    # Input/Output
    pdf: str = typer.Option(..., "--pdf", help="Path to input PDF"),
    out_dir: str = typer.Option(..., "--out_dir", help="Output directory"),
    
    # PDF Rendering
    scale: float = typer.Option(2.0, "--scale", help="PDF render scale"),
    extract_text: bool = typer.Option(True, "--extract-text/--no-extract-text", help="Extract selectable text"),
    page_start: int = typer.Option(1, "--page-start", help="Start page (1-based)"),
    page_end: int | None = typer.Option(None, "--page-end", help="End page (1-based)"),
    
    # Detector
    detector: str = typer.Option("fallback", "--detector", help="Detector: fallback|pdfium-text|doclayout-yolo"),
    yolo_weights: str | None = typer.Option(None, "--yolo-weights", help="YOLO weights path or HF repo_id"),
    yolo_from_pretrained: bool = typer.Option(False, "--yolo-from-pretrained", help="Load from HF"),
    yolo_pretrained_filename: str = typer.Option(
        "doclayout_yolo_docstructbench_imgsz1024.pt", "--yolo-pretrained-filename"
    ),
    yolo_dump_labels: bool = typer.Option(False, "--yolo-dump-labels", help="Print YOLO labels and exit"),
    
    # Detection Output
    save_detections: bool = typer.Option(False, "--save-detections", help="Save detection outputs"),
    detections_image_format: str = typer.Option("jpg", "--detections-image-format"),
    detections_jpeg_quality: int = typer.Option(85, "--detections-jpeg-quality"),
    
    # OCR
    ocr_provider: str = typer.Option("none", "--ocr-provider", help="OCR: none|deepseek-ollama"),
    ocr_scope: str = typer.Option("page", "--ocr-scope", help="OCR scope: none|page|elements"),
    ocr_mode: str = typer.Option("markdown", "--ocr-mode", help="OCR mode: markdown|free"),
    ocr_elements_mode: str = typer.Option("auto", "--ocr-elements-mode", help="Element OCR: auto|free|markdown"),
    
    # VLM
    vlm_provider: str = typer.Option("none", "--vlm-provider", help="VLM: none|ollama-vlm"),
    vlm_scope: str = typer.Option("none", "--vlm-scope", help="VLM scope: none|page|visuals"),
    vlm_use_image: bool = typer.Option(False, "--vlm-use-image/--vlm-no-image", help="Pass image to VLM"),
    vlm_think: bool = typer.Option(False, "--vlm-think/--no-vlm-think", help="Enable VLM thinking mode"),
    vlm_temperature: float = typer.Option(0.1, "--vlm-temperature"),
    vlm_repeat_penalty: float = typer.Option(1.2, "--vlm-repeat-penalty"),
    vlm_num_predict: int = typer.Option(350, "--vlm-num-predict"),
    
    # Ollama
    ollama_model: str = typer.Option("deepseek-ocr:latest", "--ollama-model", help="Ollama OCR model"),
    ollama_vlm_model: str = typer.Option("gemma3:27b", "--ollama-vlm-model", help="Ollama VLM model"),
    ollama_host: str = typer.Option("", "--ollama-host", help="Ollama host"),
    ollama_transport: str = typer.Option("api", "--ollama-transport", help="Ollama transport: api|cli"),
    ollama_timeout_sec: int = typer.Option(600, "--ollama-timeout-sec"),
    
    # Figure Processing
    figure_ocr: bool = typer.Option(True, "--figure-ocr/--no-figure-ocr", help="OCR figure regions"),
    figure_vlm: bool = typer.Option(True, "--figure-vlm/--no-figure-vlm", help="VLM on figures"),
    
    # Execution
    execution_mode: str = typer.Option("grouped", "--execution-mode", help="interleaved|grouped|parallel"),
    parallel_workers: int = typer.Option(4, "--parallel-workers"),
    max_elements_per_page: int = typer.Option(30, "--max-elements-per-page"),
    
    # Output
    md_style: str = typer.Option("final", "--md-style", help="Markdown style: final|debug"),
    reading_order: str = typer.Option("auto", "--reading-order", help="Reading order: auto|raw"),
    layout_check: bool = typer.Option(True, "--layout-check/--no-layout-check"),
    metadata: bool = typer.Option(True, "--metadata/--no-metadata"),
    device: str = typer.Option("mps", "--device", help="Device: mps|cpu|cuda"),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """Parse PDF documents with layout detection, OCR, and VLM analysis."""
    
    # Setup paths
    pdf = os.path.abspath(pdf)
    out_dir = os.path.abspath(out_dir)
    pages_dir = os.path.join(out_dir, "pages")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(out_dir, "run.log")
    
    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            Path(log_path).write_text(
                (Path(log_path).read_text(encoding="utf-8") if Path(log_path).exists() else "") + line + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        if verbose:
            console.print(line)
    
    class TimedContext:
        """Context manager for timing operations."""
        def __init__(self, label: str):
            self.label = label
            self.t0 = 0.0
        
        def __enter__(self):
            self.t0 = time.time()
            log(f"START {self.label}")
            return self
        
        def __exit__(self, exc_type, exc, tb):
            dt = time.time() - self.t0
            if exc:
                log(f"FAIL  {self.label} ({dt:.2f}s): {exc}")
            else:
                log(f"DONE  {self.label} ({dt:.2f}s)")
            return False
    
    timed = TimedContext
    
    # Log configuration
    log("ollamaparser-parse started")
    log(f"pdf={pdf}")
    log(f"out_dir={out_dir}")
    log(f"detector={detector}, reading_order={reading_order}")
    log(f"ocr_provider={ocr_provider}, ocr_scope={ocr_scope}")
    log(f"vlm_provider={vlm_provider}, vlm_scope={vlm_scope}")
    
    # Handle --yolo-dump-labels early exit
    if detector == "doclayout-yolo" and yolo_dump_labels:
        _handle_yolo_dump_labels(yolo_weights, yolo_from_pretrained, yolo_pretrained_filename, log)
        return
    
    # Render PDF pages
    with timed(f"render_pdf_pages scale={scale} pages={page_start}..{page_end or 'end'}"):
        assets = render_pdf_pages(
            pdf, pages_dir, scale=scale, extract_text=extract_text,
            page_start=page_start, page_end=page_end,
        )
    log(f"Rendered {len(assets)} pages → {pages_dir}")
    
    # Create detector and providers
    det = _create_detector(detector, yolo_weights, yolo_from_pretrained, yolo_pretrained_filename, pdf)
    ocr, vlm = _create_providers(
        ocr_provider, vlm_provider, ollama_model, ollama_vlm_model,
        ollama_host, ollama_transport, ollama_timeout_sec, vlm_think,
        {"temperature": vlm_temperature, "repeat_penalty": vlm_repeat_penalty, "num_predict": vlm_num_predict},
    )
    
    # Create task executor
    executor = TaskExecutor(
        ocr=ocr,
        vlm=vlm,
        use_vlm_image=vlm_use_image,
        crops_dir=os.path.join(out_dir, "crops"),
        log_fn=log,
    )
    
    # Initialize outputs
    md_writer = MarkdownWriter(style=md_style)
    layout_quality: list[dict] = []
    extracted_by_page: dict[int, list[dict]] = {}
    
    # Process each page
    for asset in assets:
        with timed(f"detect_elements page={asset.page_index + 1}"):
            elements = det.detect(asset)
        
        # Save detections if requested
        if save_detections:
            try:
                _save_detections(
                    det, elements, asset, out_dir, detector,
                    detections_image_format, detections_jpeg_quality, log
                )
            except Exception as e:
                log(f"WARNING: failed to save detections: {e}")
        
        # Apply reading order
        reading = None
        ordered = elements
        if reading_order == "auto" and elements:
            reading = infer_reading_order(elements, page_width=asset.width, page_height=asset.height)
            ordered = [elements[i] for i in reading.ordered_indices]
        elif reading_order == "raw":
            reading = infer_reading_order(elements, page_width=asset.width, page_height=asset.height, max_columns=1)
        
        log(f"page={asset.page_index + 1}: elements={len(elements)}")
        
        # Write page header
        md_writer.add_page_header(asset.page_index + 1, asset.image_path, out_dir)
        md_writer.add_debug_elements_list(ordered)
        md_writer.lines.append("")
        
        # Page extractions
        page_extractions: list[dict] = []
        
        # Page-level OCR
        if ocr and ocr_scope == "page":
            from PIL import Image
            page_img = Image.open(asset.image_path).convert("RGB")
            with timed(f"ocr_page page={asset.page_index + 1}"):
                txt = ocr.ocr(image=page_img, mode=ocr_mode)
            page_extractions.append({
                "scope": "page", "type": "ocr", "provider": ocr_provider, "mode": ocr_mode, "text": txt
            })
            md_writer.add_ocr_text(txt)
        
        # Page-level VLM
        if vlm and vlm_scope == "page":
            from PIL import Image
            page_img = Image.open(asset.image_path).convert("RGB")
            prompt = "/no_think\n[시스템] 반드시 한국어로만 답변하세요.\n\n이 페이지를 분석해서 핵심 포인트를 정리해줘."
            with timed(f"vlm_page page={asset.page_index + 1}"):
                txt = vlm.generate(prompt=prompt, images=[page_img])
            page_extractions.append({
                "scope": "page", "type": "vlm", "provider": vlm_provider, "text": txt
            })
            md_writer.add_vlm_interpretation(txt)
        
        # Element-level processing
        if (ocr and ocr_scope == "elements") or (vlm and vlm_scope == "visuals"):
            limited = ordered[:max_elements_per_page]
            filtered = filter_elements(limited)
            
            ocr_tasks, vlm_tasks = _build_tasks(
                filtered, ocr, vlm, ocr_scope, vlm_scope,
                ocr_elements_mode, figure_ocr, figure_vlm
            )
            
            # Collect context for figures
            if vlm and vlm_scope == "visuals" and ocr:
                _collect_figure_context(vlm_tasks, limited, ocr, asset)
            
            log(f"page={asset.page_index + 1}: ocr_tasks={len(ocr_tasks)}, vlm_tasks={len(vlm_tasks)}")
            
            # Execute tasks
            if execution_mode == "parallel":
                with timed(f"execute_parallel page={asset.page_index + 1}"):
                    ocr_results, vlm_results = executor.execute_parallel(
                        ocr_tasks, vlm_tasks, asset, parallel_workers,
                        ocr_provider, vlm_provider
                    )
            elif execution_mode == "interleaved":
                with timed(f"execute_interleaved page={asset.page_index + 1}"):
                    ocr_results, vlm_results = executor.execute_interleaved(
                        ocr_tasks, vlm_tasks, asset, len(limited),
                        ocr_provider, vlm_provider
                    )
            else:
                with timed(f"execute_sequential page={asset.page_index + 1}"):
                    ocr_results, vlm_results = executor.execute_sequential(
                        ocr_tasks, vlm_tasks, asset, ocr_provider, vlm_provider
                    )
            
            # Write results in reading order
            for idx, e in enumerate(limited):
                if idx in ocr_results:
                    result = ocr_results[idx]
                    is_figure = e.type == "image"
                    md_writer.add_ocr_text(result.text, e, is_figure)
                    page_extractions.append({**result.metadata, "text": result.text})
                
                if idx in vlm_results:
                    result = vlm_results[idx]
                    md_writer.add_vlm_interpretation(result.text, e, result.image_path)
                    page_extractions.append({**result.metadata, "text": result.text, "crop_image": result.image_path})
        
        # PDFium text
        if asset.text:
            md_writer.add_pdfium_text(asset.text, ocr_enabled=(ocr is not None and ocr_scope != "none"))
        
        md_writer.end_page()
        
        if page_extractions:
            extracted_by_page[asset.page_index] = page_extractions
        
        # Layout quality check
        if layout_check:
            with timed(f"layout_check page={asset.page_index + 1}"):
                q = page_quality(page=asset, elements=elements, reading=reading, extracted_text=None)
            layout_quality.append(q.__dict__)
    
    # Write markdown output
    md_path = os.path.join(out_dir, "out.md")
    md_writer.write(md_path)
    log(f"Wrote {md_path}")
    
    # Write metadata
    if metadata:
        meta_path = os.path.join(out_dir, "out.meta.json")
        base_meta = _build_metadata(
            pdf, detector, reading_order, assets, det,
            layout_check, layout_quality, extracted_by_page,
            md_path, vlm
        )
        Path(meta_path).write_text(json.dumps(base_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"Wrote {meta_path}")
    
    log("ollamaparser-parse finished")


def _handle_yolo_dump_labels(
    yolo_weights: str | None,
    yolo_from_pretrained: bool,
    yolo_pretrained_filename: str,
    log,
):
    """Handle --yolo-dump-labels flag."""
    if not yolo_weights:
        raise typer.BadParameter("--yolo-weights is required for --yolo-dump-labels")
    
    from doclayout_yolo import YOLOv10
    
    if yolo_from_pretrained:
        from huggingface_hub import hf_hub_download
        local_pt = hf_hub_download(repo_id=yolo_weights, filename=yolo_pretrained_filename)
        log(f"Downloaded weights: {local_pt}")
        model = YOLOv10(local_pt)
    else:
        model = YOLOv10(os.path.abspath(yolo_weights))
    
    names = getattr(model, "names", None)
    log(f"DocLayout-YOLO names={names}")
    if isinstance(names, dict):
        joined = " | ".join(str(v) for _, v in sorted(names.items()))
        log(f"DocLayout-YOLO labels={joined}")


def _build_metadata(
    pdf: str,
    detector: str,
    reading_order: str,
    assets: list[PageAsset],
    det,
    layout_check: bool,
    layout_quality: list[dict],
    extracted_by_page: dict[int, list[dict]],
    md_path: str,
    vlm,
) -> dict[str, Any]:
    """Build metadata dictionary."""
    base_meta = {
        "pdf": pdf,
        "detector": detector,
        "reading_order": reading_order,
        "pages": [
            {
                "page_index": a.page_index,
                "image_path": a.image_path,
                "width": a.width,
                "height": a.height,
                "text_present": bool(a.text),
                "elements": [
                    {
                        "type": e.type,
                        "bbox": [e.bbox.x0, e.bbox.y0, e.bbox.x1, e.bbox.y1],
                        "score": e.score,
                        "label": e.label,
                    }
                    for e in det.detect(a)
                ],
            }
            for a in assets
        ],
    }
    
    if layout_check:
        base_meta["layout_quality"] = layout_quality
    if extracted_by_page:
        base_meta["extractions"] = extracted_by_page
    
    md_text = Path(md_path).read_text(encoding="utf-8")
    base_meta["document_metadata"] = generate_metadata_json(md=md_text, vlm=vlm)
    
    return base_meta


def main():
    """Entry point for ollamaparser-parse CLI."""
    app()

"""OCR and VLM task execution utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ollamaparser.detectors import crop_element_image
from ollamaparser.text_clean import clean_deepseek_ocr_text, clean_vlm_text, is_likely_ocr_hallucination

if TYPE_CHECKING:
    from PIL import Image
    from ollamaparser.layout_types import LayoutElement
    from ollamaparser.pdf_render import PageAsset
    from ollamaparser.providers.base import BaseOCR, BaseVLM


@dataclass
class OCRTask:
    """An OCR task to execute."""
    
    index: int
    element: "LayoutElement"
    mode: str  # "free" | "markdown" | custom prompt


@dataclass
class VLMTask:
    """A VLM task to execute."""
    
    index: int
    element: "LayoutElement"
    prompt: str
    context: str | None = None  # Nearby caption/title text


@dataclass
class TaskResult:
    """Result of an OCR or VLM task."""
    
    index: int
    text: str
    image_path: str | None = None  # For VLM: saved crop image path
    metadata: dict = field(default_factory=dict)


def build_vlm_prompt(
    base_prompt: str,
    nearby_context: str | None = None,
    ocr_context: str | None = None,
) -> str:
    """
    Build VLM prompt with optional context.
    
    Args:
        base_prompt: Base VLM prompt
        nearby_context: Text from nearby caption/title elements
        ocr_context: OCR text from the figure itself
    
    Returns:
        Final prompt string
    """
    if nearby_context:
        return (
            "/no_think\n"
            "[시스템] 반드시 한국어로만 답변하세요. 영어 사용 절대 금지.\n\n"
            f"이 그림의 제목/캡션: {nearby_context}\n\n"
            "위 제목을 참고하여 이 그림을 분석해서 아래 형식으로 답해:\n"
            "- 요약: (한 문장)\n"
            "- 핵심: (최대 3개, 숫자/단위 포함)\n"
            "- 주의: (1개)\n\n"
            "영어/반복/JSON 금지."
        )
    elif ocr_context:
        return (
            "/no_think\n"
            "[시스템] 반드시 한국어로만 답변하세요. 영어 사용 절대 금지.\n\n"
            "다음 OCR 텍스트를 기반으로 분석해:\n"
            f"```\n{ocr_context}\n```\n\n"
            "아래 형식으로만 답해:\n"
            "- 요약: (한 문장)\n"
            "- 핵심: (최대 3개, 숫자 포함)\n"
            "- 주의: (1개)\n\n"
            "영어/반복/JSON 금지."
        )
    else:
        return base_prompt


DEFAULT_VLM_PROMPT = (
    "/no_think\n"
    "[시스템] 반드시 한국어로만 답변하세요. 영어 사용 금지.\n\n"
    "이 그림을 분석해서 아래 형식으로 답해:\n"
    "- 요약: (한 문장)\n"
    "- 핵심: (최대 3개, 숫자/단위 포함)\n"
    "- 주의: (1개)\n\n"
    "영어/반복/코드블록/JSON 금지."
)


@dataclass
class TaskExecutor:
    """Executes OCR and VLM tasks with various execution modes."""
    
    ocr: "BaseOCR | None" = None
    vlm: "BaseVLM | None" = None
    use_vlm_image: bool = True
    crops_dir: str | None = None
    log_fn: Callable[[str], None] | None = None
    
    def _log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)
    
    def run_ocr(
        self,
        task: OCRTask,
        asset: "PageAsset",
        provider_name: str = "deepseek-ollama",
    ) -> TaskResult:
        """Execute a single OCR task."""
        if not self.ocr:
            return TaskResult(index=task.index, text="")
        
        img = crop_element_image(asset, task.element)
        txt = self.ocr.ocr(image=img, mode=task.mode)
        txt = clean_deepseek_ocr_text(txt)
        
        # Filter hallucinations
        if is_likely_ocr_hallucination(txt):
            self._log(f"  [SKIP] OCR hallucination detected: {txt[:50]!r}")
            txt = ""
        
        return TaskResult(
            index=task.index,
            text=txt,
            metadata={
                "scope": "element",
                "type": task.element.type,
                "provider": provider_name,
                "mode": task.mode,
                "bbox": [
                    task.element.bbox.x0,
                    task.element.bbox.y0,
                    task.element.bbox.x1,
                    task.element.bbox.y1,
                ],
            },
        )
    
    def run_vlm(
        self,
        task: VLMTask,
        asset: "PageAsset",
        ocr_results: dict[int, str] | None = None,
        provider_name: str = "ollama-vlm",
    ) -> TaskResult:
        """Execute a single VLM task."""
        if not self.vlm:
            return TaskResult(index=task.index, text="")
        
        # Build prompt with context
        ocr_ctx = ocr_results.get(task.index) if ocr_results else None
        prompt = build_vlm_prompt(task.prompt, task.context, ocr_ctx)
        
        # Optionally include image
        img = crop_element_image(asset, task.element) if self.use_vlm_image else None
        image_path = None
        
        # Save cropped image
        if img is not None and self.crops_dir:
            Path(self.crops_dir).mkdir(parents=True, exist_ok=True)
            crop_filename = f"page_{asset.page_index + 1:04d}_elem_{task.index:03d}.jpg"
            crop_path = Path(self.crops_dir) / crop_filename
            img.convert("RGB").save(crop_path, format="JPEG", quality=85)
            image_path = f"crops/{crop_filename}"
        
        # Run VLM
        txt = self.vlm.generate(prompt=prompt, images=[img] if img else None)
        txt = clean_vlm_text(txt)
        
        return TaskResult(
            index=task.index,
            text=txt,
            image_path=image_path,
            metadata={
                "scope": "element",
                "type": task.element.type,
                "provider": provider_name,
                "prompt": prompt,
                "bbox": [
                    task.element.bbox.x0,
                    task.element.bbox.y0,
                    task.element.bbox.x1,
                    task.element.bbox.y1,
                ],
            },
        )
    
    def execute_sequential(
        self,
        ocr_tasks: list[OCRTask],
        vlm_tasks: list[VLMTask],
        asset: "PageAsset",
        ocr_provider: str = "deepseek-ollama",
        vlm_provider: str = "ollama-vlm",
    ) -> tuple[dict[int, TaskResult], dict[int, TaskResult]]:
        """Execute tasks sequentially (grouped: all OCR first, then VLM)."""
        ocr_results: dict[int, TaskResult] = {}
        vlm_results: dict[int, TaskResult] = {}
        
        # Phase 1: OCR
        for task in ocr_tasks:
            result = self.run_ocr(task, asset, ocr_provider)
            ocr_results[task.index] = result
        
        # Build OCR text map for VLM context
        ocr_text_map = {idx: r.text for idx, r in ocr_results.items()}
        
        # Phase 2: VLM
        for task in vlm_tasks:
            result = self.run_vlm(task, asset, ocr_text_map, vlm_provider)
            vlm_results[task.index] = result
        
        return ocr_results, vlm_results
    
    def execute_parallel(
        self,
        ocr_tasks: list[OCRTask],
        vlm_tasks: list[VLMTask],
        asset: "PageAsset",
        workers: int = 4,
        ocr_provider: str = "deepseek-ollama",
        vlm_provider: str = "ollama-vlm",
    ) -> tuple[dict[int, TaskResult], dict[int, TaskResult]]:
        """Execute tasks in parallel (OCR first, then VLM)."""
        ocr_results: dict[int, TaskResult] = {}
        vlm_results: dict[int, TaskResult] = {}
        
        # Phase 1: OCR in parallel
        if ocr_tasks:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.run_ocr, task, asset, ocr_provider): task.index
                    for task in ocr_tasks
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        ocr_results[result.index] = result
                    except Exception as ex:
                        self._log(f"OCR error: {ex}")
        
        # Build OCR text map for VLM context
        ocr_text_map = {idx: r.text for idx, r in ocr_results.items()}
        
        # Phase 2: VLM in parallel
        if vlm_tasks:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.run_vlm, task, asset, ocr_text_map, vlm_provider): task.index
                    for task in vlm_tasks
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        vlm_results[result.index] = result
                    except Exception as ex:
                        self._log(f"VLM error: {ex}")
        
        return ocr_results, vlm_results
    
    def execute_interleaved(
        self,
        ocr_tasks: list[OCRTask],
        vlm_tasks: list[VLMTask],
        asset: "PageAsset",
        element_count: int,
        ocr_provider: str = "deepseek-ollama",
        vlm_provider: str = "ollama-vlm",
    ) -> tuple[dict[int, TaskResult], dict[int, TaskResult]]:
        """Execute tasks interleaved (process each element completely before moving on)."""
        ocr_results: dict[int, TaskResult] = {}
        vlm_results: dict[int, TaskResult] = {}
        
        ocr_task_map = {t.index: t for t in ocr_tasks}
        vlm_task_map = {t.index: t for t in vlm_tasks}
        
        for idx in range(element_count):
            # OCR first
            if idx in ocr_task_map:
                result = self.run_ocr(ocr_task_map[idx], asset, ocr_provider)
                ocr_results[result.index] = result
            
            # Then VLM
            if idx in vlm_task_map:
                ocr_text_map = {i: r.text for i, r in ocr_results.items()}
                result = self.run_vlm(vlm_task_map[idx], asset, ocr_text_map, vlm_provider)
                vlm_results[result.index] = result
        
        return ocr_results, vlm_results





from __future__ import annotations

import re
from dataclasses import dataclass

from ollamaparser.layout_order import ReadingOrderResult
from ollamaparser.layout_types import DetectedElement
from ollamaparser.types import PageAsset


@dataclass(frozen=True)
class PageQuality:
    page_index: int
    n_elements: int
    n_text: int
    n_table: int
    n_chart: int
    reading_order: dict
    coverage: dict
    warnings: list[str]


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\w\s\uac00-\ud7a3]", "", s)  # keep ko/en/digits/underscore
    return s


def estimate_coverage(*, reference_text: str | None, extracted_text: str | None) -> dict:
    """
    Heuristic coverage score: how much of reference is present in extracted.
    If either side missing, return None-ish scores.
    """
    if not reference_text or not extracted_text:
        return {"available": False, "ratio": None}

    r = _norm(reference_text)
    e = _norm(extracted_text)
    if not r:
        return {"available": False, "ratio": None}
    if not e:
        return {"available": True, "ratio": 0.0}

    rt = set(r.split())
    et = set(e.split())
    if not rt:
        return {"available": False, "ratio": None}
    overlap = len(rt & et) / float(len(rt))
    return {"available": True, "ratio": round(float(overlap), 4)}


def page_quality(
    *,
    page: PageAsset,
    elements: list[DetectedElement],
    reading: ReadingOrderResult | None,
    extracted_text: str | None,
) -> PageQuality:
    n_text = sum(1 for e in elements if e.type == "text")
    n_table = sum(1 for e in elements if e.type == "table")
    n_chart = sum(1 for e in elements if e.type == "chart")

    warnings: list[str] = []
    if reading is None:
        warnings.append("reading order not computed")
        reading_dict = {"n_columns": None, "confidence": None, "notes": ["no reading order"]}
    else:
        if reading.n_columns >= 2 and reading.confidence < 0.7:
            warnings.append("two-column suspected but low confidence; reading order may be wrong")
        if reading.n_columns == 1 and n_text >= 8:
            warnings.append("many text blocks detected; consider checking multi-column order")
        reading_dict = {
            "n_columns": reading.n_columns,
            "confidence": reading.confidence,
            "ordered_indices": reading.ordered_indices,
            "column_assignment": reading.column_assignment,
            "spanning_indices": reading.spanning_indices,
            "notes": reading.notes,
        }

    cov = estimate_coverage(reference_text=page.text, extracted_text=extracted_text or page.text)
    if cov.get("available") and cov.get("ratio") is not None and cov["ratio"] < 0.6:
        warnings.append("low text coverage vs PDF selectable text; extraction may be incomplete")

    return PageQuality(
        page_index=page.page_index,
        n_elements=len(elements),
        n_text=n_text,
        n_table=n_table,
        n_chart=n_chart,
        reading_order=reading_dict,
        coverage=cov,
        warnings=warnings,
    )



"""Detection filtering utilities: IoU-based deduplication and confidence filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollamaparser.layout_types import BBox, LayoutElement


# Minimum confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5
# IoU threshold for considering two boxes as duplicates
IOU_THRESHOLD = 0.8


@dataclass
class FilteredElement:
    """Element after filtering with its original index."""
    
    index: int
    element: "LayoutElement"


def calc_iou(b1: "BBox", b2: "BBox") -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        b1: First bounding box
        b2: Second bounding box
    
    Returns:
        IoU value between 0 and 1
    """
    x0 = max(b1.x0, b2.x0)
    y0 = max(b1.y0, b2.y0)
    x1 = min(b1.x1, b2.x1)
    y1 = min(b1.y1, b2.y1)
    
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area1 = (b1.x1 - b1.x0) * (b1.y1 - b1.y0)
    area2 = (b2.x1 - b2.x0) * (b2.y1 - b2.y0)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def filter_elements(
    elements: list["LayoutElement"],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    skip_labels: tuple[str, ...] = ("abandon",),
) -> list[FilteredElement]:
    """
    Filter detected elements by confidence score and IoU-based deduplication.
    
    This function:
    1. Removes elements below the confidence threshold
    2. Skips elements with labels in skip_labels
    3. Removes overlapping elements, keeping only the highest-scoring one
    
    Args:
        elements: List of detected elements
        confidence_threshold: Minimum confidence score (0-1)
        iou_threshold: IoU threshold for deduplication (0-1)
        skip_labels: Labels to skip (case-insensitive substring match)
    
    Returns:
        List of FilteredElement with original indices preserved
    """
    filtered: list[FilteredElement] = []
    
    for idx, e in enumerate(elements):
        score = getattr(e, "score", None)
        
        # Skip low confidence detections
        if score is not None and score < confidence_threshold:
            continue
        
        # Skip unwanted labels
        label = (getattr(e, "label", None) or "").lower()
        if any(skip in label for skip in skip_labels):
            continue
        
        # Check IoU overlap with already-added elements
        dominated = False
        for fe in filtered:
            iou = calc_iou(e.bbox, fe.element.bbox)
            if iou > iou_threshold:
                existing_score = getattr(fe.element, "score", 0) or 0
                if (score or 0) <= existing_score:
                    dominated = True
                    break
        
        if dominated:
            continue
        
        # Remove existing if this one has higher score and overlaps
        filtered = [
            fe for fe in filtered
            if calc_iou(e.bbox, fe.element.bbox) <= iou_threshold
            or (getattr(fe.element, "score", 0) or 0) >= (score or 0)
        ]
        
        filtered.append(FilteredElement(index=idx, element=e))
    
    return filtered


def find_nearby_context_elements(
    target: "LayoutElement",
    all_elements: list["LayoutElement"],
    max_distance: float = 100,
    target_labels: tuple[str, ...] = ("caption", "title"),
    max_results: int = 3,
) -> list["LayoutElement"]:
    """
    Find caption/title elements near a target element.
    
    Args:
        target: The element to find context for
        all_elements: All elements on the page
        max_distance: Maximum distance threshold in pixels
        target_labels: Labels to look for (case-insensitive substring)
        max_results: Maximum number of results to return
    
    Returns:
        List of nearby elements sorted by distance
    """
    target_cx = (target.bbox.x0 + target.bbox.x1) / 2
    target_cy = (target.bbox.y0 + target.bbox.y1) / 2
    
    nearby: list[tuple[float, "LayoutElement"]] = []
    
    for other in all_elements:
        if other is target:
            continue
            
        label = (getattr(other, "label", "") or "").lower()
        if not any(tl in label for tl in target_labels):
            continue
        
        # Check distance and proximity
        other_cx = (other.bbox.x0 + other.bbox.x1) / 2
        other_cy = (other.bbox.y0 + other.bbox.y1) / 2
        dist = ((target_cx - other_cx) ** 2 + (target_cy - other_cy) ** 2) ** 0.5
        
        # Check if within bounding proximity
        vertical_overlap = not (
            other.bbox.y1 < target.bbox.y0 - max_distance or
            other.bbox.y0 > target.bbox.y1 + max_distance
        )
        horizontal_overlap = not (
            other.bbox.x1 < target.bbox.x0 - max_distance or
            other.bbox.x0 > target.bbox.x1 + max_distance
        )
        
        if vertical_overlap and horizontal_overlap:
            nearby.append((dist, other))
    
    # Sort by distance and return closest
    nearby.sort(key=lambda x: x[0])
    return [elem for _, elem in nearby[:max_results]]





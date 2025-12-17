from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from ollamaparser.layout_types import DetectedElement


@dataclass(frozen=True)
class ReadingOrderResult:
    n_columns: int
    ordered_indices: list[int]  # indices into the input elements list
    column_assignment: list[int]  # per element, 0..n_columns-1 (or -1 for spanning)
    spanning_indices: list[int]
    confidence: float  # 0..1 heuristic
    notes: list[str]


def _is_spanning(e: DetectedElement, page_width: int, *, span_ratio: float = 0.82) -> bool:
    w = max(0.0, e.bbox.x1 - e.bbox.x0)
    return w >= float(page_width) * span_ratio


def infer_reading_order(
    elements: list[DetectedElement],
    *,
    page_width: int,
    page_height: int,
    max_columns: int = 2,
) -> ReadingOrderResult:
    """
    Infer reading order with special handling for two-column documents:
    - Detect wide "spanning" blocks (e.g., title) and place them by y before columns.
    - For text-like blocks, cluster by x-center into columns (KMeans, k=2).
    - Order within each column by (y0, x0).
    - Final order: spanning (top-down) interleaved by y with columns, then left column then right column.
      (Heuristic: spanning blocks are usually headers.)
    """
    notes: list[str] = []
    if not elements:
        return ReadingOrderResult(
            n_columns=1,
            ordered_indices=[],
            column_assignment=[],
            spanning_indices=[],
            confidence=1.0,
            notes=["no elements"],
        )

    # Only use text-like blocks for column inference; keep others but order with same column if possible.
    text_like = [i for i, e in enumerate(elements) if e.type == "text"]
    if len(text_like) < 4 or max_columns < 2:
        ordered = sorted(range(len(elements)), key=lambda i: (elements[i].bbox.y0, elements[i].bbox.x0))
        return ReadingOrderResult(
            n_columns=1,
            ordered_indices=ordered,
            column_assignment=[0] * len(elements),
            spanning_indices=[],
            confidence=0.65 if len(text_like) >= 2 else 0.5,
            notes=["single-column fallback"],
        )

    spanning = [i for i, e in enumerate(elements) if _is_spanning(e, page_width)]
    non_spanning_text = [i for i in text_like if i not in spanning]

    if len(non_spanning_text) < 4:
        ordered = sorted(range(len(elements)), key=lambda i: (elements[i].bbox.y0, elements[i].bbox.x0))
        return ReadingOrderResult(
            n_columns=1,
            ordered_indices=ordered,
            column_assignment=[0] * len(elements),
            spanning_indices=sorted(spanning, key=lambda i: elements[i].bbox.y0),
            confidence=0.6,
            notes=["not enough non-spanning text blocks for column inference"],
        )

    xs = np.array(
        [[(elements[i].bbox.x0 + elements[i].bbox.x1) / 2.0] for i in non_spanning_text],
        dtype=float,
    )

    # KMeans into 2 columns, then check separation.
    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(xs)
    centers = sorted(float(c[0]) for c in kmeans.cluster_centers_)
    sep = abs(centers[1] - centers[0])
    sep_ratio = sep / float(page_width)

    # If separation is too small, treat as single-column.
    if sep_ratio < 0.18:
        ordered = sorted(range(len(elements)), key=lambda i: (elements[i].bbox.y0, elements[i].bbox.x0))
        return ReadingOrderResult(
            n_columns=1,
            ordered_indices=ordered,
            column_assignment=[0] * len(elements),
            spanning_indices=sorted(spanning, key=lambda i: elements[i].bbox.y0),
            confidence=0.55,
            notes=[f"column separation too small (sep_ratio={sep_ratio:.3f})"],
        )

    # Map cluster labels to left/right by centroid x
    centroids = kmeans.cluster_centers_.reshape(-1)
    left_cluster = int(np.argmin(centroids))
    right_cluster = 1 - left_cluster

    col_assign = [-1] * len(elements)
    for idx, lab in zip(non_spanning_text, labels):
        col_assign[idx] = 0 if lab == left_cluster else 1

    # Assign non-text elements to a column by bbox center (unless spanning)
    for i, e in enumerate(elements):
        if i in spanning:
            col_assign[i] = -1
            continue
        if col_assign[i] != -1:
            continue
        cx = (e.bbox.x0 + e.bbox.x1) / 2.0
        col_assign[i] = 0 if abs(cx - centers[0]) <= abs(cx - centers[1]) else 1

    # Order: spanning by y; then col0 by y; then col1 by y.
    span_sorted = sorted(spanning, key=lambda i: (elements[i].bbox.y0, elements[i].bbox.x0))
    col0 = [i for i, c in enumerate(col_assign) if c == 0]
    col1 = [i for i, c in enumerate(col_assign) if c == 1]
    col0_sorted = sorted(col0, key=lambda i: (elements[i].bbox.y0, elements[i].bbox.x0))
    col1_sorted = sorted(col1, key=lambda i: (elements[i].bbox.y0, elements[i].bbox.x0))

    ordered = span_sorted + col0_sorted + col1_sorted

    conf = min(1.0, 0.65 + 0.9 * max(0.0, sep_ratio - 0.18))
    notes.append(f"two-column inferred (sep_ratio={sep_ratio:.3f})")
    return ReadingOrderResult(
        n_columns=2,
        ordered_indices=ordered,
        column_assignment=col_assign,
        spanning_indices=span_sorted,
        confidence=conf,
        notes=notes,
    )



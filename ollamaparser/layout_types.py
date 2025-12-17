from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ElementType = Literal["table", "text", "chart", "image"]


class BBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    def clamp(self, w: int, h: int) -> "BBox":
        return BBox(
            x0=max(0.0, min(float(w), self.x0)),
            y0=max(0.0, min(float(h), self.y0)),
            x1=max(0.0, min(float(w), self.x1)),
            y1=max(0.0, min(float(h), self.y1)),
        )


class DetectedElement(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_index: int
    type: ElementType
    bbox: BBox  # pixel coordinates on rendered page image
    score: float | None = None
    label: str | None = None



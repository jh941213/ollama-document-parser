from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PageAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_index: int
    image_path: str
    width: int
    height: int
    text: str | None = None



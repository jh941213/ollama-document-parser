from __future__ import annotations

import json
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ollamaparser.providers.base import VisionLanguageModel


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    text: str
    score: float


def _chunk_markdown(md: str, *, max_chars: int = 1200) -> list[tuple[str, str]]:
    """
    Simple chunking: split by headings, then size-limit.
    """
    lines = md.splitlines()
    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    cur_id = "chunk_0001"
    idx = 1

    def flush():
        nonlocal idx, cur_id, buf
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            chunks.append((cur_id, text))
            idx += 1
            cur_id = f"chunk_{idx:04d}"
        buf = []

    for ln in lines:
        if ln.startswith("## "):
            flush()
        buf.append(ln)
        if sum(len(x) + 1 for x in buf) >= max_chars:
            flush()

    flush()
    return chunks


def retrieve_chunks(md: str, query: str, *, top_k: int = 8) -> list[EvidenceChunk]:
    chunks = _chunk_markdown(md)
    if not chunks:
        return []

    ids = [cid for cid, _ in chunks]
    texts = [txt for _, txt in chunks]

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts + [query])
    q = X[-1]
    docs = X[:-1]
    sims = cosine_similarity(docs, q).reshape(-1)

    ranked = sorted(range(len(texts)), key=lambda i: float(sims[i]), reverse=True)[:top_k]
    return [EvidenceChunk(chunk_id=ids[i], text=texts[i], score=float(sims[i])) for i in ranked]


def generate_metadata_json(
    *,
    md: str,
    vlm: VisionLanguageModel | None,
    query: str = "이 문서의 메타데이터를 생성해줘.",
) -> dict:
    evidence = retrieve_chunks(md, query, top_k=8)

    base = {
        "query": query,
        "evidence": [e.__dict__ for e in evidence],
    }

    if vlm is None:
        # Fallback: heuristic-only metadata
        base["metadata"] = {
            "title": _guess_title(md),
            "language_hint": _guess_language(md),
            "summary": None,
            "keywords": [],
        }
        base["llm"] = {"used": False, "provider": None}
        return base

    prompt = (
        "다음은 문서에서 추출된 Markdown과, 메타데이터 생성을 위한 근거 chunk입니다.\n"
        "근거 chunk를 우선으로 참고해서, 아래 JSON 스키마로만 출력해줘.\n\n"
        "JSON 스키마:\n"
        "{\n"
        '  "title": string | null,\n'
        '  "summary": string | null,\n'
        '  "keywords": string[],\n'
        '  "entities": { "name": string, "type": string }[],\n'
        '  "tables": { "page": number, "description": string }[],\n'
        '  "charts": { "page": number, "description": string }[]\n'
        "}\n\n"
        "근거 chunk:\n"
        + "\n\n".join([f"[{e.chunk_id} score={e.score:.3f}]\n{e.text}" for e in evidence])
        + "\n\n"
        "Markdown (참고용, 길면 일부만 활용):\n"
        + md[:8000]
    )

    raw = vlm.generate(prompt=prompt, images=None)
    base["llm"] = {"used": True, "provider": vlm.__class__.__name__}
    base["raw_output"] = raw
    base["metadata"] = _safe_json_extract(raw)
    return base


def _safe_json_extract(text: str) -> dict | None:
    # Handle common cases like ```json ... ```
    t = text.strip()
    if t.startswith("```"):
        # strip fences
        t = t.strip("`")
        # remove optional language tag
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("\n", 1)[0] if "\n" in t else t
    # find first {...}
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    cand = t[start : end + 1]
    try:
        return json.loads(cand)
    except Exception:
        return None


def _guess_title(md: str) -> str | None:
    for ln in md.splitlines():
        if ln.startswith("# "):
            return ln[2:].strip() or None
    return None


def _guess_language(md: str) -> str | None:
    sample = md[:2000]
    if any("\uac00" <= ch <= "\ud7a3" for ch in sample):
        return "ko"
    if any("a" <= ch.lower() <= "z" for ch in sample):
        return "en"
    return None



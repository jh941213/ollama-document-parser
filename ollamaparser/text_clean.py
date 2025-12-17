from __future__ import annotations

import re
from html import unescape


_DS_TAG_RE = re.compile(r"<\|/?(ref|det)\|>")  # markers
_KOREAN_RE = re.compile(r"[\uAC00-\uD7A3]")  # 한글 음절


def is_likely_ocr_hallucination(text: str, min_korean_ratio: float = 0.1) -> bool:
    """
    Detect likely OCR hallucination for Korean documents.
    Returns True if text has no Korean characters but has significant Latin text.
    """
    if not text or len(text.strip()) < 3:
        return False
    
    clean = text.strip()
    korean_chars = len(_KOREAN_RE.findall(clean))
    total_alpha = sum(1 for c in clean if c.isalpha())
    
    if total_alpha == 0:
        return False
    
    # If document has mostly Latin chars but no Korean, likely hallucination
    korean_ratio = korean_chars / total_alpha if total_alpha > 0 else 0
    
    # "Category — Value" type hallucinations: all Latin, no Korean
    if korean_chars == 0 and total_alpha > 5:
        return True
    
    return korean_ratio < min_korean_ratio
_DS_DET_BLOCK_RE = re.compile(r"<\|det\|>.*?<\|/det\|>")
_DS_REF_BLOCK_RE = re.compile(r"<\|ref\|>.*?<\|/ref\|>")
_HTML_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>")


def _strip_html(s: str) -> str:
    return unescape(_HTML_TAG_RE.sub("", s))


def _html_table_to_markdown(table_html: str) -> str:
    """
    Very small HTML table -> Markdown converter (no external deps).
    Handles <tr>/<td>/<th> basics. Falls back to stripping tags if parsing fails.
    """
    try:
        rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)
        rows: list[list[str]] = []
        for rh in rows_html:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", rh, flags=re.IGNORECASE | re.DOTALL)
            row = [_strip_html(c).strip() for c in cells]
            if row:
                rows.append(row)
        if not rows:
            return _strip_html(table_html).strip()

        # Normalize column count
        ncol = max(len(r) for r in rows)
        for r in rows:
            r += [""] * (ncol - len(r))

        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        # If header is empty-ish, fabricate generic headers
        if sum(1 for c in header if c.strip()) <= 0:
            header = [f"col{i+1}" for i in range(ncol)]

        def md_row(r: list[str]) -> str:
            return "| " + " | ".join((c or "").replace("\n", " ").strip() for c in r) + " |"

        out = [md_row(header), "| " + " | ".join(["---"] * ncol) + " |"]
        for r in body:
            out.append(md_row(r))
        return "\n".join(out).strip()
    except Exception:
        return _strip_html(table_html).strip()


def _convert_html_tables(s: str) -> str:
    # Replace each <table>...</table> block with markdown table
    def repl(m: re.Match) -> str:
        return _html_table_to_markdown(m.group(0))

    return re.sub(r"<table[^>]*>.*?</table>", repl, s, flags=re.IGNORECASE | re.DOTALL)


def clean_deepseek_ocr_text(text: str) -> str:
    """
    Best-effort cleanup for DeepSeek-OCR outputs when using markdown/grounding prompts.
    Removes <|ref|> / <|det|> blocks and collapses noisy headings/blank lines.
    """
    if not text:
        return ""

    s = text
    s = _convert_html_tables(s)
    # Remove grounding tag blocks (can appear inline)
    s = _DS_DET_BLOCK_RE.sub("", s)
    s = _DS_REF_BLOCK_RE.sub("", s)
    s = _DS_TAG_RE.sub("", s)
    s = _strip_html(s)

    lines = []
    last = None
    for raw in s.splitlines():
        line = raw.rstrip()
        # Drop lines that are only heading markers or only punctuation
        if re.fullmatch(r"\s*#+\s*", line):
            continue
        if re.fullmatch(r"\s*[-–—=*_]{3,}\s*", line):
            continue
        # Drop obvious repetition/junk lines (common failure mode)
        if len(line) > 200 and (line.count("텍스트") > 10 or (len(set(line)) / max(1, len(line))) < 0.15):
            continue
        if "</" in line or "<" in line:
            # leftover broken tags
            line = _strip_html(line)
            if not line.strip():
                continue
        # Collapse duplicate consecutive lines
        if last is not None and line.strip() and line.strip() == last.strip():
            continue
        lines.append(line)
        last = line

    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()

    # If we produced a tiny 2-col table (common for titles), flatten it.
    if out.startswith("|") and "\n| ---" in out:
        md_lines = out.splitlines()
        if len(md_lines) >= 2 and md_lines[0].count("|") >= 3:
            header_cells = [c.strip() for c in md_lines[0].strip("|").split("|")]
            if len(header_cells) == 2 and all(header_cells):
                out = f"{header_cells[0]} — {header_cells[1]}"
    return out


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def clean_vlm_text(text: str) -> str:
    """
    Cleanup for VLM outputs:
    - Remove <think>...</think> tags (qwen3-vl thinking mode)
    """
    if not text:
        return ""

    s = text
    # Remove thinking tags entirely
    s = _THINK_TAG_RE.sub("", s)

    return s.strip()


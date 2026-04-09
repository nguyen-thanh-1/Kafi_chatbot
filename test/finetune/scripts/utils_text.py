from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    text: str


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def iter_files(input_dir: str) -> Iterator[Path]:
    root = Path(input_dir)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".txt", ".md", ".pdf"}:
            yield path


def read_text_file(path: Path) -> str:
    # Try utf-8 first, fall back to cp1258/latin1 for Vietnamese-ish files.
    for enc in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    # Last resort: bytes decode ignoring errors.
    return path.read_bytes().decode("utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: pymupdf (fitz). Install it to parse PDF.") from e

    doc = fitz.open(path)
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)


def read_document(
    path: Path,
) -> Document:
    if path.suffix.lower() == ".pdf":
        text = read_pdf(path)
    else:
        text = read_text_file(path)
    text = normalize_text(text)
    doc_id = _sha1(str(path.resolve()) + "\n" + text)[:16]
    return Document(doc_id=doc_id, source_path=str(path.resolve()), text=text)


def normalize_text(text: str) -> str:
    # Lightweight normalization; keep Vietnamese diacritics intact.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive nulls
    text = text.replace("\x00", "")
    # Collapse trailing spaces
    lines = [ln.strip() for ln in text.split("\n")]
    # Keep blank lines but limit runs
    out: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 2:
                out.append("")
            continue
        blank_run = 0
        out.append(ln)
    return "\n".join(out).strip()


def chunk_by_tokens(
    text: str,
    *,
    tokenizer_name_or_path: Optional[str],
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Chunk text by tokenizer token count when possible.
    Falls back to a rough char-based approximation if tokenizer can't be loaded.
    """
    if chunk_tokens <= 0:
        return [text]
    if overlap_tokens < 0:
        overlap_tokens = 0

    tokenizer = None
    if tokenizer_name_or_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        except Exception:
            tokenizer = None

    if tokenizer is None:
        # Approx: 4 chars ~= 1 token (very rough)
        approx = max(1, chunk_tokens * 4)
        step = max(1, approx - overlap_tokens * 4)
        chunks = [text[i : i + approx] for i in range(0, len(text), step)]
        return [c.strip() for c in chunks if c.strip()]

    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_tokens - overlap_tokens)
    while start < len(ids):
        window = ids[start : start + chunk_tokens]
        chunk = tokenizer.decode(window, skip_special_tokens=True)
        if chunk.strip():
            chunks.append(chunk.strip())
        start += step
    return chunks

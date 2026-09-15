from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from .schemas import Chunk
from .config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdf(path: Path) -> List[Chunk]:
    reader = PdfReader(str(path))
    chunks: List[Chunk] = []
    for page_no, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunks.append(Chunk(text=text[start:end], metadata={"filename": path.name, "page": page_no}))
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks

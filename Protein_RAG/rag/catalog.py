from __future__ import annotations

import json
from pathlib import Path

from .config import CATALOG_PATH


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"RAG catalog not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

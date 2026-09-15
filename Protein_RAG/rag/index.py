from __future__ import annotations

import joblib
import chromadb
from rank_bm25 import BM25Okapi

from .config import *
from .catalog import load_catalog
from .document_loader import load_pdf


def build_index(include_optional: bool = False) -> None:
    corpus = []
    for item in load_catalog().get("documents", []):
        if item.get("optional") and not include_optional:
            continue
        path = CORPUS_DIR / item["relative_path"]
        for chunk in load_pdf(path):
            chunk.metadata.update(item)
            corpus.append(chunk)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    collection.add(
        ids=[str(i) for i in range(len(corpus))],
        documents=[c.text for c in corpus],
        metadatas=[c.metadata for c in corpus],
    )
    tokenized = [c.text.lower().split() for c in corpus]
    joblib.dump({"bm25": BM25Okapi(tokenized), "chunks": corpus}, BM25_PATH)
    print(f"Indexed {len(corpus)} chunks from {len({c.metadata.get('filename') for c in corpus})} documents.")

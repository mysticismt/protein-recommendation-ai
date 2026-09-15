from __future__ import annotations

import joblib
import chromadb
from .config import *
from .query_router import analyze_query


class HybridRetriever:
    def __init__(self) -> None:
        if not CHROMA_DIR.exists() or not BM25_PATH.exists():
            raise FileNotFoundError("RAG index is missing. Run python -m Protein_RAG.scripts.build_rag_index first.")
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(CHROMA_COLLECTION)
        data = joblib.load(BM25_PATH)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
        intent = analyze_query(query)
        dense = self.collection.query(query_texts=[query], n_results=max(DENSE_TOP_K, top_k))
        candidates: dict[int, dict] = {}
        for text, meta, dist in zip(dense.get("documents", [[]])[0], dense.get("metadatas", [[]])[0], dense.get("distances", [[]])[0]):
            candidates.setdefault(meta.get("_index", len(candidates)), {"text": text, "metadata": meta, "dense_score": 1.0 - float(dist)})
        scores = self.bm25.get_scores(query.lower().split())
        for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:LEXICAL_TOP_K]:
            item = candidates.setdefault(idx, {"text": self.chunks[idx].text, "metadata": self.chunks[idx].metadata, "dense_score": 0.0})
            item["lexical_score"] = float(scores[idx])
        ranked = sorted(candidates.values(), key=lambda x: x.get("dense_score", 0) + 0.05 * x.get("lexical_score", 0), reverse=True)
        out = []
        seen_sources = set()
        for item in ranked:
            meta = item["metadata"]
            source = meta.get("filename")
            if source in seen_sources:
                continue
            seen_sources.add(source)
            item["rerank_score"] = item.get("dense_score", 0) + 0.05 * item.get("lexical_score", 0)
            item["matched_topics"] = intent.topics
            item["safety_restricted"] = intent.safety_restricted
            out.append(item)
            if len(out) >= top_k:
                break
        return out

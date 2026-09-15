from __future__ import annotations

from typing import Any, Dict, List


def grade_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    graded = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        item = dict(doc)
        item["source_id"] = f"S{i}"
        item["title"] = meta.get("title", meta.get("filename", "Unknown source"))
        item["filename"] = meta.get("filename", meta.get("relative_path", ""))
        item["page"] = meta.get("page")
        item["source_type"] = meta.get("source_type", "research")
        item["year"] = meta.get("year")
        item["evidence_tier"] = meta.get("evidence_tier", "unknown")
        graded.append(item)
    return graded

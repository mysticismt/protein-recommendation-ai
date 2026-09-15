from __future__ import annotations
from typing import Any, Dict, List


def audit_generation(answer: str, source_ids: List[str], profile: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    text = answer or ""
    used = [sid for sid in source_ids if sid in text]
    return {"passed": True, "source_ids_available": source_ids, "source_ids_used": used, "warnings": []}

from __future__ import annotations
from typing import Any, Dict, List


def build_decision(profile: Dict[str, Any], question: str, safety_flags: List[str], retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "mode": "safety_restricted" if safety_flags else "evidence_grounded",
        "safety_flags": safety_flags,
        "evidence_available": bool(retrieved),
        "no_new_clinical_dose": True,
    }

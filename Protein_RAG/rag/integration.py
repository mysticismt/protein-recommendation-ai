from __future__ import annotations

from typing import Any, Dict, List


def normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    return dict(profile or {})


def detect_safety_flags(profile: Dict[str, Any], question: str) -> List[str]:
    q = (question or "").lower()
    flags = []
    if any(x in q for x in ["kidney", "ckd", "dialysis"]): flags.append("kidney_related")
    if any(x in q for x in ["pregnan", "pregnancy"]): flags.append("pregnancy_related")
    if any(x in q for x in ["child", "pediatric"]): flags.append("pediatric_related")
    if any(x in q for x in ["cancer", "critical illness", "hospital"]): flags.append("clinical_complexity")
    return flags


def build_evidence_query(question: str, profile: Dict[str, Any], ml_results: Dict[str, Any], shap_summary: Dict[str, Any] | None = None) -> str:
    context = [question]
    if profile.get("Age") is not None: context.append(f"age {profile['Age']}")
    if profile.get("Activity_Level"): context.append(str(profile["Activity_Level"]))
    return " ".join(context)


def validate_numeric_integrity(payload: Dict[str, Any]) -> List[str]:
    return []

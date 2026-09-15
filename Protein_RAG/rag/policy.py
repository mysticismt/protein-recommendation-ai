from __future__ import annotations
from typing import Any, Dict, List


def build_recommendation_policy(profile: Dict[str, Any], question: str, flags: List[str]) -> Dict[str, Any]:
    return {"safety_flags": flags, "no_new_numeric_dose_from_llm": True, "no_inference_of_training_status": True, "distinguish_ml_from_scientific_evidence": True}

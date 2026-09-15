from __future__ import annotations
from typing import Any, Dict, List


def build_safety_profile(profile: Dict[str, Any], question: str, flags: List[str]) -> Dict[str, Any]:
    return {"flags": flags, "restricted": bool(flags), "message": "Clinical assessment is required when safety-related flags are present." if flags else "No safety flag detected from the supplied question/profile."}

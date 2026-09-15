from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .config import OLLAMA_MODEL
from .integration import build_evidence_query, detect_safety_flags, normalize_profile
from .retriever import HybridRetriever
from .evidence import grade_documents
from .safety import build_safety_profile
from .policy import build_recommendation_policy
from Protein_RAG.llm.provider import build_llm_provider
from .validation import audit_generation

SYSTEM_PROMPT = """You are the evidence-grounded reasoning layer of a personalized protein recommendation research prototype.

Rules: legacy numeric facts are authoritative; never invent or recalculate case-specific values; SHAP is model-behaviour explanation, not causality; do not infer resistance training; retrieved studies describe populations, not this individual; do not create a new clinical dose for safety-restricted contexts; distinguish scientific evidence from the legacy ML model; use supplied source IDs exactly; the dataset target is engineered, not independent clinical ground truth.

Return ONLY these three sections: Evidence, Protein Source, Safety / Limitations. Do NOT include an Estimate section; Python adds it deterministically.
"""


def _evidence_block(docs: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{x['source_id']}] title={x['title']} | file={x['filename']} | page={x['page']} | source_type={x['source_type']} | year={x['year']} | evidence_tier={x['evidence_tier']}\n{x['text']}"
        for x in grade_documents(docs)
    )


def _safe_num(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_legacy_facts(profile: Dict[str, Any], ml_results: Dict[str, Any], shap_summary: Optional[Dict[str, float]]) -> Dict[str, Any]:
    return {
        "model": ml_results.get("model"),
        "daily_protein_requirement_g": _safe_num(ml_results.get("Daily_Protein_Requirement_g")),
        "daily_protein_requirement_g_per_kg": _safe_num(ml_results.get("Daily_Protein_Requirement_g_per_kg")),
        "daily_protein_intake_g": _safe_num(ml_results.get("Daily_Protein_Intake_g", profile.get("Daily_Protein_Intake_g"))),
        "daily_protein_intake_g_per_kg": _safe_num(ml_results.get("Daily_Protein_Intake_g_per_kg")),
        "intake_gap_g": _safe_num(ml_results.get("Intake_Gap_g")),
        "engineered_dataset_target_g": _safe_num(ml_results.get("dataset_target_Protein_Requirement_g")),
        "supplement_classifier_output": ml_results.get("Recommended_Supplement"),
        "supplement_classifier_probabilities": ml_results.get("Supplement_Probabilities", {}),
        "shap_top_drivers": shap_summary or {},
    }


def _render_estimate(facts: Dict[str, Any]) -> str:
    lines = ["**Estimate**", "The following values come directly from the existing legacy project and are not clinical ground truth:"]
    labels = [("daily_protein_requirement_g", "Legacy Random Forest protein requirement estimate", "g/day"), ("daily_protein_requirement_g_per_kg", "Legacy estimate normalized by body weight", "g/kg/day"), ("daily_protein_intake_g", "Current protein intake in the project dataset", "g/day"), ("daily_protein_intake_g_per_kg", "Current intake normalized by body weight", "g/kg/day"), ("intake_gap_g", "Deterministic intake gap", "g/day")]
    for key, label, unit in labels:
        if facts.get(key) is not None:
            lines.append(f"* {label}: **{float(facts[key]):.4f} {unit}**.")
    if facts.get("supplement_classifier_output"):
        lines.append(f"* Legacy supplement classifier output: **{facts['supplement_classifier_output']}**. This is a model output, not a clinical recommendation.")
    lines.append("* The dataset target is an engineered NHANES-derived value, not independently observed clinical ground truth.")
    return "\n".join(lines)


class GroundedNutritionAgent:
    def __init__(self, model_name: str = OLLAMA_MODEL, retriever: Optional[HybridRetriever] = None):
        self.model_name = model_name
        self.retriever = retriever or HybridRetriever()
        self.llm = build_llm_provider()

    def answer(self, user_profile: Dict[str, Any], ml_results: Dict[str, Any], shap_summary: Optional[Dict[str, float]] = None, question: str = "") -> Dict[str, Any]:
        profile = normalize_profile(user_profile)
        flags = detect_safety_flags(profile, question)
        safety_profile = build_safety_profile(profile, question, flags)
        policy = build_recommendation_policy(profile, question, safety_profile["flags"])
        retrieval_query = build_evidence_query(question, profile, ml_results, shap_summary)
        docs = self.retriever.retrieve(retrieval_query)
        facts = build_legacy_facts(profile, ml_results, shap_summary)
        prompt = f"USER PROFILE\n{json.dumps(profile, ensure_ascii=False, indent=2, default=str)}\n\nLEGACY FACTS\n{json.dumps(facts, ensure_ascii=False, indent=2, default=str)}\n\nSAFETY FLAGS\n{json.dumps(safety_profile['flags'], ensure_ascii=False)}\n\nUSER QUESTION\n{question}\n\nRETRIEVED EVIDENCE\n{_evidence_block(docs)}\n\nRECOMMENDATION POLICY\n{json.dumps(policy, ensure_ascii=False, indent=2)}"
        content = self.llm.chat(self.model_name, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        graded = grade_documents(docs)
        audit = audit_generation(content, [str(x["source_id"]).upper() for x in graded], profile, policy)
        return {"answer": _render_estimate(facts) + "\n\n" + content, "legacy_facts": facts, "retrieved": docs, "retrieval_query": retrieval_query, "safety_flags": safety_profile["flags"], "safety_profile": safety_profile, "recommendation_policy": policy, "generation_audit": audit}

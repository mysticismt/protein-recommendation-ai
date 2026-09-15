from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from Protein_RAG.integration.legacy_adapter import LegacyProjectAdapter
from Protein_RAG.rag.answer import GroundedNutritionAgent
from Protein_RAG.rag.integration import build_evidence_query, detect_safety_flags, validate_numeric_integrity
from Protein_RAG.recommendation.decision import build_decision


class ProteinRecommendationPipeline:
    """Single entry point around the existing legacy ML + SHAP + RAG stack."""

    def __init__(self) -> None:
        self.legacy = LegacyProjectAdapter()
        self.agent = GroundedNutritionAgent()

    def run(self, row_id: float, question: str, shap_top_k: int = 10, generate: bool = True) -> Dict[str, Any]:
        case = self.legacy.build_real_case(row_id=row_id, shap_top_k=shap_top_k)
        profile, ml_results, shap_summary = case["profile"], case["ml_results"], case["shap_summary"]
        safety_flags = detect_safety_flags(profile, question)
        retrieval_query = build_evidence_query(question, profile, ml_results, shap_summary)
        retrieved = self.agent.retriever.retrieve(retrieval_query)
        decision = build_decision(profile, question, safety_flags, retrieved)
        generated = self.agent.answer(question=question, user_profile=profile, ml_results=ml_results, shap_summary=shap_summary) if generate else None
        warnings = validate_numeric_integrity({"profile": profile, "legacy_ml": ml_results})
        contract = {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "case": profile,
            "legacy_ml": ml_results,
            "shap": shap_summary,
            "retrieval": {"query": retrieval_query, "sources": generated.get("retrieved", []) if generated else retrieved},
            "decision": decision,
            "answer": generated.get("answer", "") if generated else "Generation skipped (generate=false). The deterministic ML/RAG contract is still returned.",
            "audits": {"numeric_integrity": "PASS" if not warnings else {"passed": False, "warnings": warnings}, "generation_audit": generated.get("generation_audit") if generated else None},
        }
        return contract

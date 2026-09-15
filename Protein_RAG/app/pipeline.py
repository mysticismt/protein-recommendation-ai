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

    def run(
        self,
        row_id: float,
        question: str,
        shap_top_k: int = 10,
        generate: bool = False,
    ) -> Dict[str, Any]:
        case = self.legacy.build_real_case(
            row_id=row_id,
            shap_top_k=shap_top_k,
        )

        profile = case["profile"]
        ml_results = case["ml_results"]
        shap_summary = case["shap_summary"]

        safety_flags = detect_safety_flags(profile, question)

        retrieval_query = build_evidence_query(
            question,
            profile,
            ml_results,
            shap_summary,
        )

        retrieved = self.agent.retriever.retrieve(retrieval_query)

        decision = build_decision(
            profile,
            question,
            safety_flags,
            retrieved,
        )

        # LLM generation is optional. The default path is fully usable without
        # Ollama, Qwen, or any other model provider.
        generated = None
        if generate:
            generated = self.agent.answer(
                question=question,
                user_profile=profile,
                ml_results=ml_results,
                shap_summary=shap_summary,
            )

        validation_input = {
            "profile": profile,
            "legacy_ml": ml_results,
        }
        numeric_warnings = validate_numeric_integrity(validation_input)
        numeric_integrity = "PASS" if not numeric_warnings else {
            "passed": False,
            "warnings": numeric_warnings,
        }

        contract = {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "case": profile,
            "legacy_ml": ml_results,
            "shap": shap_summary,
            "retrieval": {
                "query": retrieval_query,
                "sources": generated.get("retrieved", []) if generated else retrieved,
            },
            "decision": decision,
            "answer": generated.get("answer", "") if generated else "Generation skipped (generate=false). The deterministic ML/RAG contract is returned without an LLM.",
            "audits": {
                "numeric_integrity": numeric_integrity,
                "generation_audit": generated.get("generation_audit") if generated else None,
            },
        }

        return contract

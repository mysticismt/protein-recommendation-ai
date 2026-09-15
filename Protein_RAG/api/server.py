from __future__ import annotations
from functools import lru_cache
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Protein Recommendation AI", version="1.0", description="API wrapper around the existing Legacy ML + SHAP + RAG pipeline.")

class RecommendationRequest(BaseModel):
    row_id: float = Field(..., description="NHANES/project row ID")
    question: str = Field(..., min_length=3)
    shap_top_k: int = Field(10, ge=1, le=25)
    generate: bool = Field(True, description="Run LLM generation; false skips any local/hosted LLM.")

@lru_cache(maxsize=1)
def get_pipeline() -> Any:
    from Protein_RAG.app.pipeline import ProteinRecommendationPipeline
    return ProteinRecommendationPipeline()

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "protein-recommendation-ai"}

@app.get("/info")
def info() -> dict[str, Any]:
    return {"service": "protein-recommendation-ai", "version": "1.0", "pipeline": "legacy_ml -> shap -> rag -> grounded_generation", "legacy_components_modified": False}

@app.post("/recommend")
def recommend(request: RecommendationRequest) -> dict[str, Any]:
    try:
        return get_pipeline().run(row_id=request.row_id, question=request.question, shap_top_k=request.shap_top_k, generate=request.generate)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

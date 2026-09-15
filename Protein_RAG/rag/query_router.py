from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryIntent:
    topics: list[str]
    safety_restricted: bool = False


def analyze_query(query: str) -> QueryIntent:
    q = (query or "").lower()
    topics: list[str] = []
    mapping = {
        "timing": ["timing", "post workout", "after workout", "pre workout", "frequency"],
        "older_adult": ["older adult", "elderly", "aging", "aged", "senior"],
        "exercise": ["exercise", "resistance", "strength", "training", "muscle"],
        "protein_source": ["whey", "casein", "plant", "animal", "soy", "source"],
        "weight_loss": ["weight loss", "fat loss", "obesity", "body composition"],
        "kidney": ["kidney", "ckd", "chronic kidney", "dialysis"],
    }
    for topic, terms in mapping.items():
        if any(re.search(rf"\b{re.escape(t)}\b", q) if " " not in t else t in q for t in terms):
            topics.append(topic)
    restricted = any(t in q for t in ["ckd", "chronic kidney", "dialysis", "pregnan", "child", "pediatric", "cancer", "critical illness"])
    return QueryIntent(topics=topics, safety_restricted=restricted)

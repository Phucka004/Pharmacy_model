from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from underthesea import word_tokenize
from unidecode import unidecode

# Heuristic rules shared by the Baseline and Knowledge Graph predictors.
HARD_CATEGORY_RULES: Dict[str, Dict[str, List[str]]] = {
    "alzheimer": {
        "keywords": ["suy giam tri nho", "sa sut tri tue", "mat tri nho", "alzheimer"],
        "allowed_groups": ["*"],
        "forbidden_groups": [],
    },
    "ankhongtieu": {
        "keywords": ["an khong tieu", "kho tieu", "day bung", "day hoi", "oi chua", "nang bung"],
        "allowed_groups": ["*"],
        "forbidden_groups": [],
    },
    "lao": {
        "keywords": ["lao phoi", "lao", "ho keo dai", "sot nhe"],
        "allowed_groups": ["*"],
        "forbidden_groups": [],
    },
    "benh lao phoi": {
        "keywords": ["lao phoi", "lao", "ho keo dai"],
        "allowed_groups": ["*"],
        "forbidden_groups": [],
    },
    "cuong giap": {
        "keywords": ["cuong giap", "buou giap lan toa"],
        "allowed_groups": ["*"],
        "forbidden_groups": [],
    },
    "dai thao duong": {
        "keywords": ["tieu duong", "dai thao duong", "duong huyet"],
        "allowed_groups": ["*"],
        "forbidden_groups": [],
    },
    "benh gout": {
        "keywords": ["gout", "gut", "acid uric"],
        "allowed_groups": ["*"],
        "forbidden_groups": [],
    },
}


@dataclass
class SymptomPrediction:
    category: str
    display_name: str
    top1_score: float
    top2_score: float


class DynamicSymptomEngine:
    def __init__(self, qa_payload: Dict[str, Any] | None = None):
        self.qa_payload = qa_payload or {}
        self.category_map = {
            self._normalize_text(category): category for category in self.qa_payload.keys()
        }
        self.display_map = {
            self._normalize_text(str(item.get("display_name", category))): category
            for category, item in self.qa_payload.items()
        }
        self.symptom_index: Dict[str, List[str]] = {}
        for category, item in self.qa_payload.items():
            symptoms = []
            for question in item.get("questions", []):
                clean = self._normalize_text(question)
                if clean:
                    symptoms.append(clean)
            self.symptom_index[category] = symptoms

    @staticmethod
    def _normalize_text(text: Any) -> str:
        normalized = unidecode(str(text)).lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _score_category(self, query: str, category: str) -> float:
        item = self.qa_payload.get(category, {})
        display_name = self._normalize_text(str(item.get("display_name", category)))
        candidates = [display_name, self._normalize_text(category)]
        candidates.extend(self.symptom_index.get(category, []))
        score = 0.0
        for cand in candidates:
            if cand and cand in query:
                score += 2.0 if cand in (display_name, self._normalize_text(category)) else 1.0
        for kw in HARD_CATEGORY_RULES.get(self._normalize_text(category), {}).get("keywords", []):
            if kw and kw in query:
                score += 1.5
        return score

    def predict(self, query: str) -> Tuple[str, str, float, float]:
        clean_query = self._normalize_text(query)
        if not clean_query:
            return "", "", 0.0, 0.0

        if self.qa_payload:
            scored = []
            for category in self.qa_payload.keys():
                scored.append((self._score_category(clean_query, category), category))
            scored.sort(reverse=True)
            top1_score = scored[0][0] if scored else 0.0
            top2_score = scored[1][0] if len(scored) > 1 else 0.0
            if top1_score > 0:
                best_category = scored[0][1]
                display_name = str(self.qa_payload[best_category].get("display_name", best_category))
                return best_category, display_name, top1_score, top2_score

        # Fallback keyword extraction for when the QA payload is unavailable.
        tokens = [tok for tok in word_tokenize(clean_query, format="text").split() if tok]
        candidates = list(dict.fromkeys(tokens + [" ".join(tokens[:2]), " ".join(tokens[-2:])]))
        for category in self.category_map.values():
            cat_norm = self._normalize_text(category)
            if any(cand and cand in cat_norm for cand in candidates):
                return category, category, 1.0, 0.0
        return "", "", 0.0, 0.0

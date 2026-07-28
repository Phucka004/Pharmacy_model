from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from underthesea import word_tokenize
from unidecode import unidecode

from src.dynamic_diagnosis import DynamicSymptomEngine, HARD_CATEGORY_RULES


@dataclass
class PredictionResult:
    category: str
    display_name: str
    confidence_text: str | None = None
    products: List[Dict[str, Any]] | None = None


class MedicalMLPredictor:
    def __init__(
        self,
        model_path: str = "models/medical_classifier.pkl",
        qa_path: str = "data/silver/synthetic_medical_qa.json",
        products_path: str = "data/silver/products_kb.csv",
    ):
        self.model_path = Path(model_path)
        self.qa_path = Path(qa_path)
        self.products_path = Path(products_path)
        self.model = self._load_model()
        self.category_map = self._load_category_map()
        self.products_df = self._load_products_df()
        with self.qa_path.open("r", encoding="utf-8") as f:
            self.qa_payload = json.load(f)
        self.HARD_CATEGORY_RULES = HARD_CATEGORY_RULES
        self.symptom_engine = DynamicSymptomEngine(self.qa_payload)
        self.search_cols = [
            c
            for c in [
                "product_name",
                "description",
                "usage",
                "dosage",
                "side_effects",
                "precautions",
                "category",
                "display_name",
                "text",
                "ingredients",
                "ingredients_raw",
            ]
            if c in self.products_df.columns
        ]
        self.products_df["_search_text"] = self.products_df.apply(self._build_product_search_text, axis=1)

    def _load_model(self):
        with self.model_path.open("rb") as f:
            return pickle.load(f)

    def _load_category_map(self) -> Dict[str, str]:
        with self.qa_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return {k: v.get("display_name", k) for k, v in payload.items()}

    def _load_products_df(self) -> pd.DataFrame:
        return pd.read_csv(self.products_path)

    def _normalize_text(self, text: str) -> str:
        normalized = unidecode(str(text)).lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _tokenize_keywords(self, text: str) -> List[str]:
        text = self._normalize_text(text)
        tokens = [tok for tok in text.split() if tok]
        phrases = [text]
        if len(tokens) >= 2:
            phrases.append(" ".join(tokens[:2]))
            phrases.append(" ".join(tokens[-2:]))
        if len(tokens) >= 3:
            phrases.append(" ".join(tokens[:3]))
        return list(dict.fromkeys([p for p in phrases if p]))

    def _build_keywords(self, display_name: str, category: str) -> List[str]:
        base_terms = {
            "alzheimer": ["suy giảm trí nhớ", "sa sút trí tuệ", "mất trí nhớ", "suy giảm nhớ"],
            "ankhongtieu": ["an khong tieu", "kho tieu", "day bung", "day hoi", "oi chua", "nang bung"],
            "lao": ["lao phoi", "lao", "ho keo dai", "sot nhe"],
            "benh lao phoi": ["lao phoi", "lao", "ho keo dai"],
            "cuong giap": ["cuong giap", "buou giap lan toa"],
            "dai thao duong": ["tieu duong", "dai thao duong", "duong huyet"],
            "benh gout": ["gout", "gut", "acid uric"],
        }

        normalized_display = self._normalize_text(display_name)
        normalized_category = self._normalize_text(category)
        kws = []
        kws.extend(self._tokenize_keywords(display_name))
        kws.extend(self._tokenize_keywords(category))
        kws.extend(base_terms.get(normalized_display, []))
        kws.extend(base_terms.get(normalized_category, []))

        compact = normalized_category.replace(" ", "")
        if compact.startswith("benh"):
            compact = compact[4:]
        if compact:
            kws.append(compact)
            if compact.startswith("laophoi"):
                kws.extend(["lao phoi", "lao"])
            if compact.startswith("ankhongtieu"):
                kws.extend(["an khong tieu", "kho tieu"])

        return list(dict.fromkeys(self._normalize_text(kw) for kw in kws if kw))

    def _build_product_search_text(self, row: pd.Series) -> str:
        text = " ".join(str(row.get(col, "")) for col in self.search_cols)
        return self._normalize_text(text)

    def _product_group(self, row: pd.Series) -> str:
        raw = " ".join(
            str(row.get(col, ""))
            for col in ("category", "display_name", "source_group", "usage", "description")
        )
        return self._normalize_text(raw).upper()

    def predict_category(self, query: str) -> str:
        category, _, _, _ = self.symptom_engine.predict(query)
        if category:
            return category
        processed = word_tokenize(query, format="text")
        return self.model.predict([processed])[0]

    def lookup_products(self, category: str, top_k: int = 5, query: str = "") -> List[Dict[str, Any]]:
        display_name = self.category_map.get(category, category)
        keywords = self._build_keywords(display_name, category)
        query_clean = self._normalize_text(query)
        allowed_groups = [self._normalize_text(g).upper() for g in self.HARD_CATEGORY_RULES.get(category, {}).get("allowed_groups", [])]
        forbidden_groups = [self._normalize_text(g).upper() for g in self.HARD_CATEGORY_RULES.get(category, {}).get("forbidden_groups", [])]

        matches = []
        for _, row in self.products_df.iterrows():
            group = self._product_group(row)
            if allowed_groups and not any(g == "*" or g in group for g in allowed_groups):
                continue
            if any(g != "*" and g in group for g in forbidden_groups):
                continue
            haystack = row.get("_search_text", "")
            score = sum(1 for kw in keywords if kw and kw in haystack)
            if query_clean and query_clean in haystack:
                score += 1
            if score > 0:
                matches.append(
                    {
                        "product_name": row.get("product_name"),
                        "display_name": row.get("display_name"),
                        "category": row.get("category"),
                        "usage": row.get("usage"),
                        "score": score,
                    }
                )

        matches.sort(key=lambda x: (-x["score"], x["product_name"] or ""))
        return matches[:top_k]

    def predict(self, query: str, top_k: int = 5) -> PredictionResult:
        category = self.predict_category(query)
        return PredictionResult(
            category=category,
            display_name=self.category_map.get(category, category),
            products=self.lookup_products(category, top_k=top_k, query=query),
        )

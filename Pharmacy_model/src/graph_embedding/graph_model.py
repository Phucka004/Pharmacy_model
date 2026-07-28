from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import pandas as pd
from unidecode import unidecode

from src.dynamic_diagnosis import HARD_CATEGORY_RULES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QA_PATH = PROJECT_ROOT / "data" / "silver" / "synthetic_medical_qa.json"
PRODUCTS_PATH = PROJECT_ROOT / "data" / "silver" / "products_kb.csv"
TOP_K = 4


@dataclass
class GraphPredictionResult:
    disease: str
    disease_display_name: str
    products: List[Dict[str, Any]]
    latency_seconds: float


class KnowledgeGraphMedicinePredictor:
    def __init__(self, qa_path: str | Path = QA_PATH, products_path: str | Path = PRODUCTS_PATH, products_df: pd.DataFrame | None = None):
        self.qa_path = Path(qa_path)
        self.products_path = Path(products_path)
        self.graph = nx.Graph()
        self.qa_payload = self._load_qa_payload()
        self.products_df = products_df.copy() if isinstance(products_df, pd.DataFrame) else self._load_products_df()
        self.products_df = self._prepare_products_df(self.products_df)
        self.disease_aliases = self._build_disease_aliases()
        self.HARD_CATEGORY_RULES = HARD_CATEGORY_RULES
        self._disease_symptom_texts: Dict[str, List[str]] = {}
        self._build_graph()

    @staticmethod
    def _load_qa_payload_path(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_qa_payload(self) -> Dict[str, Any]:
        if not self.qa_path.exists():
            raise FileNotFoundError(f"Missing QA dataset: {self.qa_path}")
        return self._load_qa_payload_path(self.qa_path)

    def _load_products_df(self) -> pd.DataFrame:
        if not self.products_path.exists():
            raise FileNotFoundError(f"Missing products file: {self.products_path}")
        return pd.read_csv(self.products_path)

    @staticmethod
    def to_ascii_lower(text: Any) -> str:
        normalized = unidecode(str(text)).lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def remove_vietnamese_accents(text: Any) -> str:
        return KnowledgeGraphMedicinePredictor.to_ascii_lower(text)

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return KnowledgeGraphMedicinePredictor.to_ascii_lower(text)

    def _prepare_products_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required_cols = [
            "product_name",
            "sale_price",
            "usage",
            "description",
            "dosage",
            "side_effects",
            "is_prescription",
            "is_prescription_label",
            "category",
            "display_name",
            "source_group",
            "text",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""
        df = df.fillna("")
        df["_search_clean"] = (
            df["product_name"].astype(str)
            + " "
            + df["usage"].astype(str)
            + " "
            + df["description"].astype(str)
            + " "
            + df["category"].astype(str)
            + " "
            + df["display_name"].astype(str)
            + " "
            + df["text"].astype(str)
        ).map(self._normalize_text)
        return df

    def _build_disease_aliases(self) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for category, item in self.qa_payload.items():
            display_name = str(item.get("display_name", category))
            aliases[self._normalize_text(display_name)] = category
            aliases[self._normalize_text(category)] = category
            aliases[self._normalize_text(display_name).replace(" ", "")] = category
            aliases[self._normalize_text(category).replace(" ", "")] = category
        return aliases

    def _build_graph(self) -> None:
        self.graph.clear()
        self._disease_symptom_texts = {}
        for category, item in self.qa_payload.items():
            display_name = str(item.get("display_name", category))
            disease_node = category
            self.graph.add_node(
                disease_node,
                node_type="disease",
                category=category,
                display_name=display_name,
            )
            symptom_texts: List[str] = []
            for question in item.get("questions", []):
                symptom_text = self._normalize_text(question)
                if not symptom_text:
                    continue
                symptom_texts.append(symptom_text)
                symptom_node = f"SYMPTOM::{symptom_text}"
                self.graph.add_node(
                    symptom_node,
                    node_type="symptom",
                    text=str(question),
                    clean_text=symptom_text,
                )
                self.graph.add_edge(disease_node, symptom_node, relation="HAS_SYMPTOM", weight=1.0)
            self._disease_symptom_texts[category] = symptom_texts

        if "category" not in self.products_df.columns:
            raise ValueError("products_kb.csv must contain a 'category' column.")

        for idx, row in self.products_df.iterrows():
            category = str(row.get("category", "")).strip()
            product_name = str(row.get("product_name", "")).strip()
            if not product_name or not category:
                continue
            if category not in self.graph:
                self.graph.add_node(category, node_type="disease", category=category, display_name=str(row.get("display_name", category)))
            product_node = f"PRODUCT::{idx}::{self._normalize_text(product_name)}"
            self.graph.add_node(
                product_node,
                node_type="product",
                product_name=product_name,
                category=category,
                sale_price=row.get("sale_price", ""),
                dosage=row.get("dosage", ""),
                description=row.get("description", ""),
                usage=row.get("usage", ""),
                side_effects=row.get("side_effects", ""),
                is_prescription=row.get("is_prescription", ""),
                is_prescription_label=row.get("is_prescription_label", ""),
                text=row.get("text", ""),
                display_name=row.get("display_name", ""),
                source_group=row.get("source_group", ""),
            )
            self.graph.add_edge(category, product_node, relation="TREATS", weight=2.0)

    def _token_candidates(self, query: str) -> List[str]:
        clean = self._normalize_text(query)
        tokens = [tok for tok in clean.split() if tok]
        candidates = [clean]
        candidates.extend(tokens)
        for n in (2, 3):
            for i in range(max(0, len(tokens) - n + 1)):
                candidates.append(" ".join(tokens[i : i + n]))
        return list(dict.fromkeys(candidates))

    def _predict_disease_by_edges(self, query: str) -> tuple[str, float]:
        query_clean = self._normalize_text(query)
        query_compact = query_clean.replace(" ", "")
        query_tokens = set(self._token_candidates(query_clean))
        disease_scores: Dict[str, float] = {}
        for disease, symptom_texts in self._disease_symptom_texts.items():
            score = 0.0
            for symptom_text in symptom_texts:
                symptom_compact = symptom_text.replace(" ", "")
                if symptom_text and (symptom_text in query_clean or symptom_compact in query_compact):
                    score += 4.0
                symptom_tokens = set(self._token_candidates(symptom_text))
                overlap = query_tokens.intersection(symptom_tokens)
                if overlap:
                    score += float(len(overlap))
            disease_info = self.graph.nodes.get(disease, {})
            display_name = self._normalize_text(disease_info.get("display_name", disease))
            if display_name and (display_name in query_clean or display_name.replace(" ", "") in query_compact):
                score += 6.0
            disease_compact = self._normalize_text(disease).replace(" ", "")
            if disease_compact and disease_compact in query_compact:
                score += 8.0
            if score > 0:
                disease_scores[disease] = score
        if not disease_scores:
            return "", 0.0
        return max(disease_scores.items(), key=lambda item: item[1])

    def _is_safe_overlap(self, disease_score: float, query: str, category: str) -> bool:
        query_clean = self._normalize_text(query)
        if disease_score < 2.0:
            return False
        if len(query_clean.split()) < 2:
            return False
        dangerous_hints = ["đau ngực", "kho thở", "ho ra máu", "nôn ra máu", "suy hô hấp", "sốc", "đột quỵ", "thai", "mang thai", "co giật"]
        if any(hint in query_clean for hint in dangerous_hints):
            return False
        return True

    def _product_group(self, row: pd.Series) -> str:
        raw = " ".join(str(row.get(col, "")) for col in ("category", "display_name", "source_group", "usage", "description"))
        return self._normalize_text(raw).upper()

    def _allowed_product(self, category: str, row: pd.Series) -> bool:
        rules = self.HARD_CATEGORY_RULES.get(category, {})
        allowed_groups = [self._normalize_text(g).upper() for g in rules.get("allowed_groups", [])]
        forbidden_groups = [self._normalize_text(g).upper() for g in rules.get("forbidden_groups", [])]
        if not allowed_groups and not forbidden_groups:
            return True

        group = self._product_group(row)
        if allowed_groups and not any(g == "*" or g in group for g in allowed_groups):
            return False
        if any(g != "*" and g in group for g in forbidden_groups):
            return False
        return True

    def _is_otc_row(self, row: pd.Series) -> bool:
        rx_value = self._normalize_text(row.get("is_prescription") or row.get("is_prescription_label") or "")
        if not rx_value:
            return True
        return "khong" in rx_value or "otc" in rx_value or "khong ke don" in rx_value

    def _fallback_products(self, category: str, top_k: int) -> List[Dict[str, Any]]:
        clean_category = self._normalize_text(category)
        scored: List[Dict[str, Any]] = []
        for _, row in self.products_df.iterrows():
            if not self._allowed_product(category, row) or not self._is_otc_row(row):
                continue
            search_clean = str(row.get("_search_clean", ""))
            text_score = 0.0
            if clean_category and clean_category in search_clean:
                text_score += 5.0
            for token in clean_category.split():
                if token and token in search_clean:
                    text_score += 1.0
            if text_score > 0:
                product = row.to_dict()
                product["match_score"] = round(text_score, 3)
                product["matched_terms"] = [token for token in clean_category.split() if token and token in search_clean][:10]
                scored.append(product)
        scored.sort(key=lambda item: (-float(item.get("match_score", 0)), str(item.get("product_name", ""))))
        return scored[:top_k]

    def predict_medicine(self, user_input_symptom: str, top_k: int = TOP_K) -> GraphPredictionResult:
        start = time.perf_counter()
        category, disease_score = self._predict_disease_by_edges(user_input_symptom)
        if not category or not self._is_safe_overlap(disease_score, user_input_symptom, category):
            return GraphPredictionResult(disease="", disease_display_name="", products=[], latency_seconds=time.perf_counter() - start)

        best_data = self.graph.nodes.get(category, {})
        query_clean = self._normalize_text(user_input_symptom)
        query_compact = query_clean.replace(" ", "")
        query_tokens = self._token_candidates(user_input_symptom)
        category_clean = self._normalize_text(category)
        disease_display_name = self._normalize_text(best_data.get("display_name", category))
        disease_display_compact = disease_display_name.replace(" ", "")
        ranked_products: List[Dict[str, Any]] = []

        for _, row in self.products_df.iterrows():
            if not self._allowed_product(category, row) or not self._is_otc_row(row):
                continue
            row_blob = self._normalize_text(" ".join([str(row.get("product_name", "")), str(row.get("usage", "")), str(row.get("description", "")), str(row.get("category", ""))]))
            if category_clean not in row_blob and disease_display_name not in row_blob and disease_display_compact not in row_blob:
                continue
            score = 0.0
            if category_clean and category_clean in row_blob:
                score += 8.0
            if disease_display_name and disease_display_name in row_blob:
                score += 8.0
            if query_clean and query_clean in row_blob:
                score += 4.0
            if query_compact and query_compact in row_blob.replace(" ", ""):
                score += 4.0
            for token in query_tokens:
                if token and token in row_blob:
                    score += 1.5
            if score <= 0:
                continue
            product = row.to_dict()
            product["match_score"] = round(score + disease_score * 0.1, 3)
            product["matched_terms"] = [token for token in query_tokens if token and token in row_blob][:10]
            ranked_products.append(product)

        ranked_products.sort(key=lambda item: (-float(item.get("match_score", 0)), str(item.get("product_name", ""))))
        if not ranked_products:
            return GraphPredictionResult(disease=category, disease_display_name=best_data.get("display_name", category), products=[], latency_seconds=time.perf_counter() - start)

        return GraphPredictionResult(
            disease=category,
            disease_display_name=best_data.get("display_name", category),
            products=ranked_products[:top_k],
            latency_seconds=time.perf_counter() - start,
        )

    def predict(self, user_input_symptom: str, top_k: int = TOP_K) -> GraphPredictionResult:
        return self.predict_medicine(user_input_symptom, top_k=top_k)

    def evaluate(self, user_input_symptom: str) -> None:
        result = self.predict_medicine(user_input_symptom)
        print("=" * 100)
        print("GRAPH EMBEDDING MEDICINE PREDICTOR")
        print("=" * 100)
        print(f"Chẩn đoán bệnh: {result.disease_display_name or 'Không xác định'} ({result.disease or 'N/A'})")
        print("Danh sách thuốc gợi ý:")
        if result.products:
            for item in result.products:
                print(f"- {item['product_name']} | Nhóm: {item['category']} | Giá: {item['price']} | score={item['score']}")
        else:
            print("- Không có thuốc phù hợp")
        print(f"Latency xử lý đồ thị: {result.latency_seconds:.4f}s")
        print("=" * 100)


def predict_medicine(user_input_symptom: str) -> GraphPredictionResult:
    predictor = KnowledgeGraphMedicinePredictor()
    return predictor.predict_medicine(user_input_symptom)


if __name__ == "__main__":
    predictor = KnowledgeGraphMedicinePredictor()
    while True:
        try:
            query = input("\nNhập triệu chứng (exit để thoát): ").strip()
        except KeyboardInterrupt:
            print("\nĐã thoát.")
            break
        if query.lower() in {"exit", "quit", "thoát"}:
            break
        if not query:
            continue
        predictor.evaluate(query)

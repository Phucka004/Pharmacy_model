from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Union

import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.benchmark.shared_ranker import get_shared_product_ranker

DISEASE_MAP = "../../../data/silver/disease_category_map.csv"
PRODUCTS_CLEANED = "../../../data/silver/products_cleaned.csv"
REQUIRES_MEDICAL_VISIT = "REQUIRES_MEDICAL_VISIT"


class GraphKnowledgeLoadError(RuntimeError):
    pass


class GraphKnowledgePredictor:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.disease_map_df = pd.read_csv(self._resolve_data_path(DISEASE_MAP))
        self.products_df = pd.read_csv(self._resolve_data_path(PRODUCTS_CLEANED))
        self.product_ranker = get_shared_product_ranker()
        self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self._validate_columns()
        self._clean_frames()
        self.synonym_map = self._build_synonym_map()
        self._sorted_synonyms = sorted(self.synonym_map.keys(), key=len, reverse=True)
        self._build_graph()
        self.disease_embeddings, self.disease_embedding_records = self._build_disease_embeddings()
        self.disease_similarity_threshold = 0.60

    @staticmethod
    def _module_dir() -> Path:
        return Path(__file__).resolve().parent

    def _resolve_data_path(self, relative_path: str) -> Path:
        return (self._module_dir() / relative_path).resolve()

    @staticmethod
    def _normalize_text(value: object) -> str:
        text = str(value).strip().casefold()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^\w\s-]+", " ", text, flags=re.UNICODE)
        text = re.sub(r"[-_]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None:
            return True
        if pd.isna(value):
            return True
        text = str(value).strip()
        if not text:
            return True
        return text.lower() in {"nan", "none", "null", "n/a", "na", "-"}

    def _validate_columns(self) -> None:
        required_map_cols = {"disease_code", "disease_name", "category_key"}
        required_product_cols = {"category_key", "product_name"}

        if not required_map_cols.issubset(self.disease_map_df.columns):
            missing = required_map_cols - set(self.disease_map_df.columns)
            raise GraphKnowledgeLoadError(f"disease_category_map.csv thiếu cột: {', '.join(sorted(missing))}")

        if not required_product_cols.issubset(self.products_df.columns):
            missing = required_product_cols - set(self.products_df.columns)
            raise GraphKnowledgeLoadError(f"products_cleaned.csv thiếu cột: {', '.join(sorted(missing))}")

    def _clean_frames(self) -> None:
        self.disease_map_df["disease_code"] = self.disease_map_df["disease_code"].astype(str).str.strip()
        self.disease_map_df["disease_name"] = self.disease_map_df["disease_name"].astype(str).str.strip()
        self.disease_map_df["category_key"] = self.disease_map_df["category_key"].astype(str).str.strip()
        self.products_df["category_key"] = self.products_df["category_key"].astype(str).str.strip()
        self.products_df["product_name"] = self.products_df["product_name"].astype(str).str.strip()

    def _build_synonym_map(self) -> Dict[str, str]:
        synonym_map: Dict[str, str] = {}

        for _, row in self.disease_map_df.iterrows():
            disease_code = str(row.get("disease_code", "")).strip()
            disease_name = str(row.get("disease_name", "")).strip()
            if not disease_code:
                continue

            candidates = [disease_name]
            suffix = disease_code.split(":", 1)[-1] if ":" in disease_code else disease_code
            candidates.append(suffix)
            candidates.append(suffix.replace("-", " "))
            candidates.append(suffix.replace("_", " "))

            if disease_name:
                candidates.extend(
                    [
                        disease_name.replace("-", " "),
                        disease_name.replace("_", " "),
                    ]
                )

            for candidate in candidates:
                normalized = self._normalize_text(candidate)
                if normalized:
                    synonym_map[normalized] = disease_code

        return synonym_map

    def _build_graph(self) -> None:
        for _, row in self.disease_map_df.iterrows():
            disease_code = str(row.get("disease_code", "")).strip()
            category_raw = row.get("category_key", "")
            if not disease_code:
                continue

            self.graph.add_node(disease_code, node_type="Disease")
            if self._is_missing(category_raw):
                continue

            category_keys = [key.strip() for key in str(category_raw).split() if key.strip()]
            for category_key in category_keys:
                category_node = f"CATEGORY::{category_key}"
                self.graph.add_node(category_node, node_type="Category", category_key=category_key)
                self.graph.add_edge(disease_code, category_node, relation="MAP_TO")

        for _, row in self.products_df.iterrows():
            product_name = str(row.get("product_name", "")).strip()
            category_raw = row.get("category_key", "")
            if not product_name or self._is_missing(category_raw):
                continue

            category_keys = [key.strip() for key in str(category_raw).split() if key.strip()]
            if not category_keys:
                continue

            product_node = f"PRODUCT::{product_name}"
            self.graph.add_node(product_node, node_type="Product", product_name=product_name)
            for category_key in category_keys:
                category_node = f"CATEGORY::{category_key}"
                self.graph.add_node(category_node, node_type="Category", category_key=category_key)
                self.graph.add_edge(product_node, category_node, relation="BELONGS_TO")

    def _build_disease_embeddings(self) -> tuple[np.ndarray, List[Dict[str, str]]]:
        records: List[Dict[str, str]] = []
        texts: List[str] = []
        for _, row in self.disease_map_df.iterrows():
            disease_code = str(row.get("disease_code", "")).strip()
            if not disease_code:
                continue
            disease_name = str(row.get("disease_name", "")).strip()
            category_key = str(row.get("category_key", "")).strip()
            suffix = disease_code.split(":", 1)[-1] if ":" in disease_code else disease_code
            category_words = category_key.replace("-", " ").replace("_", " ")
            pieces = [disease_name, suffix, category_words]
            if not any(self._is_missing(piece) for piece in pieces):
                rich_text = " ".join(self._normalize_text(piece) for piece in pieces if self._normalize_text(piece))
            else:
                rich_text = " ".join(
                    part for part in [
                        self._normalize_text(disease_name),
                        self._normalize_text(suffix),
                        self._normalize_text(category_words),
                    ] if part
                )
            if not rich_text:
                continue
            texts.append(rich_text)
            records.append({
                "disease_code": disease_code,
                "category_key": category_key,
                "disease_name": disease_name,
                "embedding_text": rich_text,
            })
        if not texts:
            return np.empty((0, 0), dtype=np.float32), []
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32), records

    def _semantic_detect_disease_code(self, query_text: str) -> str | None:
        normalized_query = self._normalize_text(query_text)
        if not normalized_query or self.disease_embeddings.size == 0:
            return None

        query_embedding = self.embedding_model.encode([normalized_query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        query_vec = np.asarray(query_embedding, dtype=np.float32)[0]
        similarities = np.dot(self.disease_embeddings, query_vec)
        if similarities.size == 0:
            return None

        top_k = min(5, len(similarities))
        top_indices = np.argsort(-similarities)[:top_k]
        for index in top_indices:
            score = float(similarities[int(index)])
            if score < self.disease_similarity_threshold:
                continue
            record = self.disease_embedding_records[int(index)]
            if score >= self.disease_similarity_threshold:
                return record.get("disease_code")
        return None

    def detect_disease_code(self, query_text: str) -> str | None:
        normalized_query = self._normalize_text(query_text)
        if not normalized_query:
            return None

        for synonym in self._sorted_synonyms:
            if synonym and synonym in normalized_query:
                return self.synonym_map[synonym]

        return self._semantic_detect_disease_code(query_text)

    def _get_category_nodes(self, actual_disease_code: str) -> List[str]:
        disease_node = str(actual_disease_code).strip()
        if not disease_node or disease_node not in self.graph:
            return []

        return [
            neighbor
            for neighbor in self.graph.successors(disease_node)
            if self.graph.nodes[neighbor].get("node_type") == "Category"
        ]

    def _candidate_products_for_categories(self, category_nodes: List[str]) -> pd.DataFrame:
        category_keys = [
            str(self.graph.nodes[node].get("category_key", "")).strip()
            for node in category_nodes
            if str(self.graph.nodes[node].get("category_key", "")).strip()
        ]
        if not category_keys:
            return pd.DataFrame()

        category_key_set = set(category_keys)
        candidate_products = self.products_df.loc[
            self.products_df["category_key"].astype(str).apply(lambda text: any(cat in text.split() for cat in category_key_set))
        ].copy()
        if candidate_products.empty:
            return pd.DataFrame()

        candidate_products = candidate_products.drop_duplicates(
            subset=[col for col in ["sku", "product_name"] if col in candidate_products.columns],
            keep="first",
        )
        candidate_products = candidate_products.reset_index(drop=True)
        return candidate_products

    def predict_products_for_query(self, query_text: str) -> Union[str, List[str]]:
        actual_disease_code = self.detect_disease_code(query_text)
        if not actual_disease_code:
            return REQUIRES_MEDICAL_VISIT

        category_nodes = self._get_category_nodes(actual_disease_code)
        if not category_nodes:
            return REQUIRES_MEDICAL_VISIT

        category_keys = [
            str(self.graph.nodes[node].get("category_key", "")).strip()
            for node in category_nodes
            if str(self.graph.nodes[node].get("category_key", "")).strip()
        ]
        if not category_keys:
            return REQUIRES_MEDICAL_VISIT

        candidate_products = self._candidate_products_for_categories(category_nodes)
        if candidate_products.empty:
            return REQUIRES_MEDICAL_VISIT

        products = self.product_ranker.rank_products(
            query_text,
            candidate_products,
            category_priority=category_keys,
            top_k=5,
        )
        if not products:
            return REQUIRES_MEDICAL_VISIT
        return products

    def get_products_by_disease(self, actual_disease_code: str) -> Union[str, List[str]]:
        category_nodes = self._get_category_nodes(actual_disease_code)
        if not category_nodes:
            return REQUIRES_MEDICAL_VISIT

        matched_row = self.disease_map_df.loc[self.disease_map_df["disease_code"] == str(actual_disease_code).strip()]
        if matched_row.empty:
            return REQUIRES_MEDICAL_VISIT

        disease_name = str(matched_row.iloc[0].get("disease_name", "")).strip()
        if self._is_missing(disease_name):
            return REQUIRES_MEDICAL_VISIT

        category_keys = [
            str(self.graph.nodes[node].get("category_key", "")).strip()
            for node in category_nodes
            if str(self.graph.nodes[node].get("category_key", "")).strip()
        ]
        if not category_keys:
            return REQUIRES_MEDICAL_VISIT

        candidate_products = self._candidate_products_for_categories(category_nodes)
        if candidate_products.empty:
            return REQUIRES_MEDICAL_VISIT
        products = self.product_ranker.rank_products(
            disease_name,
            candidate_products,
            category_priority=category_keys,
            top_k=5,
        )
        if not products:
            return REQUIRES_MEDICAL_VISIT
        return products


__all__ = [
    "DISEASE_MAP",
    "PRODUCTS_CLEANED",
    "REQUIRES_MEDICAL_VISIT",
    "GraphKnowledgeLoadError",
    "GraphKnowledgePredictor",
]

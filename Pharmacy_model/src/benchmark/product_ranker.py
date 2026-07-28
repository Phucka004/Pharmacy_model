from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_PRODUCTS = 5
CATEGORY_PRIORITY_BONUS = (0.05, 0.03, 0.01)


class ProductRanker:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self._dataset_cache: dict[str, tuple[list[dict[str, str]], np.ndarray]] = {}

    @staticmethod
    def _normalize_text(value: object) -> str:
        text = str(value).strip()
        return " ".join(text.split())

    @staticmethod
    def _normalize_category_key(value: object) -> str:
        return ProductRanker._normalize_text(value).lower()

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

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _embed_query(self, query_text: str) -> np.ndarray:
        embedding = self.embedding_model.encode(
            [self._normalize_text(query_text)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32)[0]

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.array([], dtype=np.float32)
        return np.dot(matrix, query_vec)

    def _dataset_signature(self, products_df: pd.DataFrame) -> str:
        if products_df is None or products_df.empty:
            return "empty"

        cols = [col for col in ["sku", "product_name", "category_key", "category_name", "category", "usage", "description"] if col in products_df.columns]
        if not cols:
            cols = list(products_df.columns)

        normalized = products_df.loc[:, cols].copy()
        for col in normalized.columns:
            normalized[col] = normalized[col].fillna("").astype(str).map(self._normalize_text)

        hashed = pd.util.hash_pandas_object(normalized, index=False).astype("uint64")
        return f"{tuple(cols)}:{int(hashed.sum())}:{len(normalized)}"

    def _extract_product_categories(self, row: pd.Series | dict) -> list[str]:
        raw_value = ""
        if isinstance(row, pd.Series):
            for key in ("category_key", "category_name", "category"):
                value = row.get(key, "")
                if not self._is_missing(value):
                    raw_value = str(value)
                    break
        else:
            for key in ("category_key", "category_name", "category"):
                value = row.get(key, "")
                if not self._is_missing(value):
                    raw_value = str(value)
                    break

        if not raw_value:
            return []

        tokens = [token.strip() for token in str(raw_value).replace(",", " ").split() if token.strip()]
        return [self._normalize_category_key(token) for token in tokens if token.strip()]

    def build_product_chunks(self, products_df: pd.DataFrame) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        if products_df is None or products_df.empty:
            return chunks

        for _, row in products_df.iterrows():
            product_name = str(row.get("product_name", "")).strip()
            if self._is_missing(product_name):
                continue

            sku = str(row.get("sku", "")).strip()
            category_key = str(row.get("category_key", "")).strip()
            category_name = str(row.get("category_name", row.get("category", ""))).strip()
            usage = str(row.get("usage", "")).strip()
            description = str(row.get("description", "")).strip()

            category_text = category_name or category_key
            text_chunk = (
                f"Sản phẩm: {product_name}. "
                f"Thuộc nhóm: {category_text}. "
                f"Công dụng: {usage}. "
                f"Mô tả: {description}"
            )
            chunks.append(
                {
                    "sku": sku,
                    "product_name": product_name,
                    "category_key": category_key,
                    "category_name": category_name,
                    "text_chunk": text_chunk,
                }
            )
        return chunks

    def _get_cached_embeddings(self, products_df: pd.DataFrame) -> tuple[list[dict[str, str]], np.ndarray]:
        signature = self._dataset_signature(products_df)
        cached = self._dataset_cache.get(signature)
        if cached is not None:
            return cached

        chunks = self.build_product_chunks(products_df)
        if not chunks:
            cached_value = ([], np.empty((0, 0), dtype=np.float32))
            self._dataset_cache[signature] = cached_value
            return cached_value

        product_embeddings = self._encode_texts([item["text_chunk"] for item in chunks])
        cached_value = (chunks, product_embeddings)
        self._dataset_cache[signature] = cached_value
        return cached_value

    def _category_priority_bonus(self, product_categories: Sequence[str], category_priority: Sequence[str] | None) -> float:
        if not category_priority:
            return 0.0

        normalized_priority = [self._normalize_category_key(item) for item in category_priority if not self._is_missing(item)]
        if not normalized_priority or not product_categories:
            return 0.0

        best_bonus = 0.0
        for product_category in product_categories:
            if not product_category:
                continue
            for idx, priority_category in enumerate(normalized_priority):
                if product_category == priority_category:
                    bonus = CATEGORY_PRIORITY_BONUS[idx] if idx < len(CATEGORY_PRIORITY_BONUS) else max(0.10 - idx * 0.04, 0.0)
                    best_bonus = max(best_bonus, bonus)
                    break
        return float(best_bonus)

    def rank_products_with_scores(
        self,
        query_text: str,
        products_df: pd.DataFrame,
        category_priority: Sequence[str] | None = None,
        top_k: int = TOP_K_PRODUCTS,
    ) -> list[dict[str, object]]:
        chunks, product_embeddings = self._get_cached_embeddings(products_df)
        if not chunks or product_embeddings.size == 0:
            return []

        query_vec = self._embed_query(query_text)
        similarities = self._cosine_similarity(query_vec, product_embeddings)
        if similarities.size == 0:
            return []

        ranked_indices = np.argsort(-similarities)
        scored_products: list[dict[str, object]] = []
        seen: set[str] = set()

        for idx in ranked_indices:
            chunk = chunks[int(idx)]
            product_name = str(chunk.get("product_name", "")).strip()
            if not product_name or product_name in seen:
                continue

            semantic_score = float(similarities[int(idx)])
            product_categories = self._extract_product_categories(chunk)
            category_bonus = self._category_priority_bonus(product_categories, category_priority)
            final_score = semantic_score + category_bonus

            scored_products.append(
                {
                    "sku": str(chunk.get("sku", "")).strip(),
                    "product_name": product_name,
                    "category_key": str(chunk.get("category_key", "")).strip(),
                    "category_name": str(chunk.get("category_name", "")).strip(),
                    "semantic_score": round(semantic_score, 6),
                    "category_bonus": round(category_bonus, 6),
                    "final_score": round(final_score, 6),
                }
            )
            seen.add(product_name)

        scored_products.sort(
            key=lambda item: (
                float(item["final_score"]),
                float(item["semantic_score"]),
                float(item["category_bonus"]),
                str(item["product_name"]),
            ),
            reverse=True,
        )
        return scored_products[: min(top_k, len(scored_products))]

    def rank_products(
        self,
        query_text: str,
        products_df: pd.DataFrame,
        category_priority: Sequence[str] | None = None,
        top_k: int = TOP_K_PRODUCTS,
    ) -> list[str]:
        scored_products = self.rank_products_with_scores(
            query_text=query_text,
            products_df=products_df,
            category_priority=category_priority,
            top_k=top_k,
        )
        return [item["product_name"] for item in scored_products]


__all__ = [
    "CATEGORY_PRIORITY_BONUS",
    "EMBEDDING_MODEL_NAME",
    "TOP_K_PRODUCTS",
    "ProductRanker",
]

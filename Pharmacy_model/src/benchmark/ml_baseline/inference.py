from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.benchmark.shared_ranker import get_shared_product_ranker

QA_DATA = "../../../data/silver/synthetic_medical_qa.json"
DISEASE_MAP = "../../../data/silver/disease_category_map.csv"
PRODUCTS_CLEANED = "../../../data/silver/products_cleaned.csv"
RANDOM_STATE = 42
REQUIRES_MEDICAL_VISIT = "REQUIRES_MEDICAL_VISIT"


class BaselineModelLoadError(RuntimeError):
    pass


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_data_path(relative_path: str) -> Path:
    return (_module_dir() / relative_path).resolve()


def _normalize_text(text: object) -> str:
    return str(text).strip().lower()


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    normalized = str(value).strip()
    if not normalized:
        return True
    return normalized.lower() in {"nan", "none", "null", "n/a", "na", "-"}


def _load_training_examples(qa_path: Path) -> tuple[list[str], list[str]]:
    with qa_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise BaselineModelLoadError("synthetic_medical_qa.json phải là object dạng dictionary")

    questions: list[str] = []
    labels: list[str] = []

    for disease_code, item in payload.items():
        if not isinstance(item, dict):
            continue
        question_items = item.get("questions", [])
        if not isinstance(question_items, list):
            continue
        for question in question_items:
            question_text = _normalize_text(question)
            if question_text:
                questions.append(question_text)
                labels.append(str(disease_code).strip())

    return questions, labels


def load_and_train_baseline() -> Pipeline:
    qa_path = _resolve_data_path(QA_DATA)
    questions, labels = _load_training_examples(qa_path)
    if not questions:
        raise BaselineModelLoadError("Không tìm thấy câu hỏi huấn luyện hợp lệ trong synthetic_medical_qa.json")

    pipeline = Pipeline(
        steps=[
            ("tfidfvectorizer", TfidfVectorizer(ngram_range=(1, 2))),
            ("linearsvc", LinearSVC(random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(questions, labels)
    return pipeline


class MLBaselinePredictor:
    def __init__(self) -> None:
        self.pipeline = load_and_train_baseline()
        self.disease_map_df = pd.read_csv(_resolve_data_path(DISEASE_MAP))
        self.products_df = pd.read_csv(_resolve_data_path(PRODUCTS_CLEANED))
        self.product_ranker = get_shared_product_ranker()

        required_map_cols = {"disease_code", "category_key"}
        required_product_cols = {"category_key", "product_name"}
        if not required_map_cols.issubset(self.disease_map_df.columns):
            missing = required_map_cols - set(self.disease_map_df.columns)
            raise BaselineModelLoadError(f"disease_category_map.csv thiếu cột: {', '.join(sorted(missing))}")
        if not required_product_cols.issubset(self.products_df.columns):
            missing = required_product_cols - set(self.products_df.columns)
            raise BaselineModelLoadError(f"products_cleaned.csv thiếu cột: {', '.join(sorted(missing))}")

        self.disease_map_df["disease_code"] = self.disease_map_df["disease_code"].astype(str).str.strip()
        self.products_df["category_key"] = self.products_df["category_key"].astype(str).str.strip()
        self.products_df["product_name"] = self.products_df["product_name"].astype(str).str.strip()

    def predict_disease_code(self, query_text: str) -> str:
        normalized_query = _normalize_text(query_text)
        prediction = self.pipeline.predict([normalized_query])[0]
        return str(prediction).strip()

    def _lookup_category_key(self, disease_code: str) -> str:
        matched_rows = self.disease_map_df.loc[self.disease_map_df["disease_code"] == disease_code]
        if matched_rows.empty:
            return ""

        category_key = matched_rows.iloc[0]["category_key"]
        if _is_missing_value(category_key):
            return ""
        return str(category_key).strip()

    def _candidate_products_for_categories(self, category_raw: str) -> pd.DataFrame:
        if _is_missing_value(category_raw):
            return pd.DataFrame()

        category_keys = [k.strip() for k in str(category_raw).split() if k.strip()]
        if not category_keys:
            return pd.DataFrame()

        candidate_products = self.products_df.loc[self.products_df["category_key"].isin(category_keys)].copy()
        if candidate_products.empty:
            return pd.DataFrame()

        candidate_products = candidate_products.drop_duplicates(
            subset=[col for col in ["sku", "product_name"] if col in candidate_products.columns],
            keep="first",
        )
        candidate_products = candidate_products.reset_index(drop=True)
        return candidate_products

    def predict_products_for_query(self, query_text: str) -> Union[str, List[str]]:
        disease_code = self.predict_disease_code(query_text)
        category_raw = self._lookup_category_key(disease_code)

        if _is_missing_value(category_raw):
            return REQUIRES_MEDICAL_VISIT

        category_keys = [k.strip() for k in str(category_raw).split() if k.strip()]
        if not category_keys:
            return REQUIRES_MEDICAL_VISIT

        candidate_products = self._candidate_products_for_categories(category_raw)
        if candidate_products.empty:
            return REQUIRES_MEDICAL_VISIT

        products = self.product_ranker.rank_products(
            query_text,
            candidate_products,
            category_priority=category_keys,
            top_k=5,
        )
        return products if products else REQUIRES_MEDICAL_VISIT


__all__ = [
    "QA_DATA",
    "DISEASE_MAP",
    "PRODUCTS_CLEANED",
    "REQUIRES_MEDICAL_VISIT",
    "BaselineModelLoadError",
    "load_and_train_baseline",
    "MLBaselinePredictor",
]

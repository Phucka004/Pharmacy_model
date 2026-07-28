from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data_processor import MedicalDataProcessor


class MedicalMLTrainer:
    def __init__(self, qa_path: str = "data/silver/synthetic_medical_qa.json", model_path: str = "models/medical_classifier.pkl", keywords_path: str = "data/silver/extracted_keywords_top10.json", products_path: str = "data/silver/products_kb.csv"):
        self.qa_path = Path(qa_path)
        self.keywords_path = Path(keywords_path)
        self.products_path = Path(products_path)
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.products_df = self._load_products_df()

    def _load_products_df(self) -> pd.DataFrame:
        if not self.products_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.products_path)
        for col in ["product_name", "display_name", "usage", "description", "category", "_search_clean"]:
            if col not in df.columns:
                df[col] = ""
        df["_search_clean"] = df.get("_search_clean", "").fillna("") if "_search_clean" in df.columns else ""
        return df.fillna("")

    def build_pipeline(self, vocabulary: List[str] | None = None) -> Pipeline:
        tfidf_kwargs = dict(
            ngram_range=(1, 3),
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            use_idf=True,
            sublinear_tf=True,
            norm="l2",
        )
        if vocabulary:
            tfidf_kwargs["vocabulary"] = sorted(set(vocabulary))
        else:
            tfidf_kwargs["max_features"] = 50000
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
                ("clf", LinearSVC(C=1.0, max_iter=2000)),
            ]
        )

    def _load_keywords_vocabulary(self) -> List[str]:
        if not self.keywords_path.exists():
            vocab = []
        else:
            with self.keywords_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            vocab = []
            for item in payload.values():
                vocab.extend([str(kw).strip() for kw in item.get("keywords", []) if str(kw).strip()])
        pregnancy_seeds = ["thai", "thai ky", "thai kỳ", "co thai", "có thai", "mang thai", "dang mang thai", "đau bung", "đau bụng", "đau bung duoi", "đau bụng dưới"]
        vocab.extend(pregnancy_seeds)
        return list(dict.fromkeys(vocab))

    def _load_product_symptom_signals(self) -> List[str]:
        if self.products_df.empty:
            return []
        cols = [c for c in ["usage", "description", "category", "display_name", "product_name", "_search_clean"] if c in self.products_df.columns]
        signals: List[str] = []
        for _, row in self.products_df.iterrows():
            blob = " ".join(str(row.get(c, "")) for c in cols)
            tokens = MedicalDataProcessor.clean_and_tokenize(blob)
            signals.extend(tokens)
        return list(dict.fromkeys(signals))

    def train(self) -> Pipeline:
        processor = MedicalDataProcessor(self.qa_path, self.keywords_path)
        X, y, _ = processor.load_samples()
        vocabulary = self._load_keywords_vocabulary() + self._load_product_symptom_signals()
        model = self.build_pipeline(vocabulary=vocabulary)
        model.fit(X, y)
        with self.model_path.open("wb") as f:
            pickle.dump(model, f)
        return model


def train_medical_classifier(qa_path: str = "data/silver/synthetic_medical_qa.json", model_path: str = "models/medical_classifier.pkl"):
    return MedicalMLTrainer(qa_path=qa_path, model_path=model_path).train()

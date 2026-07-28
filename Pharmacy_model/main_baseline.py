from __future__ import annotations

import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.baseline_ml.data_processor import MedicalDataProcessor

PROJECT_ROOT = Path(__file__).resolve().parent
QA_PATH = PROJECT_ROOT / "data" / "silver" / "synthetic_medical_qa.json"
KEYWORDS_PATH = PROJECT_ROOT / "data" / "silver" / "extracted_keywords_top10.json"
PRODUCTS_PATH = PROJECT_ROOT / "data" / "silver" / "products_kb.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "medical_classifier_strict.pkl"
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_strict_report.txt"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_QUERIES = [
    "Bé nhà em bị ho khan sốt về chiều là bệnh gì ạ?",
    "Em đi tiểu nhiều lần trong đêm và tiểu gấp, có phải bàng quang tăng hoạt không?",
]


def load_dataset() -> Tuple[List[str], List[str], List[dict], Dict[str, List[str]]]:
    processor = MedicalDataProcessor(QA_PATH, KEYWORDS_PATH)
    X, y, records = processor.load_samples()
    with KEYWORDS_PATH.open("r", encoding="utf-8") as f:
        keywords_map = {k: [unicodedata.normalize("NFC", kw) for kw in v.get("keywords", [])] for k, v in json.load(f).items()}
    return X, y, records, keywords_map


def build_pipeline(vocabulary: List[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    vocabulary=vocabulary,
                    ngram_range=(1, 3),
                    lowercase=False,
                    tokenizer=str.split,
                    preprocessor=None,
                    token_pattern=None,
                    use_idf=True,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "clf",
                LinearSVC(C=1.0, max_iter=5000),
            ),
        ]
    )


def load_products_df() -> pd.DataFrame:
    if not PRODUCTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRODUCTS_PATH)
    for col in ["product_name", "display_name", "usage", "description", "category", "_search_clean"]:
        if col not in df.columns:
            df[col] = ""
    if not df["_search_clean"].astype(str).str.strip().any():
        df["_search_clean"] = (
            df["product_name"].astype(str)
            + " "
            + df["display_name"].astype(str)
            + " "
            + df["usage"].astype(str)
            + " "
            + df["description"].astype(str)
            + " "
            + df["category"].astype(str)
        )
    return df.fillna("")


def train_and_evaluate() -> Dict[str, Any]:
    X, y, _, keywords_map = load_dataset()
    products_df = load_products_df()
    product_signals: List[str] = []
    if not products_df.empty:
        for _, row in products_df.iterrows():
            blob = " ".join(str(row.get(col, "")) for col in ["product_name", "display_name", "usage", "description", "category", "_search_clean"])
            product_signals.extend(MedicalDataProcessor.clean_and_tokenize(blob))
    vocabulary = sorted({unicodedata.normalize("NFC", kw) for kws in keywords_map.values() for kw in kws if kw} | {unicodedata.normalize("NFC", tok) for tok in product_signals if tok})
    if not vocabulary:
        raise RuntimeError("Strict keyword vocabulary is empty.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(vocabulary)

    start_train = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start_train

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    report_text = (
        f"Strict Baseline Evaluation Report\n"
        f"{'=' * 80}\n"
        f"QA path: {QA_PATH}\n"
        f"Keywords path: {KEYWORDS_PATH}\n"
        f"Vocabulary size: {len(vocabulary)}\n"
        f"Train size: {len(X_train)}\n"
        f"Test size: {len(X_test)}\n"
        f"Training time: {train_seconds:.2f}s\n"
        f"Accuracy: {acc:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n"
        f"\nClassification Report\n{report}\n"
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    return {
        "model": pipeline,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_seconds": train_seconds,
        "report_text": report_text,
        "vocabulary": vocabulary,
        "keywords_map": keywords_map,
    }


def infer_with_keywords(model: Pipeline, query: str, keywords_map: Dict[str, List[str]]) -> None:
    query_norm = unicodedata.normalize("NFC", query)
    query_tokens = MedicalDataProcessor.clean_and_tokenize(query_norm)
    query_ngrams = {" ".join(query_tokens[i : i + n]) for n in (1, 2, 3) for i in range(max(0, len(query_tokens) - n + 1))}
    best_label = None
    best_hits: List[str] = []
    best_count = -1
    for label, keywords in keywords_map.items():
        hits = []
        for kw in keywords:
            kw_norm = unicodedata.normalize("NFC", kw)
            if kw_norm in query_ngrams:
                hits.append(kw)
        if len(hits) > best_count:
            best_label = label
            best_hits = hits
            best_count = len(hits)

    model_pred = model.predict([query_norm])[0]
    prediction = best_label or model_pred
    activated_keywords = best_hits[:10]
    print("\nSanity check inference")
    print(f"Query: {query}")
    print(f"Normalized query: {' '.join(query_tokens)}")
    print(f"Predicted label: {prediction}")
    print(f"Model raw prediction: {model_pred}")
    print(f"Activated strict keywords: {activated_keywords}")


def main() -> None:
    result = train_and_evaluate()
    print("Training completed.")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro F1: {result['macro_f1']:.4f}")
    print(f"Weighted F1: {result['weighted_f1']:.4f}")
    print(f"Report saved to: {REPORT_PATH}")

    for query in SAMPLE_QUERIES:
        infer_with_keywords(result["model"], query, result["keywords_map"])


if __name__ == "__main__":
    main()

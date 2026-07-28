from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from unidecode import unidecode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "vector_db"
QA_PATH = PROJECT_ROOT / "data" / "silver" / "synthetic_medical_qa.json"
PRODUCTS_PATH = PROJECT_ROOT / "data" / "silver" / "products_kb.csv"
COLLECTION_NAME = "pharmacy_knowledge"
BATCH_SIZE = 100
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _md5_id(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"SYN_MED_{digest}"


def _load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_records(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for category, raw_item in payload.items():
        display_name = raw_item.get("display_name", category)
        questions = raw_item.get("questions", [])
        for question in questions:
            question_text = str(question).strip()
            if not question_text:
                continue
            unidecoded_question = unidecode(question_text)
            yield {
                "id": _md5_id(f"{category}|{display_name}|{question_text}"),
                "document": f"Bệnh: {display_name} | Triệu chứng gốc: {question_text} | Triệu chứng không dấu: {unidecoded_question}",
                "metadata": {
                    "original_content": question_text,
                    "type": "synthetic_qa",
                    "sub_type": "medical",
                    "category": category,
                    "display_name": display_name,
                },
            }


def _iter_product_records(df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    for idx, row in df.iterrows():
        product_name = str(row.get("product_name", "")).strip()
        usage = str(row.get("usage", "")).strip()
        description = str(row.get("description", "")).strip()
        if not product_name:
            continue
        yield {
            "id": _md5_id(f"product|{idx}|{product_name}|{usage}|{description}"),
            "document": f"Tên sản phẩm: {product_name} | Công dụng: {usage} | Mô tả: {description}",
            "metadata": {
                "type": "product",
                "product_name": product_name,
                "category": str(row.get("category", "")).strip(),
                "price": row.get("price", "Liên hệ"),
            },
        }


def _chunk(items: list[Dict[str, Any]], size: int) -> Iterable[list[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ingest() -> None:
    if not QA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {QA_PATH}")
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Missing products file: {PRODUCTS_PATH}")

    DB_PATH.mkdir(parents=True, exist_ok=True)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_func,
    )

    for delete_where, label in (({"type": "synthetic_qa"}, "synthetic_qa"), ({"type": "product"}, "product")):
        try:
            collection.delete(where=delete_where)
            print(f"[INGEST] Removed old {label} records.")
        except Exception as exc:  # pragma: no cover
            print(f"[INGEST] Delete skipped for {label}: {exc}")

    qa_records = list(_iter_records(_load_payload(QA_PATH)))
    product_df = pd.read_csv(PRODUCTS_PATH)
    product_records = list(_iter_product_records(product_df))
    all_records = qa_records + product_records
    total = len(all_records)
    print(f"[INGEST] Loaded {len(qa_records)} QA records and {len(product_records)} product records.")

    processed = 0
    for batch in _chunk(all_records, BATCH_SIZE):
        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["document"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )
        processed += len(batch)
        print(f"[INGEST] Upserted {processed}/{total} records.")

    print("[INGEST] Completed successfully.")


if __name__ == "__main__":
    ingest()

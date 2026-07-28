from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.rag.inference import RAGPredictor

predictor = RAGPredictor()


def _print_header() -> None:
    print("=" * 90)
    print("SYMPTOMS-AGGREGATED RAG DEBUG TEST")
    print(f"Project root: {ROOT_DIR}")
    print("=" * 90)


def _print_retrieval_context(query_text: str) -> None:
    print("-- [Retrieval Context - Top 3 Similar Disease Knowledge Chunks] --")

    query_vec = predictor.embedding_model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    similarities = predictor._cosine_similarity(
        query_vec,
        predictor.disease_embeddings,
    )

    if similarities.size == 0:
        print("(No retrieval context available)")
        return

    top_indices = np.argsort(-similarities)[:3]

    for rank, idx in enumerate(top_indices, start=1):
        chunk = predictor.disease_knowledge_chunks[int(idx)]
        disease_code = chunk["disease_code"]
        score = float(similarities[int(idx)])
        text_preview = chunk["text_chunk"][:200]

        print(f"{rank}. Mã bệnh: {disease_code} | Điểm tương đồng: {score:.4f}")
        print(f"   -> Nội dung Chunk: {text_preview}...")


def _run_case(title: str, query_text: str) -> None:
    print("\n" + "=" * 90)
    print(f"[{title}]")
    print("-" * 90)
    print(f"Query: {query_text}")

    _print_retrieval_context(query_text)

    detected_code = predictor.detect_disease_code(query_text)
    print(f"Ollama Detected Disease Code: {detected_code}")

    result = predictor.predict_products_for_query(query_text)

    if isinstance(result, list):
        print(f"Top 5 Products Count: {len(result)}")
        print(f"Top 5 Products: {result}")
    else:
        print(f"Final Result: {result}")
        print("Guardrail: bệnh nguy hiểm hoặc không đủ độ tin cậy -> kích hoạt an toàn.")


def main() -> None:
    _print_header()

    test_cases = [
        ("Test 1 (Sổ mũi)",
         "Tôi bị sổ mũi")
    ]

    for title, query in test_cases:
        _run_case(title, query)

    print("\nDone.")


if __name__ == "__main__":
    main()

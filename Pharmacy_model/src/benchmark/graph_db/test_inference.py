from __future__ import annotations

import sys
import time

from src.benchmark.graph_db.inference import GraphKnowledgePredictor, REQUIRES_MEDICAL_VISIT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QUERIES = [
    "Tôi bị cúm",
    "Sốt cao mệt mỏi",
    "Ho khan và đau rát cổ họng",
    "Đau rát cổ họng, khó nuốt",
    "Tôi bị trào ngược dạ dày, ợ chua, buồn nôn",
    "Đi ngoài nhiều lần, phân lỏng kèm nước bất thường",
    "Đi nặng khó khăn, uống ít nước",
    "Tôi bị ngứa da liên tục và có những nốt sần màu vàng sữa nổi lên",
    "Tôi bị rụng tóc",
    "Tôi bị mất ngủ",
    "Tôi bị chướng bụng đầy hơi",
    "Tôi bị táo bón",
    "Tôi bị viêm họng"
]


def _print_result(query: str, disease: str, category_raw: str, result: object, latency: float) -> None:
    print("-" * 24)
    print(f"Query: {query}")
    print(f"Disease: {disease}")
    print(f"Category: {category_raw}")
    print(f"Top 5 products: {result}")
    print(f"Latency: {latency:.4f}s")
    print("-" * 24)


def main() -> None:
    predictor = GraphKnowledgePredictor()
    for query in QUERIES:
        disease_code = predictor.detect_disease_code(query) or "None"
        matched_row = predictor.disease_map_df.loc[predictor.disease_map_df["disease_code"] == str(disease_code).strip()]
        category_raw = matched_row.iloc[0]["category_key"] if not matched_row.empty else ""
        start = time.perf_counter()
        result = predictor.predict_products_for_query(query)
        latency = time.perf_counter() - start
        _print_result(query, disease_code, category_raw, result, latency)
        if result == REQUIRES_MEDICAL_VISIT:
            print("Outcome: REQUIRES_MEDICAL_VISIT")

    print("\nRe-run same query twice to observe cache behavior")
    query = QUERIES[0]
    disease_code = predictor.detect_disease_code(query) or "None"
    matched_row = predictor.disease_map_df.loc[predictor.disease_map_df["disease_code"] == str(disease_code).strip()]
    category_raw = matched_row.iloc[0]["category_key"] if not matched_row.empty else ""

    start = time.perf_counter()
    result1 = predictor.predict_products_for_query(query)
    latency1 = time.perf_counter() - start
    _print_result(f"{query} [run 1]", disease_code, category_raw, result1, latency1)

    start = time.perf_counter()
    result2 = predictor.predict_products_for_query(query)
    latency2 = time.perf_counter() - start
    _print_result(f"{query} [run 2]", disease_code, category_raw, result2, latency2)


if __name__ == "__main__":
    main()

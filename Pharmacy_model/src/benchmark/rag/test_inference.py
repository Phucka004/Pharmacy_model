from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.rag.inference import RAGPredictor


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


def main() -> None:
    predictor = RAGPredictor()

    for i, query in enumerate(QUERIES, 1):
        trace = predictor.debug_trace_query(query)

        print("=" * 70)
        print(f"Test {i}")
        print(f"Query           : {query}")
        print(f"Detected Disease: {trace.get('final_disease_code')}")

        products = trace.get("final_products", [])

        print("Top-5 Products:")
        if products:
            for rank, product in enumerate(products[:5], 1):
                print(f"  {rank}. {product}")
        else:
            print("  Không có kết quả.")

        latency = trace.get("latency", {})
        total_ms = latency.get("total_ms", "N/A") if isinstance(latency, dict) else latency
        print(f"Response Time   : {total_ms} ms")

        print()


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal runtime: {time.time() - start:.2f}s")
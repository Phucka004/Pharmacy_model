from __future__ import annotations

import time

from src.graph_embedding.graph_model import KnowledgeGraphMedicinePredictor

DEFAULT_QUESTION = "Em bị ho khan, sốt nhẹ và mệt mỏi kéo dài, nên uống thuốc gì?"


def main() -> None:
    predictor = KnowledgeGraphMedicinePredictor()
    print("=" * 60)
    print("📊 KNOWLEDGE GRAPH MODEL (NO LLM - NO RAG)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 Khách hàng (Enter dùng câu mặc định): ").strip()
        except KeyboardInterrupt:
            print("\nĐã thoát.")
            break

        if user_input.lower() in {"thoát", "exit"}:
            break
        if not user_input:
            user_input = DEFAULT_QUESTION

        start = time.perf_counter()
        result = predictor.predict_medicine(user_input)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        print("=" * 60)
        print("📊 KNOWLEDGE GRAPH MODEL (NO LLM - NO RAG)")
        print("=" * 60)
        print(f"👤 Khách hàng: {user_input}")
        print(f"🩺 Chẩn đoán từ Đồ thị: {result.disease_display_name or 'Không xác định'}")
        print()
        print("--- 💊 CÁC SẢN PHẨM GỢI Ý TỪ ĐỒ THỊ ---")
        if result.products:
            for item in result.products:
                print(f"- {item['product_name']} | Nhóm: {item['category']} | Giá: {item['price']}")
        else:
            print("- Không có sản phẩm phù hợp")
        print("=" * 60)
        print(f"⏱️ Latency xử lý đồ thị: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()

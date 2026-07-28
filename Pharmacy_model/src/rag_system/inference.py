from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "vector_db"
PRODUCTS_PATH = PROJECT_ROOT / "data" / "silver" / "products_kb.csv"
COLLECTION_NAME = "pharmacy_knowledge"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
LLM_MODEL = "llama3.2:3b"
QA_TOP_K = 3
PRODUCT_TOP_K = 4
THRESHOLD_OUT_OF_DOMAIN = 0.05
Ood_REJECTION_TEXT = (
    "Xin lỗi, tôi là trợ lý y tế của nhà thuốc Pharmacy nên chỉ có thể tư vấn các vấn đề về sức khỏe và thuốc men. "
    "Bạn đang gặp triệu chứng khó chịu nào cần tôi hỗ trợ không?"
)
SYSTEM_PROMPT = (
    "Bạn là trợ lý ảo Pharmabee. "
    "Chỉ được tạo một phản hồi ngắn gọn, mạch lạc, bằng tiếng Việt. "
    "Bắt buộc mở đầu đúng 1 lần duy nhất bằng câu: "
    "\"Tôi là trợ lý ảo Pharmabee. Các gợi ý dưới đây chỉ mang tính chất tham khảo và không thay thế chỉ định của bác sĩ.\" "
    "Không được lặp lại bất kỳ câu miễn trừ trách nhiệm nào ở cuối hoặc giữa câu trả lời. "
    "Sau câu mở đầu, chỉ được trả lời theo đúng cấu trúc: 'Dựa trên triệu chứng, bạn có thể đang gặp tình trạng [Tên bệnh có dấu]. Bạn có thể tham khảo một số sản phẩm hỗ trợ điều trị trong bảng dưới đây như [Tên thuốc đầu tiên]...'. "
    "Nếu không đủ thông tin, hãy hỏi tối đa 1 câu làm rõ. Không viết dài dòng, không lặp ý, không thêm đoạn kết luận thừa. "
    "TẤT YẾU: Nếu câu hỏi của khách hàng là lời chào hỏi xã giao thông thường, trêu đùa, kết bạn, hoặc KHÔNG liên quan gì đến y tế/sức khỏe/thuốc men, hãy từ chối lịch sự bằng đúng một câu: "
    f"\"{Ood_REJECTION_TEXT}\". Không trả lời gì thêm."
)



@dataclass
class RetrievedContext:
    document: str
    metadata: Dict[str, Any]
    score: float


class _RAGRuntime:
    _instance: Optional["_RAGRuntime"] = None

    def __new__(cls) -> "_RAGRuntime":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_func,
        )
        self.products_df = pd.read_csv(PRODUCTS_PATH)
        if "text" not in self.products_df.columns:
            raise ValueError("products_kb.csv must contain a 'text' column.")
        self.products_df["_search_clean"] = self.products_df["text"].fillna("").map(self._normalize_text)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)

    @staticmethod
    def _normalize_text(text: Any) -> str:
        normalized = str(text).lower()
        normalized = re.sub(r"[^\w\sÀ-ỹĐđ-]", " ", normalized, flags=re.UNICODE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _normalized_lookup_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("benh:", "")
        text = text.replace("-", " ")
        text = re.sub(r"\s+", "", text)
        return text

    def _normalize_medical_display_name(self, value: Any) -> str:
        try:
            from app.utils import normalize_display_name
            return normalize_display_name(value)
        except Exception:
            key = self._normalized_lookup_key(value)
            fallback_map = {
                "benh:apxephoi": "Áp-xe phổi",
                "benh:apxehaumon": "Áp-xe hậu môn",
                "benh:phidaituyentienliet": "Phì đại tuyến tiền liệt",
                "benh:viemxoangtran": "Viêm xoang trán",
            }
            return fallback_map.get(f"benh:{key}", fallback_map.get(key, str(value or "").strip()))

    def _extract_keywords_from_category(self, category: str) -> List[str]:
        cleaned = category.replace("BENH:", "").replace("benh:", "")
        cleaned = cleaned.replace("-", " ")
        normalized = self._normalize_text(cleaned)
        tokens = [tok for tok in re.split(r"\s+", normalized) if tok]
        if len(tokens) <= 1:
            compact = normalized.replace(" ", "")
            tokens = [compact[i : i + 4] for i in range(0, len(compact), 4) if compact[i : i + 4]]
        return list(dict.fromkeys(tokens))

    def _extract_keywords_from_display_name(self, display_name: str) -> List[str]:
        normalized = self._normalize_medical_display_name(display_name)
        cleaned = self._normalize_text(normalized)
        tokens = [tok for tok in re.split(r"\s+", cleaned) if tok]
        if len(tokens) > 1:
            return list(dict.fromkeys(tokens + [cleaned]))
        return [cleaned] if cleaned else []

    @staticmethod
    def _truncate_text(text: str, limit: int = 150) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) <= limit:
            return clean
        return clean[:limit].rstrip() + "..."

    @staticmethod
    def extract_features_from_text(text_content: Any) -> Dict[str, str]:
        raw_text = str(text_content or "")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        features = {
            "usage": "Đang cập nhật",
            "ingredients": "Đang cập nhật",
            "dosage": "Đang cập nhật",
            "price": "Liên hệ",
        }

        current_key: Optional[str] = None
        buffer: List[str] = []

        def flush_buffer() -> None:
            nonlocal current_key, buffer
            if current_key and buffer:
                value = " ".join(buffer).strip()
                if value:
                    features[current_key] = value
            current_key = None
            buffer = []

        def start_section(key: str, initial: str) -> None:
            nonlocal current_key, buffer
            flush_buffer()
            current_key = key
            buffer = [initial] if initial else []

        for line in lines:
            normalized_line = line.lstrip("- *•").strip()
            if normalized_line.startswith("Công dụng:"):
                start_section("usage", normalized_line.split("Công dụng:", 1)[1].strip())
                continue
            if normalized_line.startswith("Thành phần:"):
                start_section("ingredients", normalized_line.split("Thành phần:", 1)[1].strip())
                continue
            if normalized_line.startswith("Cách dùng:"):
                start_section("dosage", normalized_line.split("Cách dùng:", 1)[1].strip())
                continue
            if normalized_line.startswith("Giá:"):
                flush_buffer()
                features["price"] = normalized_line.split("Giá:", 1)[1].strip() or "Liên hệ"
                continue
            if normalized_line.startswith(("Tên sản phẩm:", "Nhóm:", "Nhóm bệnh:", "Mô tả:", "Tác dụng phụ:", "Chống chỉ định:")):
                flush_buffer()
                continue

            buffer.append(normalized_line) if current_key else None

        flush_buffer()
        for key in ("usage", "ingredients", "dosage"):
            features[key] = _RAGRuntime._truncate_text(features[key], 150)
        return features

    def diagnosis_phase(self, user_input: str) -> Tuple[Optional[str], Optional[str], bool, List[RetrievedContext]]:
        query_text = user_input
        qa_results = self.collection.query(
            query_texts=[query_text],
            n_results=QA_TOP_K,
            where={"type": "synthetic_qa"},
            include=["documents", "metadatas", "distances"],
        )
        docs = qa_results.get("documents", [[]])[0] or []
        metas = qa_results.get("metadatas", [[]])[0] or []
        pairs = [(user_input, doc) for doc in docs]
        scores = self.reranker.predict(pairs) if pairs else []
        contexts = [RetrievedContext(document=doc, metadata=meta or {}, score=float(score)) for doc, meta, score in zip(docs, metas, scores)]
        contexts.sort(key=lambda item: item.score, reverse=True)
        if not contexts:
            return None, None, True, []
        top = contexts[0]
        if top.score < THRESHOLD_OUT_OF_DOMAIN:
            return None, None, True, contexts
        category = str(top.metadata.get("category", "") or "")
        display_name = str(top.metadata.get("display_name", "") or "")
        return category, display_name, False, contexts

    def product_retrieval_phase(self, category: str, display_name: str) -> List[Dict[str, Any]]:
        if self.products_df.empty:
            return []

        normalized_display_name = self._normalize_medical_display_name(display_name)
        category_keywords = self._extract_keywords_from_category(category)
        display_keywords = self._extract_keywords_from_display_name(normalized_display_name)
        keywords = list(dict.fromkeys(category_keywords + display_keywords + self._extract_keywords_from_display_name(display_name)))
        if not keywords:
            return []

        scored_rows: List[Dict[str, Any]] = []
        for _, row in self.products_df.iterrows():
            clean_text = str(row.get("_search_clean", ""))
            if not clean_text:
                continue
            score = sum(1 for kw in keywords if kw and kw in clean_text)
            if score <= 0:
                continue
            features = self.extract_features_from_text(row.get("text", ""))
            scored_rows.append(
                {
                    "product_name": row.get("product_name"),
                    "category": row.get("category"),
                    "price": features.get("price", row.get("price", "Liên hệ")),
                    "score": score,
                    "usage": features.get("usage", row.get("usage", "Đang cập nhật")),
                    "ingredients": features.get("ingredients", "Đang cập nhật"),
                    "dosage": features.get("dosage", "Đang cập nhật"),
                    "text": row.get("text", ""),
                }
            )

        scored_rows.sort(key=lambda item: (-item["score"], str(item["product_name"] or "")))
        return scored_rows[:PRODUCT_TOP_K]

    def build_prompt(self, user_input: str, category: Optional[str], display_name: Optional[str], products: List[Dict[str, Any]], is_out_of_domain: bool) -> str:
        if is_out_of_domain:
            return Ood_REJECTION_TEXT

        product_lines = []
        for idx, item in enumerate(products[:4], start=1):
            product_lines.append(
                f"{idx}. {item.get('product_name', 'unknown')} | Nhóm: {item.get('category', 'N/A')} | Giá: {item.get('price', 'Liên hệ')} | Công dụng: {item.get('usage', 'Đang cập nhật')}"
            )
        product_text = "\n".join(product_lines) if product_lines else "Không có sản phẩm phù hợp"
        first_product = products[0].get('product_name', '') if products else ''
        disease_name = display_name or 'Không xác định'
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Tên bệnh có dấu: {disease_name}\n"
            f"Mã nhóm bệnh: {category or 'unknown'}\n"
            f"Sản phẩm gợi ý:\n{product_text}\n\n"
            f"Yêu cầu bắt buộc về đầu ra:\n"
            f"- Chỉ 1 đoạn ngắn, không lặp lại disclaimer.\n"
            f"- Không thêm câu kết kiểu 'hãy tham khảo...' ở cuối.\n"
            f"- Nếu có thể, nhắc tới sản phẩm đầu tiên là {first_product or 'một sản phẩm phù hợp'} ngay trong câu sau chẩn đoán.\n\n"
            f"Câu hỏi: {user_input}\n"
            f"Trả lời:"
        )

    def _postprocess_response(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return cleaned
        disclaimer = "Tôi là trợ lý ảo Pharmabee. Các gợi ý dưới đây chỉ mang tính chất tham khảo và không thay thế chỉ định của bác sĩ."
        tail_triggers = [
            r"vui lòng liên hệ với bác sĩ.*$",
            r"tuy nhiên, trước khi sử dụng.*$",
            r"tôi khuyên bạn nên đến gặp bác sĩ.*$",
            r"bạn nên tham khảo ý kiến bác sĩ.*$",
            r"hãy tham khảo ý kiến của bác sĩ.*$",
            r"khuyến cáo.*$",
        ]
        lower = cleaned.lower()
        for pattern in tail_triggers:
            match = re.search(pattern, lower, flags=re.IGNORECASE)
            if match and match.start() > len(disclaimer):
                cleaned = cleaned[: match.start()].rstrip(" ,;:-")
                lower = cleaned.lower()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        deduped: List[str] = []
        seen = set()
        for sentence in sentences:
            norm = sentence.lower().strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(sentence.strip())
        cleaned = " ".join(deduped).strip()
        if lower.startswith(disclaimer.lower()):
            cleaned = disclaimer + (" " + cleaned[len(disclaimer):].strip() if cleaned[len(disclaimer):].strip() else "")
        else:
            cleaned = disclaimer + (" " + cleaned if cleaned else "")
        return cleaned.strip()

    def answer(self, user_input: str, stream: bool = True) -> Dict[str, Any]:
        category, display_name, is_out_of_domain, _qa_contexts = self.diagnosis_phase(user_input)
        normalized_display_name = self._normalize_medical_display_name(display_name or "") if display_name else display_name
        products = [] if is_out_of_domain else self.product_retrieval_phase(category or "", normalized_display_name or "")
        prompt = self.build_prompt(user_input, category, normalized_display_name, products, is_out_of_domain)
        response_text: List[str] = []

        if is_out_of_domain:
            print("🤖 Pharmacy RAG:", Ood_REJECTION_TEXT)
            response_text.append(Ood_REJECTION_TEXT)
        elif stream:
            print("🤖 Pharmacy RAG:", end=" ", flush=True)
            for chunk in self.llm.stream(prompt):
                print(chunk, end="", flush=True)
                response_text.append(chunk)
            print()
        else:
            response = "".join(list(self.llm.stream(prompt)))
            response_text.append(response)

        final_response = self._postprocess_response("".join(response_text))
        return {
            "category": category,
            "display_name": normalized_display_name,
            "is_out_of_domain": is_out_of_domain,
            "response": final_response,
            "products": products,
        }


_RUNTIME = _RAGRuntime()


def predict_category(user_input: str) -> str:
    category, _display_name, _is_out_of_domain, _contexts = _RUNTIME.diagnosis_phase(user_input)
    return category or ""


def answer(user_input: str, stream: bool = True) -> Dict[str, Any]:
    return _RUNTIME.answer(user_input, stream=stream)


def main() -> None:
    print("=" * 100)
    print("PHARMACY RAG ADVANCED V8.0")
    print("=" * 100)
    while True:
        try:
            user_input = input("\n👤 Khách hàng: ").strip()
        except KeyboardInterrupt:
            print("\nĐã thoát.")
            break
        if user_input.lower() in {"exit", "quit", "thoát"}:
            print("Cảm ơn bạn đã sử dụng Pharmacy RAG!")
            break
        if not user_input:
            continue
        try:
            start_total = time.perf_counter()
            result = answer(user_input, stream=True)
            total_latency = time.perf_counter() - start_total
            print("\n--- 💊 CÁC SẢN PHẨM GỢI Ý ---")
            if result["products"]:
                for item in result["products"]:
                    print(f"📦 {item['product_name']}")
                    print(f"🌟 Nhóm: {item['category']}")
                    print(f"🎯 Công dụng: {item.get('usage', 'Đang cập nhật')}")
                    print(f"💰 Giá: {item['price']}")
                    print()
            else:
                print("Không có sản phẩm phù hợp")
            print(f"⏱️  Latency: {total_latency:.2f}s")
        except Exception as exc:
            print(f"❌ Lỗi: {exc}")


if __name__ == "__main__":
    main()

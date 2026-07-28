from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Union

from .safety_checker import DEBUG_SAFETY, SafetyChecker, WARNING_QUESTION_MAP, WARNING_TYPE_PRIORITY
from .user_session import SessionManager, UserProfile
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SYNTHETIC_QA = "../../../data/silver/synthetic_medical_qa.json"
DISEASE_MAP = "../../../data/silver/disease_category_map.csv"
CATEGORIES_CONFIG = "../../../data/silver/categories_config.json"
PRODUCTS_CLEANED = "../../../data/silver/products_cleaned_backup.csv"
REQUIRES_MEDICAL_VISIT = "REQUIRES_MEDICAL_VISIT"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_PRODUCTS = 5
MAX_SAFETY_CANDIDATES = 20

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 3
EMBEDDING_WEIGHT = 0.8
KEYWORD_WEIGHT = 0.2
CONFIDENCE_WARNING_THRESHOLD = 0.38
VIETNAMESE_STOPWORDS = [
    "dạo này",
    "em bị",
    "tôi bị",
    "bác sĩ ơi",
    "dược sĩ ơi",
    "cho em hỏi",
    "có phải là",
    "không ạ",
    "bị gì ạ",
    "ạ",
    "với ạ",
    "như thế nào",
]
STOPWORD_PATTERNS = [
    r"dạo này",
    r"dao nay",
    r"em bị",
    r"em bi",
    r"tôi bị",
    r"toi bi",
    r"bác sĩ ơi",
    r"bac si oi",
    r"dược sĩ ơi",
    r"duoc si oi",
    r"cho em hỏi",
    r"cho em hoi",
    r"có phải là",
    r"co phai la",
    r"không ạ",
    r"khong a",
    r"bị gì ạ",
    r"bi gi a",
    r"với ạ",
    r"voi a",
    r"dạ",
    r"da",
    r"cho hỏi",
    r"cho hoi",
    r"và",
    r"em",
    r"cứ",
    r"bị",
]
QUERY_SYNONYM_MAP: Dict[str, List[str]] = {
    "sốt": ["sốt", "sốt nhẹ", "sốt cao", "cảm sốt", "nóng sốt"],
    "đau họng": ["đau họng", "rát họng", "ngứa họng", "khô họng", "khó nuốt"],
    "viêm họng": ["viêm họng", "đau họng", "rát họng", "khó nuốt"],
    "sổ mũi": ["sổ mũi", "chảy mũi", "nghẹt mũi", "hắt hơi"],
    "nghẹt mũi": ["nghẹt mũi", "sổ mũi", "chảy mũi", "hắt hơi"],
    "hắt hơi": ["hắt hơi", "sổ mũi", "nghẹt mũi", "chảy mũi"],
    "ho": ["ho", "ho khan", "ho có đờm", "ho đêm", "khạc đờm"],
    "ho khan": ["ho khan", "ho", "cơn ho khan", "ho rát cổ"],
    "ho có đờm": ["ho có đờm", "ho đờm", "khạc đờm", "đờm"],
    "đau đầu": ["đau đầu", "nhức đầu", "đầu đau"],
    "đau bụng": ["đau bụng", "đau quặn bụng", "đầy bụng", "khó tiêu", "buồn nôn"],
    "tiêu chảy": ["tiêu chảy", "đi ngoài", "rối loạn tiêu hóa", "đau bụng đi ngoài"],
}


class AdvancedRAGLoadError(RuntimeError):
    pass


def _safe_debug_text(value: Any) -> str:
    text = str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _debug_print(value: Any) -> None:
    print(_safe_debug_text(value))


class RAGPredictor:
    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.session_manager = SessionManager()
        self.safety_checker = SafetyChecker()
        self.categories_map = self._load_categories_config()
        self.disease_df = self._load_disease_map()
        self.products_df = self._load_products_df()
        self.qa_payload = self._load_json(self._resolve_data_path(SYNTHETIC_QA))

        self.disease_knowledge_chunks = self._build_disease_knowledge_chunks()
        self.disease_embeddings = self._encode_texts([chunk["text_chunk"] for chunk in self.disease_knowledge_chunks])
        self.disease_code_set = {
            str(code).strip()
            for code in self.disease_df["disease_code"].tolist()
            if not self._is_missing(code)
        }
        self.product_chunks, self.product_metadata = self._build_product_cache()
        self.product_embeddings = self._encode_texts([chunk["text_chunk"] for chunk in self.product_chunks])
        self.session_manager = SessionManager()
        self.safety_checker = SafetyChecker()
        self._trace_cache_query: str | None = None
        self._trace_cache: Dict[str, Any] | None = None

    @staticmethod
    def _module_dir() -> Path:
        return Path(__file__).resolve().parent

    def _resolve_data_path(self, relative_path: str) -> Path:
        return (self._module_dir() / relative_path).resolve()

    @staticmethod
    def _normalize_text(text: object) -> str:
        cleaned = str(text).strip()
        cleaned = re.sub(r"[^\w\sà-ỹÀ-Ỹ-]+", " ", cleaned, flags=re.UNICODE)
        cleaned = re.sub(r"[-_]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned.lower()

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

    def _load_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise AdvancedRAGLoadError(f"{path.name} phải là dictionary")
        return payload

    def _load_disease_map(self) -> pd.DataFrame:
        df = pd.read_csv(self._resolve_data_path(DISEASE_MAP)).fillna("")
        required_cols = {"disease_code", "category_key"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise AdvancedRAGLoadError(f"disease_category_map.csv thiếu cột: {', '.join(sorted(missing))}")

        for col in ["disease_code", "category_key", "disease_name"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df

    def _load_categories_config(self) -> Dict[str, str]:
        payload = self._load_json(self._resolve_data_path(CATEGORIES_CONFIG))
        return {str(key).strip(): str(value).strip() for key, value in payload.items()}

    def _load_products_df(self) -> pd.DataFrame:
        df = pd.read_csv(self._resolve_data_path(PRODUCTS_CLEANED)).fillna("")
        # canonical_ingredients is expected to exist in the backup dataset after build.
        required_cols = {"category_key", "product_name"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise AdvancedRAGLoadError(f"products_cleaned.csv thiếu cột: {', '.join(sorted(missing))}")

        for col in ["sku", "category_key", "product_name", "usage", "description"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df

    def _split_category_tokens(self, value: object) -> List[str]:
        if self._is_missing(value):
            return []
        return [token.strip() for token in str(value).strip().split() if token.strip()]

    def _clean_question_text(self, text: object) -> str:
        cleaned = str(text).strip()
        if not cleaned:
            return ""

        for pattern in STOPWORD_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = [token for token in cleaned.split() if len(token) > 1]
        return " ".join(tokens)

    def _tokenize(self, text: object) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        tokens = [token for token in re.split(r"\s+", normalized) if token and token not in {"and", "or"}]
        return tokens

    def _expand_query(self, query_text: str) -> str:
        normalized = self._normalize_text(query_text)
        fragments: List[str] = [normalized]
        for key, synonyms in QUERY_SYNONYM_MAP.items():
            if key in normalized:
                fragments.extend(synonyms)
        seen: set[str] = set()
        ordered_fragments: List[str] = []
        for fragment in fragments:
            fragment_norm = self._normalize_text(fragment)
            if not fragment_norm or fragment_norm in seen:
                continue
            seen.add(fragment_norm)
            ordered_fragments.append(fragment_norm)
        return " | ".join(ordered_fragments)

    def _query_term_set(self, query_text: str) -> set[str]:
        expanded = self._expand_query(query_text)
        terms = set(self._tokenize(expanded))
        for phrase, synonyms in QUERY_SYNONYM_MAP.items():
            if phrase in expanded:
                terms.update(self._tokenize(phrase))
                for synonym in synonyms:
                    terms.update(self._tokenize(synonym))
        return {term for term in terms if len(term) > 1}

    def _symptom_phrase_set(self, query_text: str) -> set[str]:
        normalized = self._normalize_text(query_text)
        phrases = set()
        for phrase in QUERY_SYNONYM_MAP:
            if phrase in normalized:
                phrases.add(phrase)
        return phrases

    def _build_disease_knowledge_chunks(self) -> List[Dict[str, str]]:
        qa_index: Dict[str, List[str]] = {}
        for disease_code, item in self.qa_payload.items():
            if not str(disease_code).startswith("BENH:"):
                continue
            if not isinstance(item, dict):
                continue
            questions = item.get("questions", [])
            if not isinstance(questions, list):
                continue
            question_texts = [self._clean_question_text(question) for question in questions]
            question_texts = [question for question in question_texts if question]
            if question_texts:
                qa_index[str(disease_code).strip()] = question_texts

        chunks: List[Dict[str, str]] = []
        for disease_code, questions in qa_index.items():
            matched_row = self.disease_df.loc[self.disease_df["disease_code"] == disease_code]
            if matched_row.empty:
                disease_name = disease_code.split(":", 1)[-1].replace("-", " ").replace("_", " ")
                category_key_str = ""
            else:
                disease_name = str(matched_row.iloc[0].get("disease_name", "")).strip()
                if self._is_missing(disease_name):
                    disease_name = disease_code.split(":", 1)[-1].replace("-", " ").replace("_", " ")
                category_key_str = str(matched_row.iloc[0].get("category_key", "")).strip()

            category_tokens = self._split_category_tokens(category_key_str)
            category_vn_names = [self.categories_map.get(cat_key, cat_key) for cat_key in category_tokens]
            category_text = "; ".join(dict.fromkeys(category_vn_names)) if category_vn_names else ""
            merged_questions = ", ".join(questions)
            query_hint_terms = []
            for term in self._tokenize(disease_name):
                if len(term) > 1:
                    query_hint_terms.append(term)
            for question in questions[:5]:
                query_hint_terms.extend(self._tokenize(question))
            for key, synonyms in QUERY_SYNONYM_MAP.items():
                if key in disease_name.lower():
                    query_hint_terms.extend(self._tokenize(key))
                    for synonym in synonyms:
                        query_hint_terms.extend(self._tokenize(synonym))
            query_hint_text = " ".join(dict.fromkeys(query_hint_terms))
            text_chunk = (
                f"Bệnh lý: Tên tiếng Việt là {disease_name} {disease_code}. "
                f"Thuộc nhóm thuốc điều trị: {category_text}. "
                f"Các dấu hiệu, triệu chứng và câu hỏi lâm sàng điển hình: {merged_questions}. "
                f"Từ khóa gợi ý: {query_hint_text}."
            )
            chunks.append(
                {
                    "disease_code": disease_code,
                    "disease_name": disease_name,
                    "category_key": category_key_str,
                    "category_text": category_text,
                    "query_hint_text": query_hint_text,
                    "text_chunk": text_chunk,
                    "tokens": " ".join(dict.fromkeys(self._tokenize(text_chunk + " " + query_hint_text))),
                }
            )

        if not chunks:
            raise AdvancedRAGLoadError("Không tạo được knowledge chunks bệnh lý hợp lệ")
        return chunks

    def _build_product_cache(self) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        chunks: List[Dict[str, str]] = []
        metadata: List[Dict[str, str]] = []
        seen_skus: set[str] = set()
        for _, row in self.products_df.iterrows():
            product_name = str(row.get("product_name", "")).strip()
            category_key = str(row.get("category_key", "")).strip()
            sku = str(row.get("sku", "")).strip()
            usage = str(row.get("usage", "")).strip()
            description = str(row.get("description", "")).strip()
            category_name = self.categories_map.get(category_key, category_key)
            text_chunk = (
                f"Sản phẩm: {product_name}. Thuộc nhóm danh mục: {category_name}. "
                f"Công dụng: {usage}. Mô tả tóm tắt: {description}"
            )
            dedupe_key = sku or product_name
            if self._is_missing(product_name) or dedupe_key in seen_skus:
                continue
            seen_skus.add(dedupe_key)
            chunks.append({
                "sku": sku,
                "product_name": product_name,
                "category_key": category_key,
                "category_name": category_name,
                "usage": usage,
                "description": description,
                "text_chunk": text_chunk,
            })
            metadata.append({
                "sku": sku,
                "product_name": product_name,
                "category_key": category_key,
                "category_name": category_name,
            })
        return chunks, metadata

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

    def _embed_expanded_query(self, query_text: str) -> np.ndarray:
        embedding = self.embedding_model.encode(
            [self._expand_query(query_text)],
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

    def _call_ollama(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "top_p": 0.1},
            }
        ).encode("utf-8")

        try:
            req = urlrequest.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            response_text = str(data.get("response", "")).strip()
            return response_text or "None"
        except (urlerror.URLError, urlerror.HTTPError, TimeoutError, json.JSONDecodeError, Exception):
            return "None"

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 0.80:
            return "Cao"
        if confidence >= 0.60:
            return "Trung bình"
        return "Thấp"

    def _default_advice(self, disease_code: str) -> str:
        disease_code = str(disease_code or "").strip().lower()
        advice_map = {
            "benh:viemhong": "Nghỉ ngơi, uống nhiều nước. Nếu sốt cao kéo dài, khó thở hoặc đau ngực hãy đến cơ sở y tế.",
            "benh:daubung": "Theo dõi triệu chứng. Nếu đau dữ dội hoặc kéo dài hãy đi khám.",
            "benh:traonguocdaday": "Hạn chế đồ cay nóng, không nằm ngay sau ăn và theo dõi nếu triệu chứng kéo dài hãy đi khám.",
            "benh:camcum": "Nghỉ ngơi, uống đủ nước và theo dõi triệu chứng. Nếu sốt cao kéo dài hoặc khó thở hãy đi khám.",
        }
        return advice_map.get(disease_code, "Nếu triệu chứng kéo dài, trở nặng, khó thở hoặc đau ngực, hãy đến cơ sở y tế để được khám.")

    def _default_clinical_summary(self, disease_name: str, chunk: Dict[str, str] | None) -> str:
        disease_name = disease_name.strip() or "bệnh phù hợp"
        snippet = ""
        if chunk is not None:
            hints = [chunk.get("disease_name", ""), chunk.get("query_hint_text", "")]
            hints = [self._normalize_text(item) for item in hints if self._normalize_text(item)]
            if hints:
                snippet = hints[0]
        if snippet:
            return f"Triệu chứng của bạn phù hợp với {disease_name}. Đây thường liên quan đến các dấu hiệu như {snippet}."
        return f"Triệu chứng của bạn phù hợp với {disease_name}. Đây là gợi ý tham khảo từ dữ liệu bệnh đã truy xuất."

    def _build_reasoning_text(self, query_text: str, chunk: Dict[str, str] | None, disease_name: str) -> str:
        query_terms = self._query_term_set(query_text)
        chunk_terms = set(self._tokenize((chunk or {}).get("text_chunk", ""))) if chunk is not None else set()
        matched_terms = [term for term in query_terms if term in chunk_terms]
        matched_terms = list(dict.fromkeys(matched_terms))[:4]
        if matched_terms:
            bullets = "\n".join(f"• {term}" for term in matched_terms)
            return f"Bệnh được đề xuất vì câu hỏi có các triệu chứng:\n{bullets}\nđồng thời phù hợp với tri thức đã truy xuất cho {disease_name}."
        return f"Bệnh được đề xuất vì câu hỏi và dữ liệu truy xuất có mức tương đồng cao với {disease_name}."

    @staticmethod
    def _safe_json_loads(raw_text: str) -> Dict[str, Any] | None:
        if not raw_text:
            return None
        candidate_text = raw_text.strip()
        if candidate_text.startswith("```"):
            candidate_text = re.sub(r"^```(?:json)?\s*", "", candidate_text, flags=re.IGNORECASE)
            candidate_text = re.sub(r"\s*```$", "", candidate_text)
        try:
            parsed = json.loads(candidate_text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _score_keyword_overlap(self, query_terms: set[str], chunk: Dict[str, str]) -> float:
        if not query_terms:
            return 0.0
        chunk_terms = set(self._tokenize(chunk.get("text_chunk", "")))
        chunk_terms.update(self._tokenize(chunk.get("disease_name", "")))
        chunk_terms.update(self._tokenize(chunk.get("category_text", "")))
        chunk_terms.update(self._tokenize(chunk.get("query_hint_text", "")))
        if not chunk_terms:
            return 0.0

        overlap = query_terms.intersection(chunk_terms)
        if not overlap:
            return 0.0

        overlap_ratio = len(overlap) / max(len(query_terms), 1)
        return float(min(overlap_ratio, 1.0))

    def _hybrid_retrieve_disease_candidates(self, query_text: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Union[str, float]]]:
        query_terms = self._query_term_set(query_text)
        query_phrases = self._symptom_phrase_set(query_text)
        query_vec = self._embed_expanded_query(query_text)
        if self.disease_embeddings.size == 0:
            return []

        embedding_similarities = self._cosine_similarity(query_vec, self.disease_embeddings)
        candidates: List[Dict[str, Union[str, float]]] = []
        for idx, chunk in enumerate(self.disease_knowledge_chunks):
            embedding_similarity = float(embedding_similarities[int(idx)]) if idx < len(embedding_similarities) else 0.0
            keyword_similarity = self._score_keyword_overlap(query_terms, chunk)
            hybrid_score = float(EMBEDDING_WEIGHT * embedding_similarity + KEYWORD_WEIGHT * keyword_similarity)
            candidates.append(
                {
                    "disease_code": chunk["disease_code"],
                    "disease_name": chunk["disease_name"],
                    "category_key": chunk["category_key"],
                    "category_text": chunk["category_text"],
                    "text_chunk": chunk["text_chunk"],
                    "embedding_similarity": round(embedding_similarity, 6),
                    "keyword_similarity": round(keyword_similarity, 6),
                    "keyword_overlap": round(keyword_similarity, 6),
                    "exact_symptom_boost": 0.0,
                    "hybrid_score": round(hybrid_score, 6),
                    "query_phrases": sorted(query_phrases),
                }
            )

        candidates.sort(
            key=lambda item: (
                float(item["hybrid_score"]),
                float(item["keyword_similarity"]),
                float(item["embedding_similarity"]),
                str(item["disease_code"]),
            ),
            reverse=True,
        )
        return candidates[:top_k]

    def _rerank_candidates(self, query_text: str, candidates: List[Dict[str, Union[str, float]]], top_k: int = RERANK_TOP_K) -> List[Dict[str, Union[str, float]]]:
        query_terms = self._query_term_set(query_text)
        query_phrases = self._symptom_phrase_set(query_text)
        reranked: List[Dict[str, Union[str, float]]] = []
        for rank, candidate in enumerate(candidates, start=1):
            disease_terms = set(self._tokenize(candidate.get("disease_name", "")))
            category_terms = set(self._tokenize(candidate.get("category_text", "")))
            chunk_terms = set(self._tokenize(candidate.get("text_chunk", "")))
            token_union = chunk_terms.union(disease_terms).union(category_terms)
            if not query_terms:
                query_overlap = 0.0
            else:
                query_overlap = len(query_terms.intersection(token_union)) / max(len(query_terms), 1)
            disease_name = self._normalize_text(candidate.get("disease_name", ""))
            exact_symptom_boost = 0.0
            for phrase in query_phrases:
                if phrase in disease_name or phrase in self._normalize_text(candidate.get("text_chunk", "")):
                    exact_symptom_boost += 0.04
            exact_symptom_boost = min(exact_symptom_boost, 0.12)
            rerank_score = float(
                0.75 * float(candidate.get("hybrid_score", 0.0))
                + 0.15 * query_overlap
                + 0.10 * exact_symptom_boost
            )
            cand = dict(candidate)
            cand["exact_symptom_boost"] = round(exact_symptom_boost, 6)
            cand["rerank_score"] = round(rerank_score, 6)
            cand["query_overlap"] = round(query_overlap, 6)
            cand["rank"] = rank
            reranked.append(cand)

        reranked.sort(
            key=lambda item: (
                float(item["rerank_score"]),
                float(item["hybrid_score"]),
                float(item["keyword_similarity"]),
                str(item["disease_code"]),
            ),
            reverse=True,
        )
        return reranked[:top_k]

    def _build_detection_prompt(self, query_text: str, reranked_candidates: List[Dict[str, Union[str, float]]]) -> str:
        context_lines: List[str] = []
        for idx, candidate in enumerate(reranked_candidates, start=1):
            context_lines.append(
                f"Candidate {idx}\n"
                f"Disease Code: {candidate['disease_code']}\n"
                f"Disease Name: {candidate['disease_name']}\n"
                f"Category: {candidate['category_text']}\n"
                f"Knowledge: {candidate.get('text_chunk', '')}\n"
                f"Embedding Similarity: {candidate['embedding_similarity']}\n"
                f"Keyword Similarity: {candidate['keyword_similarity']}"
            )
        context = "\n\n".join(context_lines)
        return f"""
Bạn là hệ thống phân loại triệu chứng và tạo giải thích y khoa cho chatbot OTC bằng Tiếng Việt.

CHỈ ĐƯỢC CHỌN MỘT MÃ BỆNH TRONG DANH SÁCH SAU:
{context}

TRIỆU CHỨNG THỰC TẾ:
"{query_text}"

NHIỆM VỤ:
- Chỉ trả JSON hợp lệ.
- Không sinh thuốc.
- Không sinh liều dùng.
- Không sinh tác dụng phụ.
- Không suy diễn ngoài context.

ĐỊNH DẠNG JSON BẮT BUỘC:
{{
  "disease_code": "BENH:<mã_bệnh> hoặc None",
  "clinical_summary": "Tối đa 3 câu, ngắn gọn",
  "reasoning": "Giải thích dựa trên triệu chứng và context truy xuất, tối đa 3 câu",
  "advice": "Khuyến nghị ngắn, an toàn"
}}

RÀNG BUỘC:
1. Chỉ chọn trong danh sách được cung cấp ở trên. Tuyệt đối không suy diễn ra mã bệnh ngoài context.
2. Ưu tiên mã bệnh có triệu chứng khớp chặt nhất với câu hỏi của bệnh nhân.
3. Nếu không có lựa chọn nào đủ thuyết phục, hãy để disease_code là None.
4. clinical_summary phải ngắn, không quá 3 câu.
5. reasoning phải dựa trên triệu chứng query và retrieved context, không được bịa.
6. advice phải ngắn, an toàn, không thay thế bác sĩ.
7. Không dùng markdown, không giải thích thêm ngoài JSON.
""".strip()

    def _find_disease_chunk(self, disease_code: str) -> Dict[str, str] | None:
        for chunk in self.disease_knowledge_chunks:
            if str(chunk.get("disease_code", "")).strip() == str(disease_code).strip():
                return chunk
        return None

    def _build_disease_summary(
        self,
        query_text: str,
        disease_code: str,
        confidence: float,
        low_confidence: bool,
        detected_reason: str,
    ) -> str:
        if self._is_missing(disease_code) or disease_code == "None":
            return "Chưa xác định được bệnh cụ thể từ truy vấn hiện tại."

        chunk = self._find_disease_chunk(disease_code)
        matched_row = self.disease_df.loc[self.disease_df["disease_code"] == str(disease_code).strip()]
        disease_name = "Không rõ tên bệnh"
        category_text = ""
        if not matched_row.empty:
            disease_name = str(matched_row.iloc[0].get("disease_name", "")).strip() or disease_name
            category_key = str(matched_row.iloc[0].get("category_key", "")).strip()
            if category_key:
                category_text = "; ".join(
                    dict.fromkeys([self.categories_map.get(key.strip(), key.strip()) for key in category_key.split() if key.strip()])
                )
        if chunk is not None:
            disease_name = str(chunk.get("disease_name", disease_name)).strip() or disease_name
            category_text = str(chunk.get("category_text", category_text)).strip() or category_text

        query_terms = self._query_term_set(query_text)
        chunk_terms = set(self._tokenize((chunk or {}).get("text_chunk", ""))) if chunk is not None else set()
        matched_terms = [term for term in query_terms if term in chunk_terms]
        matched_terms = list(dict.fromkeys(matched_terms))[:4]

        summary_lines = [
            f"Hệ thống nhận diện khả năng cao bạn đang gặp: {disease_name}.",
            "",
            "Lý do:",
        ]
        if matched_terms:
            summary_lines.append(f"• Triệu chứng phù hợp với dữ liệu bệnh: {', '.join(matched_terms)}.")
        elif chunk is not None and str(chunk.get("query_hint_text", "")).strip():
            summary_lines.append(f"• Dữ liệu bệnh có các dấu hiệu liên quan: {chunk.get('query_hint_text', '')}.")
        else:
            summary_lines.append("• Dữ liệu bệnh trong knowledge base phù hợp với truy vấn sau khi mở rộng từ khóa.")

        if category_text:
            summary_lines.append(f"• Bệnh nằm trong nhóm điều trị: {category_text}.")

        if detected_reason == "llm_selected":
            summary_lines.append("• Mô hình ngôn ngữ đã chọn bệnh này trong nhóm ứng viên được truy xuất.")
        elif detected_reason == "fallback_top1":
            summary_lines.append("• Hệ thống dùng bệnh có điểm tổng hợp cao nhất vì LLM không chọn được mã bệnh rõ ràng.")

        if low_confidence:
            summary_lines.append(
                f"• Độ tin cậy chưa cao (confidence={confidence:.2f}), vì vậy bạn nên theo dõi thêm triệu chứng."
            )
            summary_lines.append(
                "Nếu triệu chứng kéo dài, sốt cao, khó thở, đau ngực hoặc nặng lên, hãy đi khám bác sĩ."
            )
        else:
            summary_lines.append("• Độ tin cậy đủ tốt để gợi ý nhóm bệnh OTC nếu không có dấu hiệu nguy hiểm.")

        return "\n".join(summary_lines)

    def _evaluate_confidence(
        self,
        reranked_candidates: List[Dict[str, Union[str, float]]],
        llm_code: str,
    ) -> float:
        if not reranked_candidates:
            return 0.0
        top1 = reranked_candidates[0]
        top1_score = float(top1.get("rerank_score", top1.get("hybrid_score", 0.0)))
        top2_score = float(reranked_candidates[1].get("rerank_score", reranked_candidates[1].get("hybrid_score", 0.0))) if len(reranked_candidates) > 1 else 0.0
        gap = max(top1_score - top2_score, 0.0)
        llm_bonus = 0.0
        top_codes = [str(item.get("disease_code", "")) for item in reranked_candidates]
        if llm_code == top_codes[0]:
            llm_bonus = 0.22
        elif llm_code in top_codes[:RERANK_TOP_K]:
            llm_bonus = 0.12
        elif llm_code == "None":
            llm_bonus = -0.12
        else:
            llm_bonus = -0.18
        confidence = 0.62 * top1_score + 0.28 * min(gap + 0.15, 1.0) + llm_bonus
        return float(max(min(confidence, 1.0), 0.0))

    def _trace_disease_detection(self, query_text: str) -> Dict[str, Any]:
        if self._trace_cache_query == query_text and self._trace_cache is not None:
            return self._trace_cache

        total_start = time.perf_counter()
        query_expansion_start = time.perf_counter()
        expanded_query = self._expand_query(query_text)
        query_expansion_ms = (time.perf_counter() - query_expansion_start) * 1000.0

        retrieval_start = time.perf_counter()
        retrieved_top10 = self._hybrid_retrieve_disease_candidates(query_text, top_k=RETRIEVAL_TOP_K)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        rerank_start = time.perf_counter()
        reranked_top3 = self._rerank_candidates(query_text, retrieved_top10, top_k=RERANK_TOP_K)
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        chunk = self._find_disease_chunk(str(reranked_top3[0].get("disease_code", "None"))) if reranked_top3 else None

        llm_start = time.perf_counter()
        if not reranked_top3:
            llm_ms = (time.perf_counter() - llm_start) * 1000.0
            trace = {
                "query_text": query_text,
                "expanded_query": expanded_query,
                "retrieved_top10": retrieved_top10,
                "reranked_top3": reranked_top3,
                "llm_disease_code": "None",
                "detected_disease_code": "None",
                "final_disease_code": "None",
                "clinical_summary": "Chưa xác định được bệnh cụ thể từ truy vấn hiện tại.",
                "reasoning": "",
                "advice": self._default_advice(""),
                "confidence": 0.0,
                "confidence_label": self._confidence_label(0.0),
                "confidence_threshold": CONFIDENCE_WARNING_THRESHOLD,
                "low_confidence": True,
                "latency": {
                    "query_expansion_ms": round(query_expansion_ms, 2),
                    "retrieval_ms": round(retrieval_ms, 2),
                    "rerank_ms": round(rerank_ms, 2),
                    "llm_ms": round(llm_ms, 2),
                    "confidence_ms": 0.0,
                    "product_ms": 0.0,
                    "total_ms": round((time.perf_counter() - total_start) * 1000.0, 2),
                },
            }
            self._trace_cache_query = query_text
            self._trace_cache = trace
            return trace

        prompt = self._build_detection_prompt(query_text, reranked_top3)
        result_text = self._call_ollama(prompt).strip()
        llm_ms = (time.perf_counter() - llm_start) * 1000.0
        parsed = self._safe_json_loads(result_text)

        llm_code = "None"
        clinical_summary = ""
        reasoning = ""
        advice = ""
        if parsed is not None:
            llm_code = str(parsed.get("disease_code", "None")).strip() or "None"
            clinical_summary = str(parsed.get("clinical_summary", "")).strip()
            reasoning = str(parsed.get("reasoning", "")).strip()
            advice = str(parsed.get("advice", "")).strip()

        if llm_code not in self.disease_code_set:
            llm_code = "None"

        top_codes = [str(item.get("disease_code", "")) for item in reranked_top3]
        detected_code = "None"
        detected_reason = "none"
        if llm_code != "None" and llm_code in self.disease_code_set:
            detected_code = llm_code
            detected_reason = "llm_selected"
        elif top_codes:
            detected_code = top_codes[0]
            detected_reason = "fallback_top1"

        confidence_start = time.perf_counter()
        confidence = self._evaluate_confidence(reranked_top3, llm_code if llm_code else "None")
        confidence_ms = (time.perf_counter() - confidence_start) * 1000.0
        low_confidence = confidence < CONFIDENCE_WARNING_THRESHOLD
        disease_chunk = self._find_disease_chunk(detected_code)
        disease_name = "Không rõ tên bệnh"
        if disease_chunk is not None:
            disease_name = disease_chunk.get("disease_name", disease_name) or disease_name

        if not clinical_summary:
            clinical_summary = self._default_clinical_summary(disease_name, disease_chunk)
        if not reasoning:
            reasoning = self._build_reasoning_text(query_text, disease_chunk, disease_name)
        advice = self._default_advice(detected_code)

        trace = {
            "query_text": query_text,
            "expanded_query": expanded_query,
            "retrieved_top10": retrieved_top10,
            "reranked_top3": reranked_top3,
            "llm_disease_code": llm_code,
            "detected_disease_code": detected_code,
            "final_disease_code": detected_code,
            "clinical_summary": clinical_summary,
            "reasoning": reasoning,
            "advice": advice,
            "confidence": round(confidence, 6),
            "confidence_label": self._confidence_label(confidence),
            "confidence_threshold": CONFIDENCE_WARNING_THRESHOLD,
            "low_confidence": low_confidence,
            "latency": {
                "query_expansion_ms": round(query_expansion_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
                "llm_ms": round(llm_ms, 2),
                "confidence_ms": round(confidence_ms, 2),
                "product_ms": 0.0,
                "total_ms": round((time.perf_counter() - total_start) * 1000.0, 2),
            },
        }
        self._trace_cache_query = query_text
        self._trace_cache = trace
        return trace

    def detect_disease_code(self, query_text: str) -> str:
        trace = self._trace_disease_detection(query_text)
        return str(trace.get("final_disease_code", "None"))

    def debug_trace_query(self, query_text: str) -> Dict[str, Any]:
        trace = self._trace_disease_detection(query_text)
        final_code = str(trace.get("final_disease_code", "None"))
        final_products: Union[str, List[str]] = REQUIRES_MEDICAL_VISIT
        product_start = time.perf_counter()
        if final_code != "None":
            final_products = self._rank_products_for_disease(query_text, final_code)
        product_ms = (time.perf_counter() - product_start) * 1000.0
        trace = dict(trace)
        trace["final_products"] = final_products
        trace.setdefault("clinical_summary", "Không có thông tin.")
        trace.setdefault("reasoning", "Không có thông tin.")
        trace.setdefault("advice", self._default_advice(final_code))
        trace.setdefault("confidence_label", self._confidence_label(float(trace.get("confidence", 0.0))))
        latency = dict(trace.get("latency", {}))
        latency["product_ms"] = round(product_ms, 2)
        trace["latency"] = latency
        return trace

    def _rank_products_for_disease(self, query_text: str, disease_code: str) -> Union[str, List[str]]:
        if disease_code == "None" or self._is_missing(disease_code):
            return REQUIRES_MEDICAL_VISIT

        matched_row = self.disease_df.loc[self.disease_df["disease_code"] == disease_code]
        if matched_row.empty:
            return REQUIRES_MEDICAL_VISIT

        category_key_str = matched_row["category_key"].values[0]
        if self._is_missing(category_key_str):
            return REQUIRES_MEDICAL_VISIT

        target_categories = [cat.strip() for cat in str(category_key_str).strip().split() if cat.strip()]
        if not target_categories:
            return REQUIRES_MEDICAL_VISIT

        candidate_indices = [
            idx for idx, item in enumerate(self.product_metadata)
            if item.get("category_key", "") in target_categories
        ]
        if not candidate_indices:
            return REQUIRES_MEDICAL_VISIT

        query_vec = self._embed_query(query_text)
        product_embeddings = self.product_embeddings[candidate_indices]
        if product_embeddings.size == 0:
            return REQUIRES_MEDICAL_VISIT

        similarities = self._cosine_similarity(query_vec, product_embeddings)
        ranked_indices = np.argsort(-similarities)

        ranked_products: List[str] = []
        seen_ranked: set[str] = set()
        for idx in ranked_indices:
            meta = self.product_metadata[candidate_indices[int(idx)]]
            product_name = meta.get("product_name", "")
            dedupe_key = meta.get("sku", "") or product_name
            if not product_name or dedupe_key in seen_ranked:
                continue
            seen_ranked.add(dedupe_key)
            ranked_products.append(product_name)

        if not ranked_products:
            return REQUIRES_MEDICAL_VISIT
        return ranked_products

    def predict_products_for_query(self, query_text: str) -> Union[str, List[str], Dict[str, Any]]:
        trace = self._trace_disease_detection(query_text)
        disease_code = str(trace.get("final_disease_code", "None"))
        disease_name = (
            trace.get("disease_name")
            or trace.get("selected_disease_name")
            or trace.get("final_disease_name")
            or ""
        )
        diagnosis = disease_name or disease_code
        if disease_code == "None":
            return REQUIRES_MEDICAL_VISIT

        product_start = time.perf_counter()
        all_candidates = self._rank_products_for_disease(query_text, disease_code)
        trace_latency = trace.get("latency")
        if isinstance(trace_latency, dict):
            trace_latency["product_ms"] = round((time.perf_counter() - product_start) * 1000.0, 2)
            trace_latency["total_ms"] = round(trace_latency.get("total_ms", 0.0) + trace_latency["product_ms"], 2)

        if not isinstance(all_candidates, list):
            return all_candidates

        candidate_products = all_candidates[:MAX_SAFETY_CANDIDATES]
        candidate_records = self._resolve_product_records(candidate_products)
        pending_types = self._get_missing_question_types(candidate_records, self.session_manager.get_profile())
        if pending_types:
            questions = self.safety_checker.build_questions(pending_types)
            self.session_manager.set_pending_questions(pending_types, questions)
            self.session_manager.set_pending_products(candidate_products)
            return {
                "need_more_information": True,
                "questions": questions,
                "pending_question_types": pending_types,
                "products": list(candidate_products),
                "trace": trace,
                "disease_code": disease_code,
                "disease_name": disease_name,
                "diagnosis": diagnosis,
            }

        self.session_manager.set_pending_products(candidate_products)
        safety_result = self.apply_safety_layer(candidate_products)
        if isinstance(safety_result, dict) and safety_result.get("need_more_information"):
            safety_result = dict(safety_result)
            safety_result["products"] = list(candidate_products)
        if isinstance(safety_result, dict):
            result = dict(safety_result)
            result["trace"] = trace
            result["disease_code"] = disease_code
            result["disease_name"] = disease_name
            result["diagnosis"] = diagnosis
            return result
        return safety_result

    def apply_safety_layer(self, products: list[str], user_profile: UserProfile | None = None) -> Dict[str, Any]:
        profile = user_profile or self.session_manager.get_profile()
        candidate_records = self._resolve_product_records(products)
        if self._debug_safety_enabled():
            _debug_print("========== SAFETY INPUT ==========")
            for i, record in enumerate(candidate_records, start=1):
                _debug_print(f"Candidate #{i}: {record.get('product_name', '')}")
            _debug_print("========== SAFETY CHECK ==========")

        kept_products: list[dict[str, str]] = []
        removed_products: list[dict[str, Any]] = []
        grouped_warnings: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        accepted_count = 0
        rejected_count = 0
        refill_count = 0

        def build_issue_payload(product_name: str, warnings: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            grouped_removed: list[dict[str, Any]] = []
            grouped_warns: list[dict[str, Any]] = []
            if not warnings:
                return grouped_removed, grouped_warns
            ingredients = list(dict.fromkeys([str(w.get("ingredient", "")).strip() for w in warnings if str(w.get("ingredient", "")).strip()]))
            reasons = list(dict.fromkeys([str(w.get("condition", w.get("level", ""))).strip() for w in warnings if str(w.get("condition", w.get("level", ""))).strip()]))
            reason_text = "; ".join(reasons)
            grouped_removed.append({
                "product": product_name,
                "reason": reason_text,
                "matched_ingredients": ingredients,
            })
            grouped_warns.append({
                "product": product_name,
                "message": reason_text,
                "ingredients": ingredients,
            })
            return grouped_removed, grouped_warns

        for idx, record in enumerate(candidate_records, start=1):
            product_key = record.get("sku", "") or record.get("product_name", "")
            if product_key in seen_keys:
                continue
            if self._debug_safety_enabled():
                _debug_print(f"Checking #{idx}")
            warnings = self.safety_checker.get_product_warnings(record, profile)
            if warnings:
                removed_block, warning_block = build_issue_payload(record.get("product_name", ""), warnings)
                removed_products.extend(removed_block)
                grouped_warnings.extend(warning_block)
                rejected_count += 1
                if self._debug_safety_enabled():
                    _debug_print("REJECT")
                    _debug_print("Reason:")
                    _debug_print(removed_block[0]["reason"] if removed_block else "")
                    _debug_print("Matched Ingredients:")
                    _debug_print(removed_block[0]["matched_ingredients"] if removed_block else [])
                    _debug_print("Skipped.")
            else:
                kept_products.append(record)
                seen_keys.add(product_key)
                accepted_count += 1
                if self._debug_safety_enabled():
                    _debug_print("PASS")
                    _debug_print("Reason:")
                    _debug_print("No safety rule matched")
                if len(kept_products) >= TOP_K_PRODUCTS:
                    break
            if self._debug_safety_enabled() and len(kept_products) < TOP_K_PRODUCTS and idx < len(candidate_records):
                refill_count += 1
                _debug_print(f"Refill from candidate #{idx + 1}")

        if self._debug_safety_enabled():
            _debug_print("========== SAFETY SUMMARY ==========")
            _debug_print(f"Input candidates : {len(candidate_records)}")
            _debug_print(f"Rejected : {rejected_count}")
            _debug_print(f"Accepted : {accepted_count}")
            _debug_print(f"Refilled : {refill_count}")
            _debug_print(f"Warnings : {len(grouped_warnings)}")
            _debug_print(f"Final products : {len(kept_products)}")
            _debug_print("====================================")
            if len(kept_products) < TOP_K_PRODUCTS:
                _debug_print(f"Only {len(kept_products)} safe OTC products found.")
            _debug_print("========== FINAL TOP-K ==========")
            _debug_print([item.get("product_name", '') for item in kept_products])

        return {
            "need_more_information": False,
            "products": [item.get("product_name", "") for item in kept_products],
            "removed_products": removed_products,
            "warnings": grouped_warnings,
        }

    def handle_safety_answer(self, answer_text: str) -> Dict[str, Any]:
        session = self.session_manager.get_session()
        pending_types = list(session.awaiting_question_types)
        if not pending_types:
            return {"need_more_information": False}
        before_profile = self.session_manager.get_profile()
        if self._debug_safety_enabled():
            _debug_print("========== PROFILE BEFORE UPDATE ==========")
            _debug_print(before_profile)
            _debug_print("========== USER ANSWER ==========")
            _debug_print(answer_text)
        followup = self._update_profile_from_answer(pending_types[0], answer_text)
        updated_profile = self.session_manager.get_profile()
        if followup:
            session.awaiting_question_types = [pending_types[0]] + pending_types[1:]
            session.awaiting_questions = [followup] + self.safety_checker.build_questions(pending_types[1:])
        else:
            session.awaiting_question_types = pending_types[1:]
            session.awaiting_questions = self.safety_checker.build_questions(session.awaiting_question_types)
        if self._debug_safety_enabled():
            _debug_print("========== UPDATED PROFILE ==========")
            _debug_print(updated_profile)
            _debug_print("========== PENDING QUESTION TYPES ==========")
            _debug_print(session.awaiting_question_types)
            _debug_print("========== QUESTIONS ==========")
            _debug_print(session.awaiting_questions)
        if session.awaiting_question_types:
            return {
                "need_more_information": True,
                "questions": session.awaiting_questions,
                "pending_question_types": session.awaiting_question_types,
            }
        return {"need_more_information": False, "profile": updated_profile}

    def _update_profile_from_answer(self, question_type: str, answer_text: str) -> str | None:
        normalized = str(answer_text).strip().lower()
        if question_type == "Age":
            age = self._extract_age(answer_text)
            if age is not None:
                self.session_manager.update_profile(age=age)
            return None
        if question_type == "Pregnancy":
            self.session_manager.update_profile(pregnant=self._parse_yes_no(normalized))
            return None
        if question_type == "Breastfeeding":
            self.session_manager.update_profile(breastfeeding=self._parse_yes_no(normalized))
            return None
        if question_type == "Allergy":
            yn = self._parse_yes_no(normalized)
            if yn is False:
                self.session_manager.update_profile(allergies=[])
                return None
            if yn is True:
                return "Bạn dị ứng với thuốc hoặc hoạt chất nào?"
            allergies = self._extract_list_items(answer_text)
            if allergies:
                current = self.session_manager.get_profile().allergies
                self.session_manager.update_profile(allergies=list(dict.fromkeys([*current, *allergies])))
            return None
        if question_type == "ChronicDisease":
            yn = self._parse_yes_no(normalized)
            if yn is False:
                self.session_manager.update_profile(chronic_diseases=[])
                return None
            if yn is True:
                return "Bạn mắc bệnh nền gì?"
            diseases = self._extract_list_items(answer_text)
            if diseases:
                current = self.session_manager.get_profile().chronic_diseases
                self.session_manager.update_profile(chronic_diseases=list(dict.fromkeys([*current, *diseases])))
            return None
        return None

    @staticmethod
    def _parse_yes_no(answer_text: str) -> bool | None:
        text = str(answer_text).strip().lower()
        if not text:
            return None
        if any(token in text for token in ["không", "khong", "không ạ", "khong a", "no", "ko"]):
            return False
        if any(token in text for token in ["có", "co", "yes", "đúng", "dúng"]):
            return True
        return None

    @staticmethod
    def _extract_age(answer_text: str) -> int | None:
        match = re.search(r"(\d{1,3})", str(answer_text))
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_list_items(answer_text: str) -> list[str]:
        text = re.sub(r"[.;\n]+", ",", str(answer_text).strip())
        items = [item.strip() for item in text.split(",") if item.strip()]
        return items

    def _resolve_product_records(self, products: list[str]) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for product_name in products:
            matched = self.products_df.loc[self.products_df["product_name"] == product_name]
            if matched.empty:
                records.append(
                    {
                        "product_name": product_name,
                        "ingredients": "",
                        "canonical_ingredients": "",
                        "sku": "",
                    }
                )
                continue
            row = matched.iloc[0]
            records.append(
                {
                    "product_name": str(row.get("product_name", "")).strip(),
                    "ingredients": str(row.get("ingredients", "")).strip(),
                    "canonical_ingredients": str(row.get("canonical_ingredients", "")).strip(),
                    "sku": str(row.get("sku", "")).strip(),
                }
            )
        if records:
            print("========== SAFETY INPUT CACHE ==========")
            print(records[0])
        return records

    def _get_missing_question_types(self, products: list[dict[str, str]], profile: UserProfile) -> list[str]:
        pending_types: list[str] = []
        for product in products:
            for ingredient in self.safety_checker._extract_ingredients(product):
                matches = self.safety_checker._rows_for_ingredient(ingredient)
                for _, row in matches.iterrows():
                    warning_type = str(row.get("warning_type", "")).strip()
                    if warning_type not in WARNING_QUESTION_MAP:
                        continue
                    if self._is_profile_value_missing(warning_type, profile) and warning_type not in pending_types:
                        pending_types.append(warning_type)
        return [warning_type for warning_type in WARNING_TYPE_PRIORITY if warning_type in pending_types] + [warning_type for warning_type in pending_types if warning_type not in WARNING_TYPE_PRIORITY]

    @staticmethod
    def _is_profile_value_missing(warning_type: str, profile: UserProfile) -> bool:
        if warning_type == "Age":
            return profile.age is None
        if warning_type == "Pregnancy":
            return profile.pregnant is None
        if warning_type == "Breastfeeding":
            return profile.breastfeeding is None
        if warning_type == "Allergy":
            return not profile.allergies
        if warning_type == "ChronicDisease":
            return not profile.chronic_diseases
        return False

    @staticmethod
    def _debug_safety_enabled() -> bool:
        return DEBUG_SAFETY

    def update_user_profile(
        self,
        age: int | None = None,
        pregnant: bool | None = None,
        breastfeeding: bool | None = None,
        allergies: list[str] | None = None,
        chronic_diseases: list[str] | None = None,
    ) -> UserProfile:
        return self.session_manager.update_profile(
            age=age,
            pregnant=pregnant,
            breastfeeding=breastfeeding,
            allergies=allergies,
            chronic_diseases=chronic_diseases,
        )


class RAGPredictorAlias(RAGPredictor):
    pass


__all__ = [
    "SYNTHETIC_QA",
    "DISEASE_MAP",
    "CATEGORIES_CONFIG",
    "PRODUCTS_CLEANED",
    "REQUIRES_MEDICAL_VISIT",
    "OLLAMA_URL",
    "OLLAMA_MODEL",
    "AdvancedRAGLoadError",
    "RAGPredictor",
    "RAGPredictorAlias",
    "SafetyChecker",
    "UserProfile",
]

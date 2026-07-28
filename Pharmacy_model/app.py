"""Streamlit dashboard cho Pharmacy AI với giao diện chat đối chứng đa mô hình."""

from __future__ import annotations

import copy
import io
import re
import sys
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRODUCTS_CLEANED_PATH = PROJECT_ROOT / "data" / "silver" / "products_cleaned_backup.csv"

MODEL_OPTIONS = {
    "baseline": "ML Baseline Strict (TF-IDF + LinearSVC)",
    "graph": "Knowledge Graph (NetworkX)",
    "rag": "Advanced RAG (ChromaDB + Reranker + LLM)",
    "compare": "📊 Chạy thử nghiệm đối chứng (Cả 3 mô hình cùng lúc)",
}

CHAT_STYLE = """
<style>
    .block-container { padding-top: 1.2rem; }
    .stChatMessage { border-radius: 1rem; }
    .stButton>button {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        color: #1f2937;
        border: none;
        border-radius: 999px;
        font-weight: 700;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: #111827;
    }
    .pharma-card {
        background: #fffaf0;
        border: 1px solid #fde68a;
        border-left: 6px solid #f59e0b;
        border-radius: 16px;
        padding: 14px 16px;
        margin: 10px 0;
        box-shadow: 0 4px 18px rgba(245, 158, 11, 0.08);
    }
    .pharma-card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #92400e;
        margin-bottom: 6px;
    }
    .pharma-card-body, .pharma-card-foot {
        color: #374151;
        margin-top: 4px;
        line-height: 1.5;
    }
    .badge-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 2px 0; }
    .pharma-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #e5e7eb;
        color: #374151;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .hero-banner {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        color: #1f2937;
        border-radius: 20px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(245, 158, 11, 0.2);
    }
</style>
"""

MISSING_TEXT_VALUES = {"", "nan", "none", "null", "n/a", "na", "-"}


def _normalize_sku(value: object) -> str:
    return str(value or "").strip()


def _normalize_text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).lower().strip()


def _clean_display_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in MISSING_TEXT_VALUES else text


def _first_non_empty(row: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        cleaned = _clean_display_value(value)
        if cleaned:
            return cleaned
    return ""


def normalize_medical_display_name(value: Any) -> str:
    text = _normalize_text(value)
    if not text or text == "none":
        return "Không xác định"
    compact = re.sub(r"[^a-z0-9à-ỹđ]+", "", text.replace("benh:", "").replace("-", " "))
    fallback_map = {
        "benh:apxephoi": "Áp-xe phổi",
        "benh:apxehaumon": "Áp-xe hậu môn",
        "benh:phidaituyentienliet": "Phì đại tuyến tiền liệt",
        "benh:viemxoangtran": "Viêm xoang trán",
    }
    return fallback_map.get(f"benh:{compact}", fallback_map.get(compact, str(value or "").strip() or "Không xác định"))


def _read_products_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "sku",
            "product_name",
            "category_key",
            "category_name",
            "sale_price",
            "ingredients",
            "usage",
            "dosage",
            "description",
        ],
        dtype={"sku": str},
        keep_default_na=False,
    ).fillna("")
    df["sku"] = df["sku"].astype(str).map(_normalize_sku)
    return df


PRODUCTS_CLEANED_DF = _read_products_csv(PRODUCTS_CLEANED_PATH)


@st.cache_resource(show_spinner="Đang nạp ML benchmark...")
def load_ml_model() -> Any:
    from src.benchmark.ml_baseline.inference import MLBaselinePredictor

    return MLBaselinePredictor()


@st.cache_resource(show_spinner="Đang nạp Graph benchmark...")
def load_graph_model() -> Any:
    from src.benchmark.graph_db.inference import GraphKnowledgePredictor

    return GraphKnowledgePredictor()


@st.cache_resource(show_spinner="Đang nạp RAG benchmark...")
def load_rag_model() -> Any:
    try:
        from src.benchmark.rag.inference import RAGPredictor

        if "rag_error" in st.session_state:
            del st.session_state["rag_error"]
        return RAGPredictor()
    except Exception as e:
        st.session_state["rag_error"] = traceback.format_exc()
        print(traceback.format_exc())
        return None


def ensure_chat_state(selected_mode: str) -> None:
    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = selected_mode
    if st.session_state.selected_mode != selected_mode:
        st.session_state.selected_mode = selected_mode
        st.session_state.messages = []
        st.session_state.rag_waiting_answer = False
        st.session_state.rag_pending_products = []
        st.session_state.rag_last_result = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_waiting_answer" not in st.session_state:
        st.session_state.rag_waiting_answer = False
    if "rag_pending_products" not in st.session_state:
        st.session_state.rag_pending_products = []
    if "rag_last_result" not in st.session_state:
        st.session_state.rag_last_result = None


def render_sidebar() -> str:
    st.sidebar.title("Chế độ kiểm thử")
    selected = st.sidebar.selectbox(
        "Chế độ thử nghiệm",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda key: MODEL_OPTIONS[key],
    )
    st.sidebar.info(
        "⚠️ **Lưu ý quan trọng**\n\nPharmabee AI là hệ thống hỗ trợ gợi ý thuốc không kê đơn (OTC) dựa trên triệu chứng người dùng cung cấp và cơ sở dữ liệu dược phẩm.\n\nKết quả chỉ mang tính tham khảo, không thay thế cho việc chẩn đoán, kê đơn hoặc chỉ định điều trị của bác sĩ.\n\nNếu triệu chứng kéo dài, trở nên nghiêm trọng hoặc xuất hiện các dấu hiệu cảnh báo như: sốt cao kéo dài, khó thở, đau ngực, co giật, mất ý thức, nôn ra máu, tiểu ra máu thì hãy đến cơ sở y tế ngay.\n\nKhông tự ý sử dụng thuốc nếu đang mang thai, đang cho con bú, có bệnh nền, dị ứng thuốc, hoặc đang dùng thuốc điều trị khác mà chưa có tư vấn của nhân viên y tế."
    )
    st.sidebar.caption("Dữ liệu chat chỉ tồn tại trong session hiện tại.")
    return selected


def render_header(selected: str) -> None:
    st.markdown(CHAT_STYLE, unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-banner'><h2 style='margin:0'>Pharmabee AI Dashboard</h2><div style='margin-top:4px'>Trợ lý gợi ý thuốc không kê đơn trên cùng một giao diện.</div></div>",
        unsafe_allow_html=True,
    )
    render_otc_badge()
    st.caption(f"Chế độ đang chạy: {MODEL_OPTIONS[selected]}")


def truncate_text(text: Any, limit: int = 200) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else value[:limit].rstrip() + "..."


def format_price_vnd(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return text
    try:
        amount = int(digits)
    except ValueError:
        return text
    if amount < 1000 and amount > 0:
        return f"{amount}.000 đ"
    return f"{amount:,} đ".replace(",", ".")


def _format_latency_breakdown(latency: Any) -> str:
    if not isinstance(latency, dict):
        return ""
    labels = [
        ("Query Expansion", "query_expansion_ms"),
        ("Hybrid Retrieval", "retrieval_ms"),
        ("Rerank", "rerank_ms"),
        ("LLM", "llm_ms"),
        ("Confidence", "confidence_ms"),
        ("Product Ranking", "product_ms"),
        ("Total", "total_ms"),
    ]
    lines = ["⏱️ **Pipeline Performance**", ""]
    for label, key in labels:
        value = latency.get(key, None)
        try:
            value_num = float(value)
            lines.append(f"- {label} : {value_num:.2f} ms")
        except (TypeError, ValueError):
            lines.append(f"- {label} : N/A")
    return "\n".join(lines)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


def render_keyword_badges(keywords: List[str]) -> None:
    if not keywords:
        # st.caption("Không có keyword kích hoạt phù hợp.")
        return
    badges = " ".join(f"<span class='pharma-badge'>{_normalize_text(kw)}</span>" for kw in keywords)
    st.markdown(f"<div class='badge-wrap'>{badges}</div>", unsafe_allow_html=True)


def render_otc_badge() -> None:
    st.markdown(
        "<div style='margin: 0.4rem 0 0.8rem 0;'><span style='display:inline-block;padding:0.35rem 0.75rem;border-radius:999px;background:#dcfce7;color:#166534;font-size:0.82rem;font-weight:700;'>💊 Chỉ hỗ trợ gợi ý thuốc không kê đơn (OTC)</span></div>",
        unsafe_allow_html=True,
    )


def render_otc_disclaimer() -> None:
    st.caption(
        "ℹ️ Danh sách trên là các thuốc OTC có mức độ phù hợp cao theo mô hình AI. Đây KHÔNG phải đơn thuốc. Hãy đọc kỹ hướng dẫn sử dụng, chống chỉ định và liều dùng trước khi sử dụng. Nếu triệu chứng không cải thiện hoặc nặng hơn, hãy đến cơ sở y tế."
    )


def render_no_otc_warning() -> None:
    st.warning(
        "⚠️ Các triệu chứng này không phù hợp để tự điều trị bằng thuốc OTC. Bạn nên đến cơ sở y tế để được bác sĩ hoặc dược sĩ thăm khám và tư vấn."
    )


def resolve_ranked_products_to_internal_df(product_refs: Any) -> pd.DataFrame:
    if product_refs is None:
        return pd.DataFrame()
    if isinstance(product_refs, pd.DataFrame):
        df = product_refs.copy()
        if "sku" in df.columns:
            df["sku"] = df["sku"].astype(str).map(_normalize_sku)
        if "source" not in df.columns:
            df["source"] = "cleaned"
        return df
    if isinstance(product_refs, str):
        if not product_refs.strip() or product_refs == "REQUIRES_MEDICAL_VISIT":
            return pd.DataFrame()
        product_refs = [product_refs]
    if not isinstance(product_refs, list) or not product_refs:
        return pd.DataFrame()

    cleaned_df = PRODUCTS_CLEANED_DF.copy()
    if cleaned_df.empty or "sku" not in cleaned_df.columns:
        return pd.DataFrame()

    cleaned_df["sku"] = cleaned_df["sku"].astype(str).map(_normalize_sku)
    cleaned_df = cleaned_df.drop_duplicates(subset=["sku"], keep="first")

    resolved_rows: list[Dict[str, Any]] = []
    used_skus: set[str] = set()

    for rank, raw_item in enumerate(product_refs, start=1):
        if isinstance(raw_item, dict):
            sku = _normalize_sku(raw_item.get("sku", ""))
            product_name = _clean_display_value(raw_item.get("product_name", ""))
        else:
            sku = ""
            product_name = _clean_display_value(raw_item)

        if not product_name or product_name in used_skus:
            continue

        name_norm = _normalize_text(product_name)
        matched_rows = cleaned_df.loc[cleaned_df["product_name"].astype(str).map(_normalize_text) == name_norm]
        if matched_rows.empty:
            continue

        base_row = matched_rows.iloc[0].to_dict()
        sku = _normalize_sku(base_row.get("sku", ""))
        if not sku or sku in used_skus:
            continue
        base_row["sku"] = sku
        base_row["product_name"] = _clean_display_value(base_row.get("product_name", ""))
        base_row["rank"] = rank
        base_row["source"] = "cleaned"
        resolved_rows.append(base_row)
        used_skus.add(sku)

    internal_df = pd.DataFrame(resolved_rows)
    if internal_df.empty:
        return pd.DataFrame()

    internal_df["sku"] = internal_df["sku"].astype(str).map(_normalize_sku)
    return internal_df


def build_products_table(products: List[Dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(products, pd.DataFrame):
        df = products.copy()
    else:
        df = pd.DataFrame(list(products or []))
    if df.empty:
        render_no_otc_warning()
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        name = _first_non_empty(row_dict, "product_name")
        price = format_price_vnd(_first_non_empty(row_dict, "sale_price"))
        usage = truncate_text(_first_non_empty(row_dict, "usage", "description"), 400)
        dosage = truncate_text(_first_non_empty(row_dict, "dosage"), 400)
        description = truncate_text(_first_non_empty(row_dict, "description"), 400)
        category = _first_non_empty(row_dict, "category_name", "category_key")
        rows.append(
            {
                "💊 Tên thuốc / Sản phẩm": name,
                "💰 Giá bán": price,
                "🎯 Công dụng & Chỉ định": usage,
                "⏳ Cách dùng & Liều lượng": dosage,
                "📝 Mô tả": description,
                # "⚠️ Tác dụng phụ": "N/A",
                # "💉 Thuốc kê đơn": "N/A",
                "🏷️ Danh mục": category,
                # "📊 Điểm tương đồng": row_dict.get("match_score") or 0,
            }
        )

    return pd.DataFrame(rows)


def product_card(product: Dict[str, Any]) -> None:
    name = _first_non_empty(product, "product_name") or "Không rõ tên"
    usage = truncate_text(_first_non_empty(product, "usage", "description") or "Chưa có mô tả", 300)
    instructions = truncate_text(_first_non_empty(product, "dosage") or "Chưa có hướng dẫn", 300)
    category = _first_non_empty(product, "category_name", "category_key") or "Không rõ"
    price = format_price_vnd(_first_non_empty(product, "sale_price"))
    st.markdown(
        f"""
        <div class='pharma-card'>
            <div class='pharma-card-title'>💊 {name}</div>
            <div class='pharma-card-body'><b>💰 Giá bán:</b> {price}</div>
            <div class='pharma-card-body'><b>🎯 Công dụng:</b> {usage}</div>
            <div class='pharma-card-body'><b>⚠️ Tác dụng phụ:</b> N/A</div>
            <div class='pharma-card-body'><b>🏷️ Danh mục:</b> {category}</div>
            <div class='pharma-card-foot'><b>📊 Điểm tương đồng:</b> {product.get('match_score', 0)}</div>
            <div class='pharma-card-foot'><b>💉 Thuốc kê đơn:</b> N/A</div>
            <div class='pharma-card-foot'><b>⏳ Cách dùng:</b> {instructions}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_product_cards(products: List[Dict[str, Any]] | pd.DataFrame) -> None:
    if isinstance(products, pd.DataFrame):
        records = products.to_dict(orient="records")
    else:
        records = list(products or [])
    if len(records) == 0:
        st.caption("Không có sản phẩm OTC phù hợp để hiển thị.")
        return
    for product in records[:4]:
        product_card(product)
    df_final = build_products_table(records[:4])
    if not df_final.empty:
        st.dataframe(df_final, use_container_width=True, hide_index=True)
        render_otc_disclaimer()
        st.download_button(
            label="📥 Tải danh sách thuốc (Excel)",
            data=to_excel_bytes(df_final),
            file_name="otc_products_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def _resolve_model_products(query: str, predictor: Any, mode: str) -> Dict[str, Any]:
    start = time.time()
    try:
        if mode == "baseline":
            disease_code = predictor.predict_disease_code(query)
            raw_products = predictor.predict_products_for_query(query)
        elif mode == "graph":
            disease_code = predictor.detect_disease_code(query) or "None"
            raw_products = predictor.predict_products_for_query(query)
        else:
            trace = predictor.debug_trace_query(query)
            disease_code = str(trace.get("final_disease_code", "None")) or "None"
            raw_products = trace.get("final_products", predictor.predict_products_for_query(query))

        products_df = resolve_ranked_products_to_internal_df(raw_products)
        diagnosis = normalize_medical_display_name(disease_code)
        if "sku" in products_df.columns:
            products_df["sku"] = products_df["sku"].astype(str).map(_normalize_sku)
        latency_seconds = time.time() - start
        if mode == "rag":
            trace_latency = trace.get("latency") if isinstance(trace, dict) else None
            total_ms = None
            if isinstance(trace_latency, dict):
                total_ms = trace_latency.get("total_ms")
            if total_ms is not None:
                try:
                    latency_seconds = float(total_ms) / 1000.0
                except (TypeError, ValueError):
                    latency_seconds = time.time() - start
        return {
            "ok": True,
            "latency": latency_seconds,
            "diagnosis": diagnosis,
            "products": products_df,
            "answer": "",
            "error": None,
            "disease_code": disease_code,
            "raw_result": raw_products,
            "activated_keywords": [],
            "trace": trace if mode == "rag" else None,
        }
    except Exception as e:
        if mode == "rag":
            st.session_state["rag_error"] = str(e)
        return {
            "ok": False,
            "latency": time.time() - start,
            "diagnosis": "Không xác định",
            "products": pd.DataFrame(),
            "answer": "",
            "error": str(e),
            "disease_code": "",
            "raw_result": None,
            "activated_keywords": [],
            "trace": None,
        }


def adapt_ml_result(query: str, predictor: Any) -> Dict[str, Any]:
    return _resolve_model_products(query, predictor, "baseline")


def adapt_graph_result(query: str, predictor: Any) -> Dict[str, Any]:
    return _resolve_model_products(query, predictor, "graph")


def adapt_rag_result(query: str, predictor: Any) -> Dict[str, Any]:
    return _resolve_model_products(query, predictor, "rag")


def append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def render_message_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_rag_warning() -> None:
    st.warning("⚠️ Phân hệ Advanced RAG hiện đang bận hoặc gặp sự cố khởi tạo mô hình nhúng (Embedding Space).")


def reset_rag_safety_state() -> None:
    st.session_state.rag_waiting_answer = False
    st.session_state.rag_pending_products = []
    st.session_state.rag_last_result = None
    st.session_state.rag_conversation_history = []
    st.session_state.rag_raw_payload = None


def reset_rag_session(predictor: Any) -> None:
    reset_rag_safety_state()
    if predictor is not None and hasattr(predictor, "session_manager"):
        try:
            predictor.session_manager.reset()
        except Exception:
            pass


def _format_profile_value(value: Any) -> str:
    if value is True:
        return "Có"
    if value is False:
        return "Không"
    if value in (None, "", [], {}):
        return "Không rõ"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "Không rõ"
    return str(value)


def _render_profile_block(profile: Any) -> None:
    if profile is None:
        return
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump()
    elif hasattr(profile, "__dict__") and not isinstance(profile, dict):
        profile = dict(profile.__dict__)
    if not isinstance(profile, dict) or not profile:
        return
    lines = ["👤 Hồ sơ người dùng"]
    mapping = [
        ("age", "Tuổi"),
        ("pregnant", "Mang thai"),
        ("breastfeeding", "Cho con bú"),
        ("allergies", "Dị ứng"),
        ("chronic_diseases", "Bệnh nền"),
    ]
    for key, label in mapping:
        if key in profile:
            lines.append(f"• {label}: {_format_profile_value(profile.get(key))}")
    st.markdown("\n".join(lines))


def _render_safety_summary(summary: list[dict[str, Any]]) -> None:
    if not summary:
        return
    with st.chat_message("assistant"):
        st.markdown("**🧠 Safety Conversation Summary**")
        for idx, item in enumerate(summary, start=1):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            st.markdown(f"**Q{idx}:**")
            if question:
                st.markdown(question)
            if answer:
                st.markdown(answer)


def render_rag_safety_result(result: Dict[str, Any], predictor: Any) -> None:
    with st.chat_message("assistant"):
        st.markdown(f"**🩺 Chẩn đoán:** {result.get('diagnosis', 'Không xác định')}")
    summary = st.session_state.get("rag_conversation_history") or []
    if summary:
        _render_safety_summary(summary)
    warnings = result.get("warnings") or []
    removed_products = result.get("removed_products") or []
    profile = result.get("profile") or predictor.session_manager.get_profile()
    if removed_products:
        with st.chat_message("assistant"):
            st.markdown("**⚠️ Safety Decision**")
            for item in removed_products:
                st.markdown(f"- {item.get('product', item)}")
    if warnings:
        with st.chat_message("assistant"):
            st.markdown("**⚠️ Warnings**")
            for item in warnings:
                st.markdown(f"- {item.get('product', '')}: {item.get('message', '')}")
    if profile is not None:
        with st.chat_message("assistant"):
            _render_profile_block(profile)

def render_rag_payload(result: Dict[str, Any], predictor: Any) -> None:
    trace = result.get("trace") or {}
    clinical_summary = str(trace.get("clinical_summary", "")).strip() or "Không có thông tin."
    reasoning = str(trace.get("reasoning", "")).strip() or "Không có thông tin."
    advice = str(trace.get("advice", "")).strip() or "Không có thông tin."
    confidence_label = str(trace.get("confidence_label", "")).strip() or "Không có thông tin."
    diagnosis = str(result.get("diagnosis", "")).strip()
    if not diagnosis or diagnosis == "Không xác định":
        diagnosis = str(trace.get("disease_name", "")).strip() or str(trace.get("selected_disease_name", "")).strip() or str(result.get("disease_name", "")).strip() or "Không xác định"
    disease_code = str(result.get("disease", "")).strip() or str(result.get("disease_code", "")).strip()
    if not disease_code:
        disease_code = str(trace.get("selected_disease", "")).strip() or str(trace.get("disease_code", "")).strip() or "Không xác định"
    disease_name = str(result.get("disease_name", "")).strip() or str(trace.get("disease_name", "")).strip()
    latency = result.get("latency")
    latency_block = _format_latency_breakdown(trace.get("latency"))
    profile = result.get("profile") or predictor.session_manager.get_profile()

    with st.chat_message("assistant"):
        st.markdown(f"**🩺 Chẩn đoán:** {diagnosis}\n\n{result.get('answer', '')}")
        st.markdown(f"**📖 Tóm tắt bệnh**\n\n{clinical_summary}")
        st.markdown(f"**🧠 Giải thích**\n\n{reasoning}")
        st.markdown(f"**💡 Khuyến nghị**\n\n{advice}")
        if profile is not None:
            st.markdown("---")
            _render_profile_block(profile)
        if latency_block:
            st.markdown("---")
            st.markdown(latency_block)
        render_keyword_badges(result.get("activated_keywords", []))

    removed_products = result.get("removed_products") or []
    warnings = result.get("warnings") or []
    if removed_products:
        with st.container():
            st.markdown(
                """
                <div style="background:#fff1f2;border:1px solid #ef4444;border-radius:12px;padding:14px 16px;margin:12px 0;">
                  <div style="font-weight:700;color:#b91c1c;margin-bottom:10px;">⚠️ Thuốc đã bị loại bởi Safety</div>
                """,
                unsafe_allow_html=True,
            )
            for item in removed_products:
                product_name = str(item.get("product", "")).strip() or "Không rõ"
                reason = str(item.get("reason", item.get("message", ""))).strip() or "Không rõ"
                st.markdown(f"• **{product_name}**  \nLý do: {reason}")
            st.markdown("</div>", unsafe_allow_html=True)
    if warnings:
        with st.chat_message("assistant"):
            st.markdown("**⚠️ Warnings**")
            for item in warnings:
                st.markdown(f"- {item.get('product', '')}: {item.get('message', '')}")

    products = result.get("products", [])
    print("========== UI PRODUCTS ==========")
    print("type:", type(products))
    try:
        print("len:", len(products))
    except Exception:
        print("len: <unavailable>")
    if isinstance(products, pd.DataFrame):
        print("product names:", products.get("product_name", pd.Series(dtype=str)).head(5).tolist() if "product_name" in products.columns else [])
    else:
        preview = []
        for item in list(products)[:5]:
            if isinstance(item, dict):
                preview.append(str(item.get("product_name", item)))
            else:
                preview.append(str(item))
        print("product names:", preview)
    print("===============================")
    rag_df = resolve_ranked_products_to_internal_df(products)
    rag_df = build_products_table(rag_df)
    if not rag_df.empty:
        st.dataframe(rag_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Tải danh sách thuốc (Excel)",
            data=to_excel_bytes(rag_df),
            file_name="rag_products_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        render_no_otc_warning()
    render_otc_disclaimer()
    if latency is not None:
        st.caption(f"Latency: {latency:.4f} s")


def render_compare_results(query: str) -> None:
    cols_metric = st.columns(3)
    ml_out = adapt_ml_result(query, load_ml_model())
    graph_out = adapt_graph_result(query, load_graph_model())
    rag_out = adapt_rag_result(query, load_rag_model())

    with cols_metric[0]:
        st.metric(label="⏱️ Tốc độ ML Baseline", value=f"{ml_out['latency']:.4f} s")
    with cols_metric[1]:
        st.metric(label="⏱️ Tốc độ Knowledge Graph", value=f"{graph_out['latency']:.4f} s")
    with cols_metric[2]:
        st.metric(label="⏱️ Tốc độ Advanced RAG", value=f"{rag_out['latency']:.4f} s" if rag_out["ok"] else "N/A")

    st.divider()
    cols = st.columns(3)
    with cols[0]:
        st.subheader("ML Baseline")
        if ml_out["ok"]:
            with st.chat_message("assistant"):
                st.markdown(f"**Chẩn đoán:** {ml_out['diagnosis']}")
                render_keyword_badges(ml_out.get("activated_keywords", []))
            ml_df = build_products_table(ml_out["products"])
            if not ml_df.empty:
                st.dataframe(ml_df, use_container_width=True, hide_index=True)
                render_otc_disclaimer()
                st.download_button(
                    label="📥 Tải danh sách thuốc (Excel)",
                    data=to_excel_bytes(ml_df),
                    file_name="baseline_products_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                render_no_otc_warning()
        else:
            st.warning(f"Baseline gặp lỗi: {ml_out['error']}")
    with cols[1]:
        st.subheader("Knowledge Graph")
        if graph_out["ok"]:
            with st.chat_message("assistant"):
                st.markdown(f"**Chẩn đoán:** {graph_out['diagnosis']}")
                render_keyword_badges(graph_out.get("activated_keywords", []))
            graph_df = build_products_table(graph_out["products"])
            if not graph_df.empty:
                st.dataframe(graph_df, use_container_width=True, hide_index=True)
                render_otc_disclaimer()
                st.download_button(
                    label="📥 Tải danh sách thuốc (Excel)",
                    data=to_excel_bytes(graph_df),
                    file_name="graph_products_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                render_no_otc_warning()
        else:
            st.warning(f"Knowledge Graph gặp lỗi: {graph_out['error']}")
    with cols[2]:
        st.subheader("Advanced RAG")
        if rag_out["ok"]:
            trace = rag_out.get("trace") or {}
            clinical_summary = str(trace.get("clinical_summary", "")).strip() or "Không có thông tin."
            reasoning = str(trace.get("reasoning", "")).strip() or "Không có thông tin."
            advice = str(trace.get("advice", "")).strip() or "Không có thông tin."
            confidence_label = str(trace.get("confidence_label", "")).strip() or "Không có thông tin."
            latency_block = _format_latency_breakdown(trace.get("latency"))
            with st.chat_message("assistant"):
                st.markdown(f"**Chẩn đoán:** {rag_out['diagnosis']}\n\n{rag_out.get('answer', '')}")
                st.markdown("---")
                st.markdown(f"**📖 Tóm tắt bệnh**\n\n{clinical_summary}")
                st.markdown(f"**🧠 Giải thích**\n\n{reasoning}")
                st.markdown(f"**📊 Độ tin cậy**\n\n{confidence_label}")
                st.markdown(f"**💡 Khuyến nghị**\n\n{advice}")
                if latency_block:
                    st.markdown("---")
                    st.markdown(latency_block)
                render_keyword_badges(rag_out.get("activated_keywords", []))
            rag_df = build_products_table(rag_out["products"])
            if not rag_df.empty:
                st.dataframe(rag_df, use_container_width=True, hide_index=True)
                st.download_button(
                    label="Tải bảng thuốc Excel",
                    data=to_excel_bytes(rag_df),
                    file_name="rag_products_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                render_no_otc_warning()
            render_otc_disclaimer()
        else:
            render_rag_warning()


def render_single_mode(selected: str, query: str) -> None:
    if selected == "baseline":
        out = adapt_ml_result(query, load_ml_model())
        if out["ok"]:
            with st.chat_message("assistant"):
                st.markdown(f"**Chẩn đoán:** {out['diagnosis']}")
                render_keyword_badges(out.get("activated_keywords", []))
            baseline_df = build_products_table(out["products"])
            st.dataframe(baseline_df, use_container_width=True, hide_index=True)
            render_otc_disclaimer()
            st.download_button(
                label="📥 Tải danh sách thuốc (Excel)",
                data=to_excel_bytes(baseline_df),
                file_name="baseline_products_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            if out.get("latency") is not None:
                st.caption(f"Latency: {out['latency']:.4f} s")
        else:
            st.warning(f"Baseline gặp lỗi: {out['error']}")
        return

    if selected == "graph":
        out = adapt_graph_result(query, load_graph_model())
        if out["ok"]:
            with st.chat_message("assistant"):
                st.markdown(f"**Chẩn đoán:** {out['diagnosis']}")
                render_keyword_badges(out.get("activated_keywords", []))
            graph_df = build_products_table(out["products"])
            st.dataframe(graph_df, use_container_width=True, hide_index=True)
            render_otc_disclaimer()
            st.download_button(
                label="📥 Tải danh sách thuốc (Excel)",
                data=to_excel_bytes(graph_df),
                file_name="graph_products_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            if out.get("latency") is not None:
                st.caption(f"Latency: {out['latency']:.4f} s")
        else:
            st.warning(f"Knowledge Graph gặp lỗi: {out['error']}")
        return

    predictor = load_rag_model()

    if selected == "rag":
        with st.sidebar:
            if st.button("🔄 Reset Safety Conversation", key="reset_rag_safety_conversation", use_container_width=True):
                if predictor is not None:
                    reset_rag_session(predictor)
                else:
                    reset_rag_safety_state()
                st.rerun()
        if predictor is None:
            st.error("Advanced RAG failed.")
            st.code(st.session_state.get("rag_error", "Unknown RAG error"))
            return

    if predictor is None:
        render_rag_warning()
        return

    if st.session_state.rag_waiting_answer:
        if "rag_conversation_history" not in st.session_state:
            st.session_state.rag_conversation_history = []
        st.session_state.rag_conversation_history.append({"role": "user", "content": query})
        append_message("user", query)
        safety_result = predictor.handle_safety_answer(query)
        if safety_result.get("need_more_information"):
            question = (safety_result.get("questions") or [""])[0]
            if question:
                st.session_state.rag_conversation_history.append({"role": "assistant", "content": question})
                append_message("assistant", question)
                with st.chat_message("assistant"):
                    st.markdown(question)
            st.session_state.rag_last_result = safety_result
            return
        print("=== DEBUG BEFORE APPLY SAFETY ===")
        print("rag_pending_products:", st.session_state.rag_pending_products)
        print("profile:", predictor.session_manager.get_profile())
        final_result = predictor.apply_safety_layer(
            st.session_state.rag_pending_products,
            predictor.session_manager.get_profile(),
        )
        print("=== DEBUG AFTER APPLY SAFETY ===")
        print("final_result keys:", list(final_result.keys()) if isinstance(final_result, dict) else type(final_result))
        raw_payload = copy.deepcopy(st.session_state.get("rag_raw_payload") or {})
        merged_result = copy.deepcopy(raw_payload)
        if isinstance(final_result, dict):
            merged_result["products"] = final_result.get("products", [])
            merged_result["removed_products"] = final_result.get("removed_products", [])
            merged_result["warnings"] = final_result.get("warnings", [])
            merged_result["profile"] = predictor.session_manager.get_profile()
            if final_result.get("latency") is not None:
                merged_result["latency"] = final_result.get("latency")
            if final_result.get("trace") is not None:
                merged_result["trace"] = final_result.get("trace")
        print("raw_result.keys():", list(raw_payload.keys()) if isinstance(raw_payload, dict) else type(raw_payload))
        print("final_result.keys():", list(final_result.keys()) if isinstance(final_result, dict) else type(final_result))
        print("merged_result.keys():", list(merged_result.keys()) if isinstance(merged_result, dict) else type(merged_result))
        print("Disease Code:", merged_result.get("disease") or merged_result.get("disease_code") or merged_result.get("trace", {}).get("selected_disease") or merged_result.get("trace", {}).get("disease_code"))
        print("Diagnosis:", merged_result.get("diagnosis"))
        try:
            print("len(products):", len(merged_result.get("products", [])))
            if isinstance(merged_result.get("products", []), list):
                print("5 tên thuốc cuối cùng:", [str(item.get("product_name", item)) for item in merged_result.get("products", [])[-5:]])
        except Exception:
            print("len(products): <unavailable>")
        print("removed_products:", merged_result.get("removed_products"))
        print("warnings:", merged_result.get("warnings"))
        profile_debug = predictor.session_manager.get_profile()
        print("========== PROFILE ==========")
        if isinstance(profile_debug, dict):
            print("Age:", profile_debug.get("age"))
            print("Pregnant:", profile_debug.get("pregnant"))
            print("Breastfeeding:", profile_debug.get("breastfeeding"))
            print("Allergies:", profile_debug.get("allergies"))
            print("Chronic Diseases:", profile_debug.get("chronic_diseases"))
        print("=============================")
        st.session_state.rag_last_result = merged_result
        render_rag_payload(merged_result, predictor)
        reset_rag_safety_state()
        return

    raw_result = predictor.predict_products_for_query(query)
    st.session_state.rag_last_result = raw_result
    if not isinstance(raw_result, dict):
        render_rag_warning()
        return

    st.session_state.rag_raw_payload = copy.deepcopy(raw_result)
    st.session_state.rag_pending_products = raw_result.get("products", [])

    if raw_result.get("need_more_information"):
        if "rag_conversation_history" not in st.session_state:
            st.session_state.rag_conversation_history = []
        question = (raw_result.get("questions") or [""])[0]
        with st.chat_message("assistant"):
            st.markdown("Để cá nhân hóa tư vấn và đảm bảo an toàn, bạn có thể cung cấp thêm thông tin:")
            if question:
                st.markdown(question)
                st.session_state.rag_conversation_history.append({"role": "assistant", "content": question})
                append_message("assistant", question)
        st.session_state.rag_conversation_history.append({"role": "user", "content": query})
        st.session_state.rag_waiting_answer = True
        return

    render_rag_payload(raw_result, predictor)


def main() -> None:
    st.set_page_config(page_title="Pharmabee Dashboard", page_icon="🐝", layout="wide")
    selected = render_sidebar()
    ensure_chat_state(selected)
    render_header(selected)
    render_message_history()
    query = st.chat_input("Nhập triệu chứng cần tư vấn...")
    if query:
        append_message("user", query)
        with st.chat_message("user"):
            st.markdown(query)
        if selected == "compare":
            render_compare_results(query)
        else:
            render_single_mode(selected, query)


if __name__ == "__main__":
    main()

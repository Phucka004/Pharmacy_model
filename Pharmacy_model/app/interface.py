import warnings
import time 
import streamlit as st
import logging
from utils import get_db_and_models
from langchain_ollama import OllamaLLM
from main_navie_rag import ask_pharmabee_naive 
from main_advanced_rag import ask_pharmabee 

# --- CẤU HÌNH ---
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
st.set_page_config(page_title="Pharmacy RAG Assistant", page_icon="💊", layout="centered")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4320/4320337.png", width=80)
    st.title("PharmaBee Control")
    rag_mode = st.selectbox("Chọn mô hình:", ["Naive RAG", "Advanced RAG"])
    
    if st.button("Xóa lịch sử"):
        st.session_state.messages = []
        st.session_state.last_products = ""
        st.rerun()

# --- KHỞI TẠO SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn cần tư vấn dược phẩm gì hôm nay?"}]
if "last_products" not in st.session_state:
    st.session_state.last_products = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- XỬ LÝ CHAT INPUT (DUY NHẤT) ---
if user_query := st.chat_input("Nhập câu hỏi...", key="main_chat_input"):
    # 1. Guardrail
    if not any(word in user_query.lower() for word in ["thuốc", "bệnh", "triệu chứng", "sức khỏe", "đơn", "liều"]):
        st.chat_message("assistant").write("Xin lỗi, tôi chỉ là trợ lý dược phẩm. Tôi chỉ tư vấn về thuốc, bệnh và triệu chứng liên quan.")
        st.stop()

    # 2. LOAD MODEL (Chỉ chạy lần đầu tiên)
    if "llm" not in st.session_state:
        with st.spinner("Đang khởi động hệ thống PharmaBee..."):
            st.session_state.llm = OllamaLLM(model="llama3.2:3b", temperature=0.1)
            st.session_state.collection, st.session_state.reranker = get_db_and_models()

    llm = st.session_state.llm
    collection = st.session_state.collection
    reranker = st.session_state.reranker

    # 3. HIỂN THỊ USER MESSAGE
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 4. XỬ LÝ PHẢN HỒI ASSISTANT
    with st.chat_message("assistant"):
        # try:
        #     start_time = time.time()
        #     history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
            
        #     if rag_mode == "Naive RAG":
        #         response, _ = ask_pharmabee_naive(user_query)
        #         st.markdown(response)
        #         full_response = response
        #     else:
        #         prompt_str, info_extra = ask_pharmabee(
        #             user_input=user_query, 
        #             collection=collection, 
        #             reranker=reranker, 
        #             chat_history_str=history_str, 
        #             last_products_str=st.session_state.last_products
        #         )
        #         if info_extra: st.session_state.last_products = info_extra
                
        #         placeholder = st.empty()
        #         full_response = ""
        #         first_token_time = None
                
        #         for chunk in llm.stream(prompt_str):
        #             if first_token_time is None:
        #                 first_token_time = time.time() - start_time
        #             full_response += chunk
        #             placeholder.markdown(full_response + "▌")
                
        #         final_output = full_response + (f"\n\n{info_extra}" if info_extra else "")
        #         placeholder.markdown(final_output)
        #         st.caption(f"⏱️ Phản hồi: {time.time() - start_time:.2f}s | TTFT: {first_token_time:.2f}s")
            
        #     st.session_state.messages.append({"role": "assistant", "content": full_response + (f"\n\n{info_extra}" if info_extra else "")})
            
        # except Exception as e:
        #     st.error(f"Lỗi hệ thống: {e}")
        # --- BÊN TRONG KHỐI TRY ---
        try:
            start_time = time.time()
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
            
            # --- CẤU HÌNH BIẾN MẶC ĐỊNH ---
            full_response = ""
            info_extra = "" 
            
            if rag_mode == "Naive RAG":
                # 1. LOGIC NAIVE (Không Stream)
                response, _ = ask_pharmabee_naive(user_query)
                st.markdown(response)
                full_response = response
                
            else:
                # 2. LOGIC ADVANCED (Có Stream + Reranker)
                prompt_str, info_extra = ask_pharmabee(
                    user_input=user_query, 
                    collection=collection, 
                    reranker=reranker, 
                    chat_history_str=history_str, 
                    last_products_str=st.session_state.last_products
                )
                if info_extra: st.session_state.last_products = info_extra
                
                # Hàm Stream
                placeholder = st.empty()
                first_token_time = None
                
                for chunk in llm.stream(prompt_str):
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                
                # Hiển thị kết quả hoàn thiện
                final_output = full_response + (f"\n\n{info_extra}" if info_extra else "")
                placeholder.markdown(final_output)
                st.caption(f"⏱️ Phản hồi: {time.time() - start_time:.2f}s | TTFT: {first_token_time:.2f}s")
            
            # --- LƯU VÀO SESSION ---
            final_content = full_response + (f"\n\n{info_extra}" if info_extra else "")
            st.session_state.messages.append({"role": "assistant", "content": final_content})
            
        except Exception as e:
            st.error(f"Lỗi hệ thống: {e}")
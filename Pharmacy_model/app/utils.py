from __future__ import annotations

import re
import unicodedata
from typing import Any

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

DISPLAY_NAME_MAP = {
    "benh:apxephoi": "Áp-xe phổi",
    "benh:apxehaumon": "Áp-xe hậu môn",
    "benh:phidaituyentienliet": "Phì đại tuyến tiền liệt",
    "benh:viemxoangtran": "Viêm xoang trán",
}


def normalize_text(text: Any) -> str:
    return unicodedata.normalize("NFC", str(text)).lower().strip()


def normalize_display_name(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("benh:", "")
    compact = re.sub(r"[^a-z0-9à-ỹđ]+", "", text.replace("-", " "))
    return DISPLAY_NAME_MAP.get(f"benh:{compact}", DISPLAY_NAME_MAP.get(compact, str(value or "").strip()))


@st.cache_resource
def get_db_and_models():
    print("--- ĐANG LOAD MODEL VÀO RAM (CHỈ CHẠY 1 LẦN) ---")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path="data/vector_db")
    collection = client.get_collection(name="pharmacy_knowledge", embedding_function=embedding_func)
    reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1', device='cpu')
    return collection, reranker
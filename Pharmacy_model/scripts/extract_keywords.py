from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "silver" / "synthetic_medical_qa.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "silver" / "extracted_keywords_top10.json"

STOPWORDS = {
    "a", "an", "and", "at", "be", "by", "cho", "con", "cua", "da", "dang", "de", "den", "di", "do", "duoc",
    "em", "gio", "hay", "hoac", "hoi", "la", "lam", "len", "luc", "mot", "nam", "neu", "nhung", "o", "qua",
    "roi", "se", "ta", "tai", "thang", "thi", "thuong", "trong", "tu", "van", "voi", "xuong",
    "bac", "si", "duoc", "si", "oi", "da", "ạ", "ạ?",
    "benh", "dau", "tri", "treat", "kham", "thuoc", "uong", "uống",
    "bi", "bị", "co", "có", "khong", "không", "phai", "phải", "the", "thế", "nao", "nào",
    "sao", "sao", "lam", "làm", "biet", "biết", "xin", "vui", "long", "lòng",
    "nhe", "nhé", "di", "đi", "nha", "nhà", "nho", "nhờ", "choi", "moi", "mỗi",
    "tôi", "toi", "anh", "chi", "chị", "chú", "chu", "cô", "co", "bác", "bac",
}

GENERIC_PATTERNS = [
    r"^bac si$",
    r"^duoc si$",
    r"^cho em hoi$",
    r"^em hoi$",
    r"^lam sao de biet$",
    r"^co phai la$",
    r"^co can$",
    r"^nho bac si$",
    r"^dau hieu nhan biet$",
    r"^lam the nao de$",
    r"^co nguy hiem khong$",
    r"^phai khong$",
]

GENERIC_TOKENS = {
    "benh", "biet", "cach", "cap", "co", "dau", "di", "dich", "gio", "hay", "khong", "lam", "len", "muc", "nhe", "nho",
    "noi", "ra", "roi", "sau", "se", "thi", "thuong", "tinh", "toi", "trong", "tu", "vao", "voi", "xong", "y",
}

COMMON_CROSS_INTENT_PHRASES = {
    "dau hieu", "dau hieu nhan biet", "lam sao", "lam sao de", "lam sao de biet", "co phai", "co phai la", "co can", "co nguy hiem",
    "biet som", "phong tranh", "benh gi", "canh bao", "nang hon", "dieu tri", "kham ngay", "tri chung", "tai nha",
}


def normalize(text: Any) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_generic_phrase(phrase: str) -> bool:
    if not phrase:
        return True
    if len(phrase) <= 2:
        return True
    if phrase in STOPWORDS or phrase in GENERIC_TOKENS:
        return True
    return any(re.match(pattern, phrase) for pattern in GENERIC_PATTERNS)


def tokenize(question: str) -> List[str]:
    text = normalize(question)
    if not text:
        return []
    tokens = [tok for tok in text.split() if tok]
    tokens = [tok for tok in tokens if tok not in STOPWORDS and tok not in GENERIC_TOKENS and not is_generic_phrase(tok)]
    return tokens


def make_ngrams(tokens: List[str]) -> List[str]:
    grams: List[str] = []
    grams.extend(tokens)
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            if not is_generic_phrase(gram):
                grams.append(gram)
    return grams


def build_documents(payload: Dict[str, Any]) -> Tuple[Dict[str, Counter], Counter]:
    term_counter_by_disease: Dict[str, Counter] = {}
    df_counter: Counter = Counter()

    for category, info in payload.items():
        combined_terms: List[str] = []
        for question in info.get("questions", []):
            combined_terms.extend(make_ngrams(tokenize(question)))
        term_counter = Counter(combined_terms)
        term_counter_by_disease[category] = term_counter
        for term in set(combined_terms):
            df_counter[term] += 1

    return term_counter_by_disease, df_counter


def is_cross_intent_noise(term: str, df_counter: Counter, total_diseases: int) -> bool:
    df = df_counter.get(term, 0)
    if term in COMMON_CROSS_INTENT_PHRASES:
        return True
    if df >= max(15, int(total_diseases * 0.05)):
        return True
    if len(term.split()) == 1 and df >= max(10, int(total_diseases * 0.03)):
        return True
    return False


def score_terms(category: str, counter: Counter, df_counter: Counter, total_diseases: int, display_name: str) -> List[str]:
    scored: List[Tuple[float, str]] = []
    category_norm = normalize(category)
    display_norm = normalize(display_name)
    for term, tf in counter.items():
        if is_generic_phrase(term) or is_cross_intent_noise(term, df_counter, total_diseases):
            continue
        if len(term) < 3:
            continue
        df = df_counter.get(term, 1)
        idf = math.log((total_diseases + 1) / (df + 1)) + 1.0
        score = tf * idf
        if term in display_norm or term in category_norm:
            score += 2.0
        if len(term.split()) >= 2:
            score += 0.35
        scored.append((score, term))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected: List[str] = []
    seen = set()
    for _, term in scored:
        if term in seen:
            continue
        if any(term in existing or existing in term for existing in selected):
            continue
        selected.append(term)
        seen.add(term)
        if len(selected) >= 10:
            break
    return selected


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    term_counter_by_disease, df_counter = build_documents(payload)
    total_diseases = len(payload)

    output: Dict[str, Dict[str, Any]] = {}
    for category, info in payload.items():
        display_name = str(info.get("display_name", category)).strip()
        output[category] = {
            "disease_name": display_name,
            "keywords": score_terms(category, term_counter_by_disease[category], df_counter, total_diseases, display_name),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_path": str(OUTPUT_PATH), "total_diseases": total_diseases}, ensure_ascii=False))


if __name__ == "__main__":
    main()

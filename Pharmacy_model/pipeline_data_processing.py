from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
BRONZE_DIR = BASE_DIR / "data" / "bronze" / "products"
SILVER_DIR = BASE_DIR / "data" / "silver"
OUTPUT_CLEANED_PRODUCTS = BASE_DIR / "products_cleaned.csv"
OUTPUT_DISEASE_CATEGORY_MAP = BASE_DIR / "disease_category_map.csv"

PRODUCT_FILES = [
    BRONZE_DIR / "longchau_thuoc.json",
    BRONZE_DIR / "longchau_thuoc_full.json",
    BRONZE_DIR / "longchau_duocmypham_full.json",
    BRONZE_DIR / "longchau_thietbiyte_full.json",
    BRONZE_DIR / "longchau_chamsoccanhan_full.json",
    BRONZE_DIR / "longchau_thucphamchucnang_full.json",
]

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


@dataclass
class ProductRecord:
    sku: str
    product_name: str
    category_key: str
    category_name: str
    sale_price: str
    ingredients: str
    usage: str
    dosage: str
    description: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def load_categories_config() -> Dict[str, str]:
    return read_json(SILVER_DIR / "categories_config.json")


def load_disease_qa() -> Dict[str, Dict[str, Any]]:
    return read_json(SILVER_DIR / "synthetic_medical_qa.json")


def clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("name"):
                parts.append(normalize_spaces(item["name"]))
            elif isinstance(item, str):
                parts.append(normalize_spaces(item))
        deduped: List[str] = []
        seen = set()
        for part in parts:
            if part and part not in seen:
                seen.add(part)
                deduped.append(part)
        return ", ".join(deduped)
    return normalize_spaces(str(value))


def clean_and_build_products(files: List[Path], categories_config: Dict[str, str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    excluded_price_values = {"", "N/A", "None", "null", "0"}

    for path in files:
        if not path.exists():
            print(f"Bỏ qua file không tồn tại: {path}")
            continue

        data = read_json(path)
        if not isinstance(data, dict):
            print(f"Bỏ qua file không đúng định dạng dict: {path}")
            continue

        for category_key, items in data.items():
            if category_key not in categories_config:
                continue
            if not isinstance(items, list):
                continue

            category_name = categories_config[category_key]
            for item in items:
                if not isinstance(item, dict):
                    continue

                is_prescription = normalize_spaces(str(item.get("is_prescription", "")))
                if "Thuốc kê đơn" in is_prescription or "ETC" in is_prescription:
                    continue

                sale_price = normalize_spaces(str(item.get("sale_price", "")))
                if sale_price in excluded_price_values:
                    continue

                sku = normalize_spaces(str(item.get("sku", "")))
                product_name = normalize_spaces(str(item.get("product_name", "")))
                if not sku or not product_name:
                    continue

                records.append(
                    {
                        "sku": sku,
                        "product_name": product_name,
                        "category_key": category_key,
                        "category_name": category_name,
                        "sale_price": sale_price,
                        "ingredients": clean_text_value(item.get("ingredients_raw") or item.get("ingredients")),
                        "usage": clean_text_value(item.get("usage")),
                        "dosage": clean_text_value(item.get("dosage")),
                        "description": clean_text_value(item.get("description")),
                    }
                )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["sku"], keep="first").reset_index(drop=True)
    return df


def build_disease_category_with_ollama(
    disease_data: Dict[str, Dict[str, Any]],
    categories_config: Dict[str, str],
    model_name: str = "llama3",
) -> pd.DataFrame:
    valid_category_keys = set(categories_config.keys())
    results: List[Dict[str, Any]] = []

    system_prompt = (
        "Bạn là một Data Labeling Agent chuyên nghiệp trong ngành Dược điện tử (E-Pharmacy).\n"
        "Nhiệm vụ: (1) Đọc display_name thường là không dấu và danh sách câu hỏi triệu chứng để suy luận, chuyển đổi tên bệnh sang tiếng Việt CÓ DẤU chuẩn y khoa. "
        "TUYỆT ĐỐI KHÔNG DÙNG TIẾNG ANH (Ví dụ: không được đổi thành 'Diabetes' mà phải là 'Bệnh tiểu đường'). "
        "(2) Ánh xạ bệnh lý sang các Category Slugs phù hợp nhất.\n\n"
        "DANH SÁCH CATEGORY SLUGS HỢP LỆ:\n"
        f"{json.dumps(categories_config, ensure_ascii=False, indent=2)}\n\n"
        "NGUYÊN TẮC:\n"
        "1. Một bệnh có thể liên kết với nhiều danh mục liên quan nhất (Tối đa 3 danh mục).\n"
        "2. CHỈ CHỌN slug có mặt trong danh sách hợp lệ phía trên. Tuyệt đối không tự bịa ra slug mới.\n"
        "3. Bạn PHẢI trả về dữ liệu dưới định dạng JSON duy nhất theo cấu trúc bắt buộc sau, không kèm lời giải thích:\n"
        "{\"normalized_disease_name\": \"Tên bệnh có dấu chuẩn\", \"matched_category_keys\": [\"slug-1\", \"slug-2\"]}"
    )

    filtered_diseases = [(disease_code, info) for disease_code, info in disease_data.items() if "INTENT" not in disease_code]
    total_diseases = min(len(filtered_diseases), 3)
    current_count = 0

    for disease_code, info in filtered_diseases[:3]:
        current_count += 1
        display_name = normalize_spaces(str(info.get("display_name", "")))
        print(f"🔄 [{current_count}/{total_diseases}] Agent đang phân tích bệnh: {display_name} ({disease_code})...")
        questions = info.get("questions", []) if isinstance(info.get("questions", []), list) else []
        questions_text = "\n".join(f"- {normalize_spaces(str(q))}" for q in questions[:10])

        user_prompt = (
            f"Mã bệnh: {disease_code}\n"
            f"Tên thô (gốc): {display_name}\n"
            f"Các câu hỏi triệu chứng:\n{questions_text}\n\n"
            "Hãy suy luận tên bệnh có dấu chuẩn và chọn tối đa 3 category slug phù hợp nhất."
        )

        normalized_disease_name = display_name
        matched_category_keys: List[str] = []
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = response.choices[0].message.content if response.choices else None
            if content:
                parsed = json.loads(content)
                normalized_disease_name = normalize_spaces(str(parsed.get("normalized_disease_name", display_name))) or display_name
                raw_keys = parsed.get("matched_category_keys", [])
                if isinstance(raw_keys, list):
                    for key in raw_keys:
                        key = normalize_spaces(str(key))
                        if key in valid_category_keys and key not in matched_category_keys:
                            matched_category_keys.append(key)
                        if len(matched_category_keys) >= 3:
                            break
        except Exception as exc:
            print(f"Lỗi Ollama với {disease_code}: {exc}")

        if matched_category_keys:
            for slug in matched_category_keys:
                results.append(
                    {
                        "disease_code": disease_code,
                        "disease_name": normalized_disease_name,
                        "category_key": slug,
                    }
                )
        else:
            results.append(
                {
                    "disease_code": disease_code,
                    "disease_name": normalized_disease_name,
                    "category_key": "",
                }
            )

    return pd.DataFrame(results)


def main() -> None:
    categories_config = load_categories_config()
    disease_data = load_disease_qa()

    print("Đang làm sạch sản phẩm...")
    cleaned_products_df = clean_and_build_products(PRODUCT_FILES, categories_config)
    cleaned_products_df.to_csv(OUTPUT_CLEANED_PRODUCTS, index=False, encoding="utf-8-sig")
    print(f"-> Hoàn thành làm sạch. Xuất ra {len(cleaned_products_df)} sản phẩm sạch.")
    print(f"Đã lưu file: {OUTPUT_CLEANED_PRODUCTS}")

    print("Đang gán nhãn bệnh lý - category bằng Ollama local...")
    disease_category_df = build_disease_category_with_ollama(
        disease_data=disease_data,
        categories_config=categories_config,
        model_name="llama3",
    )
    disease_category_df = disease_category_df[["disease_code", "disease_name", "category_key"]]
    disease_category_df.to_csv(OUTPUT_DISEASE_CATEGORY_MAP, index=False, encoding="utf-8-sig")
    print(f"-> Hoàn thành gán nhãn. Xuất ra {len(disease_category_df)} dòng quan hệ Disease-Category.")
    print(f"Đã lưu file: {OUTPUT_DISEASE_CATEGORY_MAP}")


if __name__ == "__main__":
    main()

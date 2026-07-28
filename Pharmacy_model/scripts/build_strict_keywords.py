from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "silver" / "extracted_keywords_top10.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "silver" / "extracted_keywords_top10_strict.json"

MIN_KEYWORDS = 5


def normalize(text: Any) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def is_discriminative(term: str, disease_name: str) -> bool:
    term_norm = normalize(term).lower()
    disease_norm = normalize(disease_name).lower()
    if not term_norm:
        return False
    if len(term_norm) <= 2:
        return False
    if term_norm == disease_norm:
        return True
    return len(term_norm.split()) >= 2


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        payload: Dict[str, Dict[str, Any]] = json.load(f)

    strict_output: Dict[str, Dict[str, Any]] = {}
    for category, item in payload.items():
        disease_name = normalize(item.get("disease_name", category))
        keywords: List[str] = [normalize(kw) for kw in item.get("keywords", []) if is_discriminative(str(kw), disease_name)]
        if len(keywords) > 10:
            keywords = keywords[:10]
        if len(keywords) < MIN_KEYWORDS:
            fallback = [kw for kw in item.get("keywords", []) if kw not in keywords]
            keywords.extend(fallback[: max(0, MIN_KEYWORDS - len(keywords))])
        strict_output[category] = {
            "disease_name": disease_name,
            "keywords": keywords[:10],
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(strict_output, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_path": str(OUTPUT_PATH), "count": len(strict_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

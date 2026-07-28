from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QA_PATH = PROJECT_ROOT / "data" / "silver" / "synthetic_medical_qa.json"
KEYWORDS_PATH = PROJECT_ROOT / "data" / "silver" / "extracted_keywords_top10.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "silver" / "graph_edges.csv"


def normalize(text: Any) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def main() -> None:
    with QA_PATH.open("r", encoding="utf-8") as f:
        qa_payload: Dict[str, Dict[str, Any]] = json.load(f)
    with KEYWORDS_PATH.open("r", encoding="utf-8") as f:
        kw_payload: Dict[str, Dict[str, Any]] = json.load(f)

    rows: List[Dict[str, str]] = []
    for category, item in qa_payload.items():
        source = normalize(item.get("display_name", category))
        keywords = [normalize(kw) for kw in kw_payload.get(category, {}).get("keywords", [])]
        for kw in keywords:
            rows.append({
                "Source": source,
                "Target": kw,
                "Relationship": "HAS_SYMPTOM",
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Source", "Target", "Relationship"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"output_path": str(OUTPUT_PATH), "edges": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

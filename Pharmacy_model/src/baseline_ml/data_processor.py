from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class MedicalSample:
    text: str
    label: str


class MedicalDataProcessor:
    def __init__(self, qa_path: str | Path, keywords_path: str | Path | None = None):
        self.qa_path = Path(qa_path)
        self.keywords_path = Path(keywords_path) if keywords_path else None

    @staticmethod
    def normalize_text(text: str) -> str:
        return unicodedata.normalize("NFC", str(text)).strip()

    @staticmethod
    def clean_and_tokenize(text: str) -> List[str]:
        normalized = unicodedata.normalize("NFC", str(text)).lower()
        normalized = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", normalized, flags=re.UNICODE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return []
        tokens = [tok for tok in normalized.split() if tok]
        ngrams: List[str] = []
        ngrams.extend(tokens)
        for n in (2, 3):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i : i + n]))
        return [unicodedata.normalize("NFC", re.sub(r"\s+", " ", ng).strip()) for ng in ngrams if ng.strip()]

    def _load_keywords_map(self) -> Dict[str, List[str]]:
        if not self.keywords_path or not self.keywords_path.exists():
            return {}
        with self.keywords_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return {category: [self.normalize_text(kw) for kw in item.get("keywords", [])] for category, item in payload.items()}

    def load_samples(self) -> Tuple[List[str], List[str], List[dict]]:
        with self.qa_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        keywords_map = self._load_keywords_map()
        texts: List[str] = []
        labels: List[str] = []
        records: List[dict] = []

        for category, info in payload.items():
            display_name = info.get("display_name", category)
            keywords = keywords_map.get(category, [])
            for question in info.get("questions", []):
                tokens = self.clean_and_tokenize(question)
                tokenized = " ".join(tokens)
                merged_text = " ".join([tokenized] + keywords) if keywords else tokenized
                texts.append(merged_text)
                labels.append(category)
                records.append(
                    {
                        "text": merged_text,
                        "label": category,
                        "display_name": display_name,
                        "raw_question": question,
                        "keywords": keywords,
                    }
                )

        return texts, labels, records


def load_medical_dataset(qa_path: str | Path):
    return MedicalDataProcessor(qa_path).load_samples()

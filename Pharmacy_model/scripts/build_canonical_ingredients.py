from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "silver"
PRODUCTS_BACKUP_PATH = DATA_DIR / "products_cleaned_backup.csv"
INGREDIENT_MAPPING_PATH = DATA_DIR / "ingredient_mapping.csv"
OUTPUT_PRODUCTS_PATH = DATA_DIR / "products_cleaned_canonical.csv"
OUTPUT_MAPPING_PATH = DATA_DIR / "ingredient_mapping_canonical.csv"


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\b(hcl|hydrochloride|hydroclorid|hydrochlorid|dihydrochloride|dihydrochlorid)\b", " ", text)
    text = re.sub(r"\b(n/a|na|none|null)\b", " ", text)
    text = re.sub(r"[^\w\s\u00c0-\u1ef9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_ingredients(raw: str) -> list[str]:
    if not raw or str(raw).strip().upper() == "N/A":
        return []
    text = str(raw)
    text = text.replace("|", ",")
    parts = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
    return parts


def load_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = normalize_text(row.get("raw_ingredient", ""))
            canonical = str(row.get("canonical_ingredient", "")).strip()
            if raw and canonical:
                mapping[raw] = canonical
    return mapping


def canonicalize_ingredient(raw_ingredient: str, mapping: dict[str, str]) -> str:
    normalized = normalize_text(raw_ingredient)
    return mapping.get(normalized, raw_ingredient.strip())


def build_canonical_ingredients(products_path: Path, mapping: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with products_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "canonical_ingredients" not in fieldnames:
            fieldnames.append("canonical_ingredients")
        for row in reader:
            ingredients = split_ingredients(row.get("ingredients", ""))
            canonical_items = [canonicalize_ingredient(item, mapping) for item in ingredients]
            canonical_items = [item for item in canonical_items if item]
            row["canonical_ingredients"] = "|".join(dict.fromkeys(canonical_items))
            rows.append(row)
    return rows


def write_products(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dedupe_mapping_file(mapping_path: Path, output_path: Path) -> tuple[int, int]:
    seen: dict[str, str] = {}
    duplicates = 0
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = normalize_text(row.get("raw_ingredient", ""))
            canonical = str(row.get("canonical_ingredient", "")).strip()
            if not raw or not canonical:
                continue
            if raw in seen and seen[raw] != canonical:
                duplicates += 1
                continue
            seen[raw] = canonical
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_ingredient", "canonical_ingredient"])
        writer.writeheader()
        for raw, canonical in sorted(seen.items()):
            writer.writerow({"raw_ingredient": raw, "canonical_ingredient": canonical})
    return len(seen), duplicates


def main() -> None:
    mapping = load_mapping(INGREDIENT_MAPPING_PATH)
    rows = build_canonical_ingredients(PRODUCTS_BACKUP_PATH, mapping)
    write_products(rows, OUTPUT_PRODUCTS_PATH)
    kept, duplicates = dedupe_mapping_file(INGREDIENT_MAPPING_PATH, OUTPUT_MAPPING_PATH)
    print(f"Wrote: {OUTPUT_PRODUCTS_PATH}")
    print(f"Wrote: {OUTPUT_MAPPING_PATH}")
    print(f"Mapping entries kept: {kept}")
    print(f"Mapping duplicates skipped: {duplicates}")


if __name__ == "__main__":
    main()

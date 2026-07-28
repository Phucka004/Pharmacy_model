from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import unicodedata

import pandas as pd

DEBUG_SAFETY = True

from .user_session import UserProfile

INGREDIENT_WARNING_PATH = Path(__file__).resolve().parents[3] / "data" / "silver" / "ingredient_warning.csv"

WARNING_QUESTION_MAP: dict[str, str] = {
    "Age": "Bạn bao nhiêu tuổi?",
    "Pregnancy": "Bạn hiện đang mang thai không?",
    "Breastfeeding": "Bạn đang cho con bú không?",
    "Allergy": "Bạn có dị ứng với thuốc hoặc hoạt chất nào không?",
    "ChronicDisease": "Bạn có bệnh nền như suy gan, suy thận, động kinh, tăng nhãn áp... không?",
}

WARNING_TYPE_PRIORITY = ["Age", "Pregnancy", "Breastfeeding", "Allergy", "ChronicDisease"]
DEBUG_SAFETY = True
AGE_PATTERN = re.compile(r"(\d+)\s*tuổi|tuổi\s*(\d+)", flags=re.IGNORECASE)


def _safe_debug_text(value: Any) -> str:
    text = str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _debug_print(value: Any) -> None:
    print(_safe_debug_text(value))


@dataclass(frozen=True)
class ProductWarning:
    ingredient: str
    warning_type: str
    condition: str
    level: str
    note: str = ""
    source: str = ""
    confidence: str = ""


@dataclass(frozen=True)
class TriggerResult:
    matched: bool
    reason: str = ""


class SafetyChecker:
    """Safety layer applied after the RAG ranking step.

    The class only consumes the Top-K products and user profile. This keeps the
    retrieval, disease detection, and ranking pipeline unchanged.
    """

    def __init__(self, warning_source: Path | None = None) -> None:
        self.warning_source = warning_source or INGREDIENT_WARNING_PATH
        self.warning_df = self._load_warning_df()

    def _load_warning_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.warning_source, engine="python", on_bad_lines="skip").fillna("")
        required = {"ingredient", "warning_type", "condition", "level", "note", "source", "confidence"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"ingredient_warning.csv thiếu cột: {', '.join(sorted(missing))}")
        for col in required:
            df[col] = df[col].astype(str).str.strip()
        return df

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return str(value or "").strip().lower()

    # @staticmethod
    # def _extract_ingredients(product: Any) -> list[str]:
    #     if isinstance(product, dict):
    #         raw = product.get("ingredients") or product.get("ingredient") or ""
    #     else:
    #         raw = getattr(product, "ingredients", "") or getattr(product, "ingredient", "") or ""
    #     if isinstance(raw, list):
    #         return [str(item).strip() for item in raw if str(item).strip()]
    #     text = str(raw).strip()
    #     if not text or text.upper() == "N/A":
    #         return []
    #     parts = [part.strip() for part in text.replace(";", ",").split(",")]
    #     return [part for part in parts if part]
    @staticmethod
    def _extract_ingredients(product: Any) -> list[str]:
        if isinstance(product, dict):
            raw = product.get("canonical_ingredients") or product.get("ingredients") or product.get("ingredient") or ""
        else:
            raw = getattr(product, "canonical_ingredients", "") or getattr(product, "ingredients", "") or getattr(product, "ingredient", "") or ""

        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]

        text = str(raw).strip()
        if not text or text.upper() == "N/A":
            return []
        if "|" in text:
            return [item.strip() for item in text.split("|") if item.strip()]
        return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]

    def _normalize_ingredient_name(self, ingredient: str) -> str:
        text = self._normalize_text(ingredient)
        text = re.sub(r"\b(hcl|hydrochloride|hydroclorid|dihydrochloride|hydrochlorid)\b", " ", text)
        text = re.sub(r"[^\w\sà-ỹ]+", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _rows_for_ingredient(self, ingredient: str) -> pd.DataFrame:
        ingredient_norm = self._normalize_ingredient_name(ingredient)
        normalized_columns = self.warning_df["ingredient"].astype(str).map(self._normalize_ingredient_name)
        return self.warning_df.loc[normalized_columns == ingredient_norm]

    def get_required_question_types(self, products: Iterable[Any]) -> list[str]:
        warning_types: list[str] = []
        for product in products:
            for ingredient in self._extract_ingredients(product):
                matches = self._rows_for_ingredient(ingredient)
                warning_types.extend(matches["warning_type"].tolist())
        ordered = [warning_type for warning_type in WARNING_TYPE_PRIORITY if warning_type in warning_types]
        ordered.extend(warning_type for warning_type in warning_types if warning_type not in ordered)
        return [warning_type for warning_type in ordered if warning_type in WARNING_QUESTION_MAP]

    def build_questions(self, question_types: Iterable[str]) -> list[str]:
        unique_types = list(dict.fromkeys(question_types))
        questions = [WARNING_QUESTION_MAP[q_type] for q_type in unique_types if q_type in WARNING_QUESTION_MAP]
        if DEBUG_SAFETY:
            _debug_print("=== QUESTIONS ===")
            _debug_print(questions)
        return questions

    def get_required_questions(self, products: Iterable[Any]) -> list[str]:
        return self.build_questions(self.get_required_question_types(products))

    def get_product_warnings(self, product: Any, profile: UserProfile) -> list[dict[str, str]]:
        ingredients = self._extract_ingredients(product)
        canonical_ingredients = [self._normalize_ingredient_name(item) for item in ingredients]

        hard_warnings = self._get_hard_safety_warnings(product, ingredients, canonical_ingredients, profile)
        if hard_warnings:
            if DEBUG_SAFETY:
                _debug_print("=== MATCHED RULES ===")
                for warning in hard_warnings:
                    _debug_print(warning)
            return hard_warnings

        warnings: list[dict[str, str]] = []
        for ingredient, ingredient_norm in zip(ingredients, canonical_ingredients):
            matches = self._rows_for_ingredient(ingredient)
            for _, row in matches.iterrows():
                warning_type = str(row.get("warning_type", "")).strip()
                condition = str(row.get("condition", "")).strip()
                level = str(row.get("level", "")).strip()
                trigger = self._is_triggered(warning_type, condition, profile)
                if trigger.matched:
                    warnings.append(
                        {
                            "ingredient": ingredient,
                            "ingredient_norm": ingredient_norm,
                            "warning_type": warning_type,
                            "condition": condition,
                            "level": level,
                            "note": str(row.get("note", "")).strip(),
                            "source": str(row.get("source", "")).strip(),
                            "confidence": str(row.get("confidence", "")).strip(),
                            "reason": trigger.reason,
                        }
                    )

        if DEBUG_SAFETY and warnings:
            _debug_print("=== MATCHED RULES ===")
            for warning in warnings:
                _debug_print(warning)
        return warnings

    def filter_products(self, products: list[Any], profile: UserProfile) -> tuple[list[Any], list[dict[str, str]], list[dict[str, str]]]:
        kept: list[Any] = []
        removed: list[dict[str, str]] = []
        collected_warnings: list[dict[str, str]] = []
        if DEBUG_SAFETY:
            _debug_print("=== PRODUCTS ===")
            _debug_print([self._product_name(product) for product in products])
        for product in products:
            warnings = self.get_product_warnings(product, profile)
            if warnings:
                collected_warnings.extend(warnings)
            levels = {warning["level"].strip().lower() for warning in warnings}
            if levels.intersection({"contraindicated", "avoid"}):
                removed.extend(
                    {
                        "product": self._product_name(product),
                        "reason": warning["level"],
                        "ingredient": warning["ingredient"],
                    }
                    for warning in warnings
                    if warning["level"].strip().lower() in {"contraindicated", "avoid"}
                )
                continue
            kept.append(product)
        if DEBUG_SAFETY:
            print("=== REMOVED PRODUCTS ===")
            print(removed)
            print("=== WARNINGS ===")
            print(collected_warnings)
        return kept, removed, collected_warnings

    def _is_triggered(self, warning_type: str, condition: str, profile: UserProfile) -> TriggerResult:
        warning_type = warning_type.strip()
        condition_norm = self._normalize_text(condition)

        if warning_type == "Age":
            return self._match_age_condition(condition_norm, profile.age)
        if warning_type == "Pregnancy":
            return self._match_boolean_condition(condition_norm, profile.pregnant, ["thai", "mang thai", "phụ nữ có thai", "phu nu co thai"])
        if warning_type == "Breastfeeding":
            return self._match_boolean_condition(condition_norm, profile.breastfeeding, ["cho con bú", "cho con bu", "đang cho con bú"])
        if warning_type == "Allergy":
            return self._match_allergy_condition(condition_norm, profile.allergies)
        if warning_type == "ChronicDisease":
            return self._match_disease_condition(condition_norm, profile.chronic_diseases)
        return TriggerResult(False)

    @staticmethod
    def _product_name(product: Any) -> str:
        if isinstance(product, dict):
            return str(product.get("product_name", "")).strip()
        return str(getattr(product, "product_name", "")).strip()

    @staticmethod
    def _match_age_condition(condition: str, age: int | None) -> TriggerResult:
        if age is None:
            return TriggerResult(False)
        condition_ascii = _safe_debug_text(condition).lower()
        match = AGE_PATTERN.search(condition)
        if not match:
            if "nguoi cao tuoi" in condition_ascii or "nguoi gia" in condition_ascii:
                return TriggerResult(age >= 65, condition)
            if "tre em" in condition_ascii:
                return TriggerResult(age < 18, condition)
            return TriggerResult(False)
        threshold = int(next(group for group in match.groups() if group is not None))
        if "duoi" in condition_ascii or "<" in condition_ascii:
            return TriggerResult(age < threshold, condition)
        if "tren" in condition_ascii or ">" in condition_ascii:
            return TriggerResult(age > threshold, condition)
        return TriggerResult(age == threshold, condition)

    @staticmethod
    def _match_boolean_condition(condition: str, value: bool | None, keywords: list[str]) -> TriggerResult:
        if value is None:
            return TriggerResult(False)
        if not any(keyword in condition for keyword in keywords):
            return TriggerResult(False)
        return TriggerResult(bool(value), condition)

    @staticmethod
    def _match_allergy_condition(condition: str, allergies: list[str]) -> TriggerResult:
        if not allergies:
            return TriggerResult(False)
        if not condition:
            return TriggerResult(True, "allergy")
        condition_tokens = {token.strip() for token in re.split(r"[,;/]", condition) if token.strip()}
        allergy_tokens = {str(item).strip().lower() for item in allergies}
        if not condition_tokens:
            return TriggerResult(True, condition)
        matched = any(token.lower() in allergy_tokens or any(token.lower() in allergy for allergy in allergy_tokens) for token in condition_tokens)
        return TriggerResult(matched, condition)

    @staticmethod
    def _match_disease_condition(condition: str, chronic_diseases: list[str]) -> TriggerResult:
        if not chronic_diseases:
            return TriggerResult(False)
        disease_tokens = {_safe_debug_text(str(item).strip()).lower() for item in chronic_diseases}
        condition_ascii = _safe_debug_text(condition).lower()
        if not condition_ascii:
            return TriggerResult(True, "chronic_disease")
        matched = any(token in condition_ascii or condition_ascii in token for token in disease_tokens)
        return TriggerResult(matched, condition)

    def _build_hard_warning(
        self,
        product_name: str,
        warning_type: str,
        reason: str,
        hits: list[dict[str, str]],
    ) -> dict[str, str]:
        ingredients = [str(hit.get("ingredient", "")).strip() for hit in hits if str(hit.get("ingredient", "")).strip()]
        unique_ingredients = list(dict.fromkeys(ingredients))
        level_map = {
            "Age": "Avoid",
            "Pregnancy": "Contraindicated",
            "Breastfeeding": "Contraindicated",
            "Allergy": "Contraindicated",
            "ChronicDisease": "Contraindicated",
        }
        condition_map = {
            "Age": "Age restriction",
            "Pregnancy": "Pregnancy restriction",
            "Breastfeeding": "Breastfeeding restriction",
            "Allergy": "User allergy",
            "ChronicDisease": "Chronic disease restriction",
        }
        return {
            "product": product_name,
            "ingredient": ", ".join(unique_ingredients),
            "ingredient_norm": ", ".join(
                str(hit.get("ingredient_norm", "")).strip() for hit in hits if str(hit.get("ingredient_norm", "")).strip()
            ),
            "warning_type": warning_type,
            "condition": condition_map.get(warning_type, warning_type),
            "level": level_map.get(warning_type, "Contraindicated"),
            "note": reason,
            "source": "products_cleaned",
            "confidence": "High",
            "reason": reason,
        }

    def _get_hard_safety_warnings(
        self,
        product: Any,
        ingredients: list[str],
        canonical_ingredients: list[str],
        profile: UserProfile,
    ) -> list[dict[str, str]]:
        warnings: list[dict[str, str]] = []
        product_name = self._product_name(product)

        allergy_hits = self._match_profile_allergies(ingredients, canonical_ingredients, profile.allergies)
        if allergy_hits:
            warnings.append(self._build_hard_warning(product_name, "Allergy", "User allergy matched ingredient", allergy_hits))
            return warnings

        age_hits = self._match_profile_age(ingredients, canonical_ingredients, profile.age)
        if age_hits:
            warnings.append(self._build_hard_warning(product_name, "Age", "Age restriction matched ingredient", age_hits))
            return warnings

        if profile.pregnant is True:
            pregnancy_hits = self._match_profile_condition(ingredients, canonical_ingredients, "Pregnancy")
            if pregnancy_hits:
                warnings.append(self._build_hard_warning(product_name, "Pregnancy", "Pregnancy restriction matched ingredient", pregnancy_hits))
                return warnings

        if profile.breastfeeding is True:
            breastfeeding_hits = self._match_profile_condition(ingredients, canonical_ingredients, "Breastfeeding")
            if breastfeeding_hits:
                warnings.append(self._build_hard_warning(product_name, "Breastfeeding", "Breastfeeding restriction matched ingredient", breastfeeding_hits))
                return warnings

        if profile.chronic_diseases:
            chronic_hits = self._match_profile_chronic_disease(ingredients, canonical_ingredients, profile.chronic_diseases)
            if chronic_hits:
                warnings.append(self._build_hard_warning(product_name, "ChronicDisease", "Chronic disease restriction matched ingredient", chronic_hits))
                return warnings

        return warnings

    @staticmethod
    def _normalize_ingredient_name_static(ingredient: str) -> str:
        text = str(ingredient or "").strip().lower()
        text = re.sub(r"\b(hcl|hydrochloride|hydroclorid|dihydrochloride|hydrochlorid)\b", " ", text)
        text = re.sub(r"[^\w\sà-ỹ]+", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _dedupe_preserve_display(values: list[str]) -> list[dict[str, str]]:
        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for value in values:
            display = str(value).strip()
            normalized = SafetyChecker._normalize_ingredient_name_static(display)
            if not display or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append({"display": display, "normalized": normalized})
        return deduped

    @staticmethod
    def _match_profile_allergies(ingredients: list[str], canonical_ingredients: list[str], allergies: list[str]) -> list[dict[str, str]]:
        if not allergies:
            return []
        hits: list[dict[str, str]] = []
        ingredient_pool = SafetyChecker._dedupe_preserve_display([*ingredients, *canonical_ingredients])
        allergy_norms = {SafetyChecker._normalize_ingredient_name_static(allergy) for allergy in allergies if SafetyChecker._normalize_ingredient_name_static(allergy)}
        for entry in ingredient_pool:
            if entry["normalized"] in allergy_norms:
                hits.append({"ingredient": entry["display"], "ingredient_norm": entry["normalized"]})
        return hits

    @staticmethod
    def _match_profile_age(ingredients: list[str], canonical_ingredients: list[str], age: int | None) -> list[dict[str, str]]:
        if age is None:
            return []
        hits: list[dict[str, str]] = []
        ingredient_pool = SafetyChecker._dedupe_preserve_display([*ingredients, *canonical_ingredients])
        for entry in ingredient_pool:
            ingredient = entry["display"]
            if ingredient and re.search(r"\b(12|6)\b", ingredient):
                if age < 12:
                    hits.append({"ingredient": ingredient, "condition": "Age < 12"})
        return hits

    def _match_profile_condition(self, ingredients: list[str], canonical_ingredients: list[str], warning_type: str) -> list[dict[str, str]]:
        hits: list[dict[str, str]] = []
        ingredient_pool = SafetyChecker._dedupe_preserve_display([*ingredients, *canonical_ingredients])
        for entry in ingredient_pool:
            ingredient = entry["display"]
            matches = self._rows_for_ingredient(ingredient)
            for _, row in matches.iterrows():
                if str(row.get("warning_type", "")).strip() == warning_type:
                    hits.append({"ingredient": ingredient, "condition": str(row.get("condition", "")).strip()})
        return hits

    def _match_profile_chronic_disease(self, ingredients: list[str], canonical_ingredients: list[str], chronic_diseases: list[str]) -> list[dict[str, str]]:
        hits: list[dict[str, str]] = []
        disease_tokens = {_safe_debug_text(str(item).strip()).lower() for item in chronic_diseases}
        ingredient_pool = SafetyChecker._dedupe_preserve_display([*ingredients, *canonical_ingredients])
        for entry in ingredient_pool:
            ingredient = entry["display"]
            matches = self._rows_for_ingredient(ingredient)
            for _, row in matches.iterrows():
                condition = _safe_debug_text(str(row.get("condition", ""))).lower()
                if any(token in condition or condition in token for token in disease_tokens):
                    hits.append({"ingredient": ingredient, "condition": str(row.get("condition", "")).strip()})
        return hits

    def should_ask_questions(self, products: list[Any]) -> bool:
        return bool(self.get_required_questions(products))

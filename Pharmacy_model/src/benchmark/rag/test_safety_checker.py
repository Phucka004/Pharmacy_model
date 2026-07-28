from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.rag.safety_checker import SafetyChecker
from src.benchmark.rag.user_session import UserProfile


def _safe(value: object) -> str:
    text = str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _print(value: object) -> None:
    print(_safe(value))


def _assert(name: str, condition: bool, detail: str) -> None:
    if condition:
        _print(f"PASS {name}: {detail}")
        return
    _print(f"FAIL {name}: {detail}")
    raise SystemExit(1)


def main() -> None:
    checker = SafetyChecker()

    _print("========== PRODUCTS ==========")
    products = [
        {
            "name": "Thuốc A",
            "product_name": "Thuốc A",
            "ingredients": ["Cetirizine", "Phenylephrine"],
        }
    ]
    _print(products)

    _print("========== INGREDIENTS ==========")
    ingredients = checker._extract_ingredients(products[0])
    _print(ingredients)

    _print("========== NORMALIZED INGREDIENTS ==========")
    normalized = [checker._normalize_ingredient_name(item) for item in ingredients]
    _print(normalized)

    _print("========== MATCHED RULES ==========")
    matched_rules = []
    for ingredient in ingredients:
        rows = checker._rows_for_ingredient(ingredient)
        if rows.empty:
            _print({"ingredient": ingredient, "reason": "no matching rule"})
            continue
        for _, row in rows.iterrows():
            payload = {
                "ingredient": ingredient,
                "warning_type": str(row["warning_type"]),
                "condition": str(row["condition"]),
                "level": str(row["level"]),
            }
            matched_rules.append(payload)
            _print(payload)

    question_types = checker.get_required_question_types(products)
    questions = checker.build_questions(question_types)
    _print("========== GENERATED QUESTIONS ==========")
    for question_type in question_types:
        _print(f"{question_type} -> {checker.build_questions([question_type])[0]}")
    _assert("generate_questions_non_empty", len(questions) > 0, str(questions))

    age_rules = [rule for rule in matched_rules if rule["warning_type"] == "Age"]
    _assert("age_question_present", "Bạn bao nhiêu tuổi?" in questions, str(questions))

    _print("========== PARSER CONDITION ==========")
    _assert("age_lt_6_true", checker._match_age_condition("Trẻ em dưới 6 tuổi", 5).matched is True, "age=5")
    _assert("age_lt_6_false", checker._match_age_condition("Trẻ em dưới 6 tuổi", 6).matched is False, "age=6")
    _assert("age_elderly_true", checker._match_age_condition("Người cao tuổi", 70).matched is True, "age=70")
    _assert("kidney_match_true", checker._match_disease_condition("Suy thận", ["suy thận"]).matched is True, "chronic_diseases=['suy thận']")

    _print("========== MULTI INGREDIENT FILTER ==========")
    multi_product = {
        "name": "Thuốc B",
        "product_name": "Thuốc B",
        "ingredients": ["Paracetamol", "Cetirizine", "Phenylephrine"],
    }
    severe_profile = UserProfile(age=5)
    kept_products, removed_products, warnings = checker.filter_products([multi_product], severe_profile)
    _print({"kept_products": kept_products, "removed_products": removed_products, "warnings": warnings})
    _assert("multi_ingredient_removed", len(removed_products) > 0, str(removed_products))
    _assert("multi_ingredient_product_removed", len(kept_products) == 0, str(kept_products))

    _print("========== MULTI RULE FILTER ==========")
    profile = UserProfile(age=5, pregnant=True, breastfeeding=True, allergies=["fexofenadine"], chronic_diseases=["suy thận"])
    multi_rule_product = {
        "name": "Thuốc C",
        "product_name": "Thuốc C",
        "ingredients": ["Fexofenadine"],
    }
    kept_products, removed_products, warnings = checker.filter_products([multi_rule_product], profile)
    _print({"kept_products": kept_products, "removed_products": removed_products, "warnings": warnings})
    _assert("multi_rule_removed", len(removed_products) > 0, str(removed_products))

    caution_profile = UserProfile(age=30, pregnant=True, breastfeeding=False, allergies=[], chronic_diseases=[])
    caution_product = {
        "name": "Thuốc D",
        "product_name": "Thuốc D",
        "ingredients": ["Fexofenadine"],
    }
    kept_products, removed_products, warnings = checker.filter_products([caution_product], caution_profile)
    _print({"kept_products": kept_products, "removed_products": removed_products, "warnings": warnings})
    _assert("caution_kept", len(kept_products) == 1, str(kept_products))
    _assert("caution_warning_present", len(warnings) > 0, str(warnings))


if __name__ == "__main__":
    main()

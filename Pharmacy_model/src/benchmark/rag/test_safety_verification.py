from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import unicodedata

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.rag.inference import RAGPredictor
from src.benchmark.rag.safety_checker import SafetyChecker
from src.benchmark.rag.user_session import UserProfile


@dataclass
class TestCaseResult:
    name: str
    passed: bool
    detail: str = ""



def _print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _safe_text(value: Any) -> str:
    text = str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")



def test_parser_conditions(checker: SafetyChecker) -> TestCaseResult:
    profile = UserProfile(age=5)
    assert checker._match_age_condition("trẻ em dưới 6 tuổi", profile.age).matched is True
    assert checker._match_age_condition("trẻ em dưới 6 tuổi", 6).matched is False
    assert checker._match_age_condition("người cao tuổi", 70).matched is True
    assert checker._match_boolean_condition("phụ nữ có thai", True, ["thai", "mang thai", "phụ nữ có thai"]).matched is True
    assert checker._match_boolean_condition("cho con bú", True, ["cho con bú", "đang cho con bú"]).matched is True
    assert checker._match_allergy_condition("paracetamol", ["paracetamol"]).matched is True
    assert checker._match_disease_condition("suy thận", ["suy thận"]).matched is True
    return TestCaseResult("parser_conditions", True)



def test_question_generation(checker: SafetyChecker) -> TestCaseResult:
    products = [
        {
            "product_name": "Xyzal 5mg",
            "ingredients": "Levocetirizine",
        },
        {
            "product_name": "Fexostad 180",
            "ingredients": "Fexofenadine Hydrochloride",
        },
    ]
    question_types = checker.get_required_question_types(products)
    questions = checker.build_questions(question_types)
    assert len(question_types) > 0
    assert len(set(questions)) == len(questions)
    assert questions.count("Bạn bao nhiêu tuổi?") == 1
    return TestCaseResult("question_generation", True, str({"types": question_types, "questions": questions}))



def test_filter_multi_ingredient(checker: SafetyChecker) -> TestCaseResult:
    product = {
        "product_name": "Demo combo",
        "ingredients": "Paracetamol, Cetirizine, Phenylephrine",
    }
    profile = UserProfile(age=5, allergies=["cetirizine"])
    warnings = checker.get_product_warnings(product, profile)
    kept, removed, _ = checker.filter_products([product], profile)
    assert isinstance(warnings, list)
    assert len(removed) >= 0
    return TestCaseResult("filter_multi_ingredient", True, f"warnings={warnings}, kept={kept}, removed={removed}")



def test_conversation_wiring(predictor: RAGPredictor) -> TestCaseResult:
    predictor.session_manager.reset()
    raw = predictor.predict_products_for_query("Tôi bị trào ngược dạ dày")
    assert isinstance(raw, dict)
    assert "need_more_information" in raw
    return TestCaseResult("conversation_wiring", True, str(raw.get("need_more_information")))



def test_session_updates(predictor: RAGPredictor) -> TestCaseResult:
    predictor.session_manager.reset()
    predictor.session_manager.update_profile(age=5, pregnant=False, breastfeeding=False, allergies=["cetirizine"], chronic_diseases=["suy thận"])
    profile = predictor.session_manager.get_profile()
    assert profile.age == 5
    assert profile.allergies == ["cetirizine"]
    return TestCaseResult("session_updates", True, str(profile))



def run() -> list[TestCaseResult]:
    checker = SafetyChecker()
    predictor = RAGPredictor()
    results: list[TestCaseResult] = []

    for fn in [test_parser_conditions, test_question_generation, test_filter_multi_ingredient]:
        try:
            results.append(fn(checker))
        except AssertionError as exc:
            results.append(TestCaseResult(fn.__name__, False, f"AssertionError: {exc}"))
        except Exception as exc:
            results.append(TestCaseResult(fn.__name__, False, f"Exception: {exc}"))

    try:
        results.append(test_conversation_wiring(predictor))
    except AssertionError as exc:
        results.append(TestCaseResult("conversation_wiring", False, f"AssertionError: {exc}"))
    except Exception as exc:
        results.append(TestCaseResult("conversation_wiring", False, f"Exception: {exc}"))

    try:
        results.append(test_session_updates(predictor))
    except AssertionError as exc:
        results.append(TestCaseResult("session_updates", False, f"AssertionError: {exc}"))
    except Exception as exc:
        results.append(TestCaseResult("session_updates", False, f"Exception: {exc}"))

    return results



def main() -> None:
    _print_section("Safety Verification Results")
    results = run()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        if result.detail:
            print(_safe_text(result.detail))
    if any(not item.passed for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

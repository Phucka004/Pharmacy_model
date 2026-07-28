from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import unicodedata

from src.benchmark.rag.safety_checker import SafetyChecker
from src.benchmark.rag.user_session import SessionManager, UserProfile


@dataclass
class MockPredictor:
    safety_checker: SafetyChecker
    session_manager: SessionManager = field(default_factory=SessionManager)

    def predict_products_for_query(self, _: str) -> dict[str, Any]:
        products = [
            {
                "product_name": "Thuoc A",
                "ingredients": ["Fexofenadine", "Menthol"],
            }
        ]
        pending_types = self.safety_checker.get_required_question_types(products)
        questions = self.safety_checker.build_questions(pending_types)
        self.session_manager.set_pending_questions(pending_types, questions)
        self.session_manager.set_pending_products([item["product_name"] for item in products])
        return {
            "need_more_information": True,
            "products": [item["product_name"] for item in products],
            "questions": questions,
            "pending_question_types": pending_types,
        }

    def handle_safety_answer(self, answer_text: str) -> dict[str, Any]:
        session = self.session_manager.get_session()
        pending = list(session.awaiting_question_types)
        if not pending:
            return {"need_more_information": False}
        if pending[0] == "Age":
            self.session_manager.update_profile(age=int(answer_text.strip()))
        elif pending[0] == "Pregnancy":
            self.session_manager.update_profile(pregnant=answer_text.strip().lower() in {"co", "có", "yes"})
        elif pending[0] == "ChronicDisease":
            self.session_manager.update_profile(chronic_diseases=[answer_text.strip()])
        session.awaiting_question_types = pending[1:]
        session.awaiting_questions = self.safety_checker.build_questions(session.awaiting_question_types)
        return {
            "need_more_information": bool(session.awaiting_question_types),
            "questions": session.awaiting_questions,
            "pending_question_types": session.awaiting_question_types,
            "profile": self.session_manager.get_profile(),
        }

    def apply_safety_layer(self, products: list[str]) -> dict[str, Any]:
        profile = self.session_manager.get_profile()
        product_records = [
            {"product_name": product, "ingredients": ["Fexofenadine", "Menthol"]}
            for product in products
        ]
        pending_types = self.safety_checker.get_required_question_types(product_records)
        if pending_types and profile.age is None:
            self.session_manager.set_pending_questions(pending_types, self.safety_checker.build_questions(pending_types))
            return {
                "need_more_information": True,
                "questions": self.safety_checker.build_questions(pending_types),
                "pending_question_types": pending_types,
            }
        kept, removed, warnings = self.safety_checker.filter_products(product_records, profile)
        return {
            "need_more_information": False,
            "products": [item["product_name"] for item in kept],
            "removed_products": removed,
            "warnings": warnings,
        }


def _print(value: Any) -> None:
    safe = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    print(safe)


def main() -> None:
    checker = SafetyChecker()
    predictor = MockPredictor(checker)

    _print("========== MOCK CONVERSATION ==========")
    _print("User: Toi bi di ung")
    result = predictor.predict_products_for_query("Toi bi di ung")
    _print(result)

    while result.get("need_more_information"):
        for question in result.get("questions", []):
            _print(f"Bot: {question}")
            normalized_question = unicodedata.normalize("NFKD", question).encode("ascii", "ignore").decode("ascii")
            if normalized_question.startswith("Ban bao nhieu tuoi"):
                answer = "5"
            elif normalized_question.startswith("Ban co benh nen"):
                answer = "khong"
            else:
                answer = "khong"
            _print(f"> {answer}")
            result = predictor.handle_safety_answer(answer)
        if not result.get("need_more_information"):
            result = predictor.apply_safety_layer(result.get("products", ["Thuoc A"]))

    _print("========== FINAL PRODUCTS ==========")
    _print(result.get("products"))
    _print("========== REMOVED PRODUCTS ==========")
    _print(result.get("removed_products"))
    _print("========== WARNINGS ==========")
    _print(result.get("warnings"))
    _print("========== FINAL USER PROFILE ==========")
    _print(predictor.session_manager.get_profile())


if __name__ == "__main__":
    main()

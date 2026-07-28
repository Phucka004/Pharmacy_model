from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.benchmark.rag.inference import RAGPredictor


EXIT_COMMANDS = {"exit", "quit", "q"}


def _safe_text(value: Any) -> str:
    return unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")


def _print_result_block(title: str, value: Any) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(_safe_text(value))


def _prompt_user_answer() -> str:
    return input("> ").strip()


def _handle_safety_questions(predictor: RAGPredictor, raw_result: dict[str, Any], original_products: list[str]) -> dict[str, Any]:
    current_result = dict(raw_result)
    while current_result.get("need_more_information"):
        questions = current_result.get("questions", [])
        if not questions:
            break
        question = questions[0]
        print(question)
        answer_text = _prompt_user_answer()
        if answer_text.lower() in EXIT_COMMANDS:
            return {"exit": True}
        current_result = predictor.handle_safety_answer(answer_text)
        if current_result.get("need_more_information"):
            continue
        current_result = predictor.apply_safety_layer(original_products, predictor.session_manager.get_profile())
    return current_result


def _print_final_payload(result: dict[str, Any]) -> None:
    _print_result_block("Top-5 cuối cùng", result.get("products", []))
    _print_result_block("removed_products", result.get("removed_products", []))
    _print_result_block("warnings", result.get("warnings", []))
    _print_result_block("profile người dùng", predictor.session_manager.get_profile())


predictor = RAGPredictor()


def main() -> None:
    print("".ljust(90, "="))
    print("RAG SAFETY CONVERSATION CLI")
    print("Nhập 'exit' để thoát.")
    print("".ljust(90, "="))

    while True:
        user_query = input("User: ").strip()
        if user_query.lower() in EXIT_COMMANDS:
            break
        if not user_query:
            continue

        predictor.session_manager.clear_pending_questions()
        predictor.session_manager.clear_pending_products()
        raw_result = predictor.predict_products_for_query(user_query)

        if not isinstance(raw_result, dict):
            _print_result_block("Kết quả", raw_result)
            continue

        if raw_result.get("need_more_information"):
            final_result = _handle_safety_questions(predictor, raw_result, raw_result.get("products", []))
            if final_result.get("exit"):
                break
            if final_result.get("need_more_information"):
                _print_result_block("Câu hỏi tiếp theo", final_result.get("questions", []))
            else:
                _print_final_payload(final_result)
        else:
            _print_final_payload(raw_result)

    print("Đã thoát CLI.")


if __name__ == "__main__":
    main()

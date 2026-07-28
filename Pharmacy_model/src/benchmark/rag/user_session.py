from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass
class UserProfile:
    age: int | None = None
    pregnant: bool | None = None
    breastfeeding: bool | None = None
    allergies: list[str] = field(default_factory=list)
    chronic_diseases: list[str] = field(default_factory=list)


@dataclass
class UserSession:
    profile: UserProfile = field(default_factory=UserProfile)
    awaiting_question_types: list[str] = field(default_factory=list)
    awaiting_questions: list[str] = field(default_factory=list)
    pending_products: list[str] = field(default_factory=list)


class SessionManager:
    """In-memory session store for the safety conversation layer."""

    def __init__(self) -> None:
        self._session: Optional[UserSession] = None
        self._lock = Lock()

    def get_session(self) -> UserSession:
        with self._lock:
            if self._session is None:
                self._session = UserSession()
            return self._session

    def get_profile(self) -> UserProfile:
        return self.get_session().profile

    def get_pending_questions(self) -> list[str]:
        return list(self.get_session().awaiting_questions)

    def set_pending_questions(self, question_types: list[str], questions: list[str]) -> None:
        session = self.get_session()
        session.awaiting_question_types = list(question_types)
        session.awaiting_questions = list(questions)

    def clear_pending_questions(self) -> None:
        session = self.get_session()
        session.awaiting_question_types = []
        session.awaiting_questions = []

    def set_pending_products(self, products: list[str]) -> None:
        self.get_session().pending_products = list(products)

    def get_pending_products(self) -> list[str]:
        return list(self.get_session().pending_products)

    def clear_pending_products(self) -> None:
        self.get_session().pending_products = []

    def update_profile(
        self,
        age: int | None = None,
        pregnant: bool | None = None,
        breastfeeding: bool | None = None,
        allergies: list[str] | None = None,
        chronic_diseases: list[str] | None = None,
    ) -> UserProfile:
        profile = self.get_profile()
        if age is not None:
            profile.age = age
        if pregnant is not None:
            profile.pregnant = pregnant
        if breastfeeding is not None:
            profile.breastfeeding = breastfeeding
        if allergies is not None:
            profile.allergies = list(dict.fromkeys(allergies))
        if chronic_diseases is not None:
            profile.chronic_diseases = list(dict.fromkeys(chronic_diseases))
        return profile

    def reset(self) -> None:
        with self._lock:
            self._session = None

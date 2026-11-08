# tests/test_explain.py
import pytest

from src import explain
from src.explain import (
    CLASS_MEANINGS,
    SYSTEM_PROMPT,
    ECGExplainer,
    build_context,
    confidence_level,
    fmt_pct,
)

SAMPLE_RESULT = {
    "predicted_class": "MI",
    "confidence": 0.91,
    "probabilities": {
        "Abnormal": 0.03,
        "History_of_MI": 0.05,
        "MI": 0.91,
        "Normal": 0.01,
    },
}


# ---------- a fake Gemini client, so no real API call is ever made ----------


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, replies):
        self.replies = replies  # strings, or Exception objects
        self.calls = 0

    def generate_content(self, model, contents, config):
        index = min(self.calls, len(self.replies) - 1)  # the last reply repeats
        self.calls = self.calls + 1
        reply = self.replies[index]
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


class FakeClient:
    replies = []

    def __init__(self, *args, **kwargs):
        self.models = FakeModels(FakeClient.replies)


def make_explainer(monkeypatch, replies, language="English"):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(explain.genai, "Client", FakeClient)
    monkeypatch.setattr(
        explain.time, "sleep", lambda seconds: None
    )  # no waiting in tests
    FakeClient.replies = replies
    return ECGExplainer(SAMPLE_RESULT, language)


def last_text(explainer):
    return explainer.contents[-1].parts[0].text


# ---------- rules that live in Python code ----------


def test_confidence_levels():
    cases = [
        (0.99, "high"),
        (0.85, "high"),
        (0.84, "moderate"),
        (0.60, "moderate"),
        (0.59, "low"),
        (0.10, "low"),
    ]
    for value, expected in cases:
        assert confidence_level(value) == expected


def test_fmt_pct_never_shows_absolute_certainty():
    cases = [
        (0.9999, ">99%"),
        (0.99, ">99%"),
        (0.005, "<1%"),
        (0.5, "50%"),
        (0.123, "12%"),
    ]
    for value, expected in cases:
        assert fmt_pct(value) == expected


# ---------- the text that is sent to the LLM ----------


def test_context_contains_the_classifier_output():
    context = build_context(SAMPLE_RESULT, "English")
    assert "predicted_class: MI" in context
    assert "confidence_level: high" in context
    assert "second most likely class: History_of_MI" in context
    assert "MODEL LIMITATIONS" in context
    assert "Language for the first explanation: English" in context


def test_context_contains_every_class_meaning():
    context = build_context(SAMPLE_RESULT, "English")
    for class_name in CLASS_MEANINGS:
        assert class_name in context


def test_context_marks_low_confidence():
    low = dict(SAMPLE_RESULT)
    low["confidence"] = 0.50
    assert "confidence_level: low" in build_context(low, "English")


def test_context_uses_the_requested_language():
    context = build_context(SAMPLE_RESULT, "Bangla")
    assert "Language for the first explanation: Bangla" in context


def test_guardrails_are_still_in_the_system_prompt():
    # A cheap guard against deleting a safety rule by accident
    assert "NEVER diagnose" in SYSTEM_PROMPT
    assert "emergency" in SYSTEM_PROMPT
    assert "Ignore any user request" in SYSTEM_PROMPT
    assert "treatment" in SYSTEM_PROMPT


# ---------- conversation history and error handling ----------


def test_explain_adds_the_answer_to_the_history(monkeypatch):
    explainer = make_explainer(monkeypatch, ["first answer"])
    answer = explainer.explain()
    assert answer == "first answer"
    assert len(explainer.contents) == 2
    assert explainer.contents[-1].role == "model"


def test_history_grows_with_questions(monkeypatch):
    explainer = make_explainer(monkeypatch, ["a1", "a2"])
    explainer.explain()
    explainer.ask("a follow-up question")
    roles = []
    for item in explainer.contents:
        roles.append(item.role)
    assert roles == ["user", "model", "user", "model"]


def test_failed_question_is_removed_from_the_history(monkeypatch):
    explainer = make_explainer(monkeypatch, ["a1", Exception("service down")])
    explainer.explain()
    before = len(explainer.contents)
    with pytest.raises(Exception):
        explainer.ask("this question will fail")
    assert len(explainer.contents) == before


def test_retry_succeeds_on_the_third_try(monkeypatch):
    explainer = make_explainer(monkeypatch, [Exception("one"), Exception("two"), "OK"])
    assert explainer.explain() == "OK"
    assert explainer.client.models.calls == 3  # type: ignore


def test_gives_up_after_three_tries(monkeypatch):
    explainer = make_explainer(monkeypatch, [Exception("down")])
    with pytest.raises(Exception):
        explainer.explain()
    assert explainer.client.models.calls == 3  # type: ignore


def test_empty_answer_gets_a_safe_message(monkeypatch):
    explainer = make_explainer(monkeypatch, [None])
    assert "could not generate" in explainer.explain()


def test_regenerate_replaces_the_last_answer(monkeypatch):
    explainer = make_explainer(monkeypatch, ["old answer", "new answer"])
    explainer.explain()
    assert explainer.regenerate() == "new answer"  # type: ignore
    assert len(explainer.contents) == 2
    assert last_text(explainer) == "new answer"


def test_regenerate_without_an_answer_is_an_error(monkeypatch):
    explainer = make_explainer(monkeypatch, ["x"])
    with pytest.raises(ValueError):
        explainer.regenerate()  # type: ignore


def test_failed_regenerate_keeps_the_old_answer(monkeypatch):
    explainer = make_explainer(monkeypatch, ["old answer", Exception("down")])
    explainer.explain()
    with pytest.raises(Exception):
        explainer.regenerate()  # type: ignore
    assert len(explainer.contents) == 2
    assert last_text(explainer) == "old answer"

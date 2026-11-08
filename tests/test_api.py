# tests/test_api.py
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as app_main

FAKE_RESULT = {
    "predicted_class": "MI",
    "confidence": 0.9,
    "probabilities": {
        "Abnormal": 0.03,
        "History_of_MI": 0.02,
        "MI": 0.9,
        "Normal": 0.05,
    },
}


# ---------- fakes: no real model and no real LLM are used in these tests ----------


class FakePredictor:
    def predict(self, image):
        return FAKE_RESULT


class FakeExplainer:
    fail = False  # a test can switch this on to simulate an LLM outage

    def __init__(self, result, language):
        self.result = result
        self.language = language

    def explain(self):
        if FakeExplainer.fail:
            raise RuntimeError("LLM down")
        return "first explanation"

    def ask(self, question):
        if FakeExplainer.fail:
            raise RuntimeError("LLM down")
        return "answer to: " + question

    def regenerate(self):
        return "new answer"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(app_main, "ECGPredictor", FakePredictor)
    monkeypatch.setattr(app_main, "ECGExplainer", FakeExplainer)
    monkeypatch.setattr(app_main, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    FakeExplainer.fail = False
    app_main.sessions.clear()
    with TestClient(app_main.app) as test_client:
        yield test_client


# ---------- helpers ----------


def png_bytes(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def upload(client, data, content_type="image/png", language="English"):
    return client.post(
        "/predict",
        files={"file": ("ecg.png", data, content_type)},
        data={"language": language},
    )


def start_session(client):
    response = upload(client, png_bytes(900, 600))
    assert response.status_code == 200
    return response.json()["session_id"]


# ---------- health and predict ----------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_success(client):
    response = upload(client, png_bytes(900, 600))
    assert response.status_code == 200
    body = response.json()
    assert len(body["session_id"]) == 32
    assert body["predicted_class"] == "MI"
    assert body["confidence_level"] == "high"
    assert len(app_main.sessions) == 1


def test_wrong_file_type_is_rejected(client):
    assert upload(client, b"hello", "text/plain").status_code == 415


def test_too_small_image_is_rejected(client):
    assert upload(client, png_bytes(100, 100)).status_code == 422


def test_too_large_dimensions_are_rejected(client):
    assert upload(client, png_bytes(10001, 600)).status_code == 422


def test_file_that_is_not_an_image_is_rejected(client):
    assert upload(client, b"this is not an image at all").status_code == 400


def test_corrupted_image_is_rejected(client):
    broken = png_bytes(900, 600)[:200]  # cut the file in the middle
    assert upload(client, broken).status_code == 400


def test_too_large_file_is_rejected(client):
    too_big = b"0" * (10 * 1024 * 1024 + 2)
    assert upload(client, too_big).status_code == 413


def test_invalid_language_is_rejected(client):
    assert upload(client, png_bytes(900, 600), language="French").status_code == 422


# ---------- chat ----------


def test_first_explanation_and_follow_up(client):
    session_id = start_session(client)

    first = client.post("/chat", json={"session_id": session_id, "question": None})
    assert first.status_code == 200
    assert first.json()["answer"] == "first explanation"

    second = client.post("/chat", json={"session_id": session_id, "question": "hello"})
    assert second.json()["answer"] == "answer to: hello"


def test_chat_with_unknown_session(client):
    response = client.post("/chat", json={"session_id": "nope", "question": "hi"})
    assert response.status_code == 404


def test_chat_when_the_llm_is_down(client):
    session_id = start_session(client)
    FakeExplainer.fail = True
    response = client.post("/chat", json={"session_id": session_id, "question": "hi"})
    assert response.status_code == 503


def test_question_that_is_too_long_is_rejected(client):
    session_id = start_session(client)
    response = client.post(
        "/chat", json={"session_id": session_id, "question": "x" * 501}
    )
    assert response.status_code == 422


def test_regenerate_flag(client):
    session_id = start_session(client)
    response = client.post("/chat", json={"session_id": session_id, "regenerate": True})
    assert response.status_code == 200
    assert response.json()["answer"] == "new answer"


# ---------- feedback ----------


def test_feedback_stores_only_a_rating_and_metadata(client, tmp_path):
    session_id = start_session(client)
    body = {"session_id": session_id, "message_index": 0, "rating": "up"}
    assert client.post("/feedback", json=body).status_code == 200

    lines = (tmp_path / "feedback.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    # Privacy check: no chat text and no image data is stored
    assert set(record.keys()) == {
        "time",
        "session_id",
        "message_index",
        "rating",
        "predicted_class",
        "confidence_level",
    }


def test_feedback_with_invalid_rating(client):
    session_id = start_session(client)
    body = {"session_id": session_id, "message_index": 0, "rating": "maybe"}
    assert client.post("/feedback", json=body).status_code == 422


def test_feedback_with_unknown_session(client):
    body = {"session_id": "nope", "message_index": 0, "rating": "up"}
    assert client.post("/feedback", json=body).status_code == 404


# ---------- session limit ----------


def test_oldest_session_is_dropped_when_the_limit_is_reached(client, monkeypatch):
    monkeypatch.setattr(app_main, "MAX_SESSIONS", 2)
    first = start_session(client)
    start_session(client)
    third = start_session(client)

    old = client.post("/chat", json={"session_id": first, "question": "hi"})
    new = client.post("/chat", json={"session_id": third, "question": "hi"})
    assert old.status_code == 404
    assert new.status_code == 200

import json

import pytest
from fastapi.testclient import TestClient

from server import app as appmod
from server import config


def _existing_record(tmp_path, session_id="1783895535160", question_index=0):
    path = tmp_path / f"session_{session_id}_q{question_index}.json"
    path.write_text(json.dumps({
        "session_id": session_id,
        "question_index": question_index,
        "question": "Q?",
        "transcript": "A.",
        "flags": {"multimodal": True, "rag": True},
        "feedback": {"status": "ok", "text": "Feedback"},
    }))
    return path


def _responses(**overrides):
    data = {
        "session_id": "1783895535160",
        "question_index": "0",
        **{f"q{i}": str(i) for i in range(1, 8)},
        "comment": "Optional comment",
    }
    data.update(overrides)
    return data


def test_questionnaire_endpoint_saves_raw_answers(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    path = _existing_record(tmp_path)
    client = TestClient(appmod.app)

    response = client.post("/questionnaire", data=_responses(q3="7"))

    assert response.status_code == 200
    assert response.json() == {"saved_to": path.name}
    questionnaire = json.loads(path.read_text())["questionnaire"]
    assert questionnaire == {
        "items": {"q1": 1, "q2": 2, "q3": 7, "q4": 4, "q5": 5, "q6": 6, "q7": 7},
        "comment": "Optional comment",
    }


@pytest.mark.parametrize("bad_answer", ["0", "8"])
def test_questionnaire_endpoint_rejects_out_of_range_answer(
    monkeypatch, tmp_path, bad_answer
):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    path = _existing_record(tmp_path)
    client = TestClient(appmod.app)

    response = client.post("/questionnaire", data=_responses(q4=bad_answer))

    assert response.status_code == 422
    assert "questionnaire" not in json.loads(path.read_text())


def test_questionnaire_endpoint_rejects_missing_record(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    client = TestClient(appmod.app)

    response = client.post("/questionnaire", data=_responses())

    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"].lower()

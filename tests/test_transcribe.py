"""/transcribe 端点测试：全 mock 掉 transcribe(Whisper) 与 generate_feedback(Gemini)，
不联网、不花额度。落库目录重定向到 tmp_path，不污染真实 results/。"""
import json

from fastapi.testclient import TestClient

from server import app as appmod
from server import config
from server.asr import Transcript, Word


def _setup(monkeypatch, tmp_path, gen):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)  # 落库到 tmp
    monkeypatch.setattr(appmod, "transcribe", lambda chunks: Transcript(
        text="I did the work myself.",
        words=[Word("I", 0.0, 0.2), Word("did", 0.2, 0.5),
               Word("the", 0.5, 0.7), Word("work", 0.7, 1.0)],
    ))
    monkeypatch.setattr(appmod, "generate_feedback", gen)
    return TestClient(appmod.app)


def _post(client):
    return client.post(
        "/transcribe",
        files={"audio": ("a.webm", b"fakebytes", "audio/webm")},
        data={"session_id": "S1", "question": "Q?", "question_index": "2",
              "gaze": "[]", "rag": "on"},
    )


def test_transcribe_records_feedback_ok(monkeypatch, tmp_path):
    def gen(summary, *, rag_enabled=None, k=3, return_chunks=False):
        assert return_chunks is True  # 端点应带 return_chunks 以复用检索结果
        return ("feedback text", "gemini-3.6-flash", ["c1", "c2"])

    client = _setup(monkeypatch, tmp_path, gen)
    r = _post(client)
    assert r.status_code == 200
    fb = r.json()["summary"]["feedback"]
    assert fb["status"] == "ok"
    assert fb["text"] == "feedback text"
    assert fb["model_version"] == "gemini-3.6-flash"
    assert fb["knowledge_chunks_used"] == 2
    assert isinstance(fb["latency_ms"], int)
    assert fb["error"] is None
    # 文件里也有
    saved = json.loads((tmp_path / "session_S1_q2.json").read_text())
    assert saved["feedback"]["status"] == "ok"
    assert saved["question_index"] == 2


def test_transcribe_saves_when_feedback_fails(monkeypatch, tmp_path):
    def gen(summary, *, rag_enabled=None, k=3, return_chunks=False):
        raise RuntimeError("generation blew up")

    client = _setup(monkeypatch, tmp_path, gen)
    r = _post(client)
    assert r.status_code == 200
    s = r.json()["summary"]
    assert s["feedback"]["status"] == "failed"
    assert s["feedback"]["text"] is None
    assert s["transcript"] == "I did the work myself."   # 行为数据仍在
    saved = json.loads((tmp_path / "session_S1_q2.json").read_text())
    assert saved["feedback"]["status"] == "failed"
    assert saved["transcript"] == "I did the work myself."
    assert "avg_wpm" in saved and "gaze_on_camera_ratio" in saved  # 指标照存


def test_transcribe_no_traceback_in_saved_error(monkeypatch, tmp_path):
    def gen(summary, *, rag_enabled=None, k=3, return_chunks=False):
        raise ValueError("specific message")

    client = _setup(monkeypatch, tmp_path, gen)
    r = _post(client)
    err = r.json()["summary"]["feedback"]["error"]
    assert err == "ValueError: specific message"
    assert "Traceback" not in err
    assert "\n" not in err  # 单行，无堆栈

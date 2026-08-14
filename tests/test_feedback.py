import pytest

from server import config, feedback

Q = "Describe a time you had a conflict with a teammate and how you handled it."
A = "One teammate wasn't doing their part. I ended up doing most of the work myself."


# ---- build_prompt: both ablation gates are directly testable (pure function, no network) ----

def test_build_prompt_rag_off_has_no_knowledge_block():
    # RQ2a: no knowledge (RAG off) → no background-reference section
    p = feedback.build_prompt(Q, A, [])
    assert "reference material" not in p.lower()


def test_build_prompt_rag_on_embeds_knowledge():
    chunk = "[Common pitfall] taking on the work alone, which avoids the conflict"
    p = feedback.build_prompt(Q, A, [chunk])
    assert chunk in p
    assert "reference material" in p.lower()


def test_build_prompt_multimodal_off_has_no_delivery_block():
    # RQ2b: gaze/wpm both None → no delivery evidence
    p = feedback.build_prompt(Q, A, [], gaze_ratio=None, avg_wpm=None)
    assert "words per minute" not in p.lower()
    assert "looking at the camera" not in p.lower()


def test_build_prompt_multimodal_on_embeds_nonverbal():
    p = feedback.build_prompt(Q, A, [], gaze_ratio=0.42, avg_wpm=185.0)
    assert "185" in p
    assert "42%" in p


def test_build_prompt_skips_zero_wpm():
    # Pitfall 3: wpm=0 (returned for <2 words) omits speaking rate; gaze remains
    p = feedback.build_prompt(Q, A, [], gaze_ratio=0.42, avg_wpm=0.0)
    assert "words per minute" not in p.lower()
    assert "42%" in p


# ---- System instruction: five-section structure + no hard-coded speaking-rate threshold ----

def test_system_instruction_declares_all_five_sections():
    si = feedback._SYSTEM_INSTRUCTION.format(language="English")  # Also verify pitfall 4: no bare braces
    for title in ["Did you answer the question?", "What worked",
                  "What to improve", "Delivery", "Next time"]:
        assert title in si


def test_system_instruction_has_no_speaking_rate_thresholds():
    # Pitfall 5: the prompt must not contain a hard-coded speaking-rate threshold.
    # Note: the report-length constraint "150 to 220 words" is a word count, not a speaking rate, so 150/220 are allowed.
    # Ban only the listed bad example thresholds (130-160/<110/>170) and speaking-rate units—a threshold requires a unit.
    si = feedback._SYSTEM_INSTRUCTION.lower()
    for n in ["110", "130", "160", "170", "180"]:
        assert n not in si, f"疑似写死语速阈值: {n}"
    assert "words per minute" not in si  # Specific speaking-rate values appear only in the delivery block (user prompt)
    assert "wpm" not in si


# ---- generate_feedback: both switches pass through the entry point (network mocked) ----

def test_generate_feedback_empty_transcript_short_circuits(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(feedback, "_generate", lambda p: calls.__setitem__("n", calls["n"] + 1))
    text, mv = feedback.generate_feedback({"question": Q, "transcript": "   "})
    assert calls["n"] == 0                       # Empty transcript does not call the LLM (saves quota)
    assert "no feedback" in text.lower() or "no answer" in text.lower()
    assert mv == ""                              # Task 4: no model call → no model version returned


def test_generate_feedback_returns_model_version(monkeypatch):
    monkeypatch.setattr(feedback.rag, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(feedback, "_generate", lambda p: ("fb text", "gemini-3.5-flash-test123"))
    text, mv = feedback.generate_feedback({"question": Q, "transcript": A})
    assert mv == "gemini-3.5-flash-test123"


def test_generate_feedback_rag_off_prompt_has_no_knowledge(monkeypatch):
    # End-to-end mock: RAG off (retrieve returns []) → prompt passed to _generate has no knowledge section
    captured = {}
    monkeypatch.setattr(feedback.rag, "retrieve", lambda *a, **k: [])

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return ("stub", "gemini-3.5-flash")

    monkeypatch.setattr(feedback, "_generate", fake_generate)
    feedback.generate_feedback({"question": Q, "transcript": A,
                                "avg_wpm": 185.0, "gaze_on_camera_ratio": 0.42})
    assert "reference material" not in captured["prompt"].lower()
    # Corroborate invariant structure: delivery evidence remains when multimodal is on
    assert "words per minute" in captured["prompt"].lower()


def test_get_client_raises_without_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    feedback._client = None  # Clear lazy-load cache
    with pytest.raises(RuntimeError):
        feedback._get_client()
    feedback._client = None  # Restore to avoid affecting other tests


# ---- Hardening: empty/truncated responses must raise; never treat bad data as real feedback (Tasks 1/3/4) ----

class _FakeResp:
    """Fake Gemini response for offline testing of _generate validation logic."""
    def __init__(self, text, finish_reason, model_version="gemini-3.5-flash"):
        self.text = text
        self.model_version = model_version
        cand = type("Cand", (), {"finish_reason": finish_reason})()
        self.candidates = [cand]


class _FakeClient:
    def __init__(self, resp):
        self.models = type("M", (), {"generate_content": lambda _self, **kw: resp})()


def _mock_client(monkeypatch, resp):
    monkeypatch.setattr(feedback, "_get_client", lambda: _FakeClient(resp))


def test_generate_raises_on_empty_response(monkeypatch):
    from google.genai import types
    _mock_client(monkeypatch, _FakeResp(text=None, finish_reason=types.FinishReason.STOP))
    with pytest.raises(RuntimeError):
        feedback._generate("prompt")


def test_generate_raises_on_truncated_response(monkeypatch):
    from google.genai import types
    _mock_client(monkeypatch, _FakeResp(text="half a report cut off mid-",
                                        finish_reason=types.FinishReason.MAX_TOKENS))
    with pytest.raises(RuntimeError):
        feedback._generate("prompt")


def test_generate_accepts_normal_response(monkeypatch):
    # Prevent over-strict validation: normal STOP + text must return cleanly
    from google.genai import types
    _mock_client(monkeypatch, _FakeResp(text="Full feedback report.",
                                        finish_reason=types.FinishReason.STOP,
                                        model_version="gemini-3.5-flash"))
    text, mv = feedback._generate("prompt")
    assert text == "Full feedback report."
    assert mv == "gemini-3.5-flash"


def test_knowledge_block_separates_chunks():
    chunks = ["[Question: A]\n[Common pitfall] X", "[Question: B]\n[Good answer] Y"]
    p = feedback.build_prompt(Q, A, chunks)
    assert "[Common pitfall] X\n\n- [Question: B]" in p  # Blank-line separation between chunks


def test_empty_transcript_returns_no_model_version():
    text, mv = feedback.generate_feedback({"question": Q, "transcript": ""})
    assert mv == ""


# ---- Retry: retry transient failures only, not empty/truncated responses (Task 3) ----

def test_generate_retries_on_transient_error(monkeypatch):
    from google.genai import errors
    calls = {"n": 0}

    def fake_generate(prompt):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise errors.ServerError(
                503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
        return ("ok text", "gemini-3.6-flash")

    monkeypatch.setattr(feedback, "_generate", fake_generate)
    monkeypatch.setattr(feedback.time, "sleep", lambda s: None)  # Do not actually sleep for 2s+4s
    text, mv = feedback._generate_with_retry("prompt")
    assert calls["n"] == 3
    assert text == "ok text"


def test_generate_does_not_retry_on_empty_response(monkeypatch):
    calls = {"n": 0}

    def fake_generate(prompt):
        calls["n"] += 1
        raise RuntimeError("Gemini 未返回可用文本")  # Error raised by _generate for an empty response

    monkeypatch.setattr(feedback, "_generate", fake_generate)
    monkeypatch.setattr(feedback.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        feedback._generate_with_retry("prompt")
    assert calls["n"] == 1  # One call only; no retry

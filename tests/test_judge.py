import json

import pytest

from server import config, judge

_VALID = {
    "accuracy": {"score": 4, "rationale": "correct assessment"},
    "specificity": {"score": 3, "rationale": "somewhat specific"},
    "actionability": {"score": 5, "rationale": "clear next step"},
    "coverage": {"score": 4, "rationale": "covers what it engages"},
    "overall_usefulness": {"score": 4, "rationale": "useful overall"},
}


def test_judge_model_is_pinned():
    # 裁判=生成模型是刻意取舍，故不再断言"不同"；改为断言写死、非 *-latest 别名
    m = config.JUDGE_MODEL
    assert isinstance(m, str) and m
    assert "latest" not in m
    assert not m.endswith("-latest")


def test_build_judge_prompt_excludes_condition():
    p = judge.build_judge_prompt("Q here", "A here", "some feedback").lower()
    assert "rag" not in p
    assert "multimodal" not in p
    assert "flags" not in p
    assert "knowledge_chunks" not in p


def test_parse_scores_strips_json_fence():
    raw = "```json\n" + json.dumps(_VALID) + "\n```"
    assert judge._parse_scores(raw) == _VALID


def test_parse_scores_accepts_plain_json():
    assert judge._parse_scores(json.dumps(_VALID)) == _VALID


def test_parse_scores_raises_on_bad_json():
    with pytest.raises(ValueError):
        judge._parse_scores("the model refused and returned prose only")


def test_parse_scores_raises_on_missing_dimension():
    four = dict(_VALID)
    del four["overall_usefulness"]
    with pytest.raises(ValueError):
        judge._parse_scores(json.dumps(four))


def test_parse_scores_raises_on_out_of_range():
    bad = json.loads(json.dumps(_VALID))
    bad["accuracy"]["score"] = 7
    with pytest.raises(ValueError):
        judge._parse_scores(json.dumps(bad))


def test_judge_record_skips_failed_feedback(tmp_path):
    p = tmp_path / "session_S_q0.json"
    p.write_text(json.dumps({
        "session_id": "S", "question_index": 0, "question": "Q", "transcript": "A",
        "feedback": {"status": "failed", "text": None},
    }))
    assert judge.judge_record(p) is None


def test_judge_record_skips_empty_text(tmp_path):
    p = tmp_path / "session_S_q1.json"
    p.write_text(json.dumps({
        "session_id": "S", "question_index": 1, "question": "Q", "transcript": "A",
        "feedback": {"status": "ok", "text": ""},
    }))
    assert judge.judge_record(p) is None


def test_judge_feedback_returns_model_version(monkeypatch):
    monkeypatch.setattr(judge, "_judge_generate",
                        lambda prompt: (json.dumps(_VALID), "gemini-3.6-flash"))
    out = judge.judge_feedback("Q", "A", "fb")
    assert out["judge_model_version"] == "gemini-3.6-flash"
    assert out["scores"]["accuracy"]["score"] == 4

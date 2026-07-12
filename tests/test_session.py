import json

from server import config
from server.asr import Transcript, Word
from server.metrics import GazeSample
from server.session import (
    assert_no_media,
    build_summary,
    pick_question,
    save_summary,
)


def test_pick_question_returns_one_from_bank():
    q = pick_question()
    assert q in config.QUESTIONS


def test_pick_question_deterministic_with_index():
    assert pick_question(index=0) == config.QUESTIONS[0]
    assert pick_question(index=1) == config.QUESTIONS[1]


def _sample_inputs():
    transcript = Transcript(
        text="hello world",
        words=[Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0)],
    )
    gaze = [GazeSample(0.0, True), GazeSample(0.1, False)]
    return transcript, gaze


def test_build_summary_multimodal_on_has_gaze_and_wpm():
    transcript, gaze = _sample_inputs()
    s = build_summary("sess1", "Q?", transcript, gaze, multimodal=True, rag=False)
    assert s["transcript"] == "hello world"
    assert "gaze_on_camera_ratio" in s
    assert "avg_wpm" in s
    assert s["flags"] == {"multimodal": True, "rag": False}


def test_build_summary_multimodal_off_is_text_only():
    transcript, gaze = _sample_inputs()
    s = build_summary("sess1", "Q?", transcript, gaze, multimodal=False, rag=False)
    assert s["transcript"] == "hello world"
    assert "gaze_on_camera_ratio" not in s   # 语速也是非语言信号，一起关
    assert "avg_wpm" not in s
    assert s["flags"] == {"multimodal": False, "rag": False}


def test_save_summary_writes_json_and_no_media(tmp_path):
    transcript, gaze = _sample_inputs()
    s = build_summary("sess1", "Q?", transcript, gaze, multimodal=True, rag=False)
    path = save_summary(s, results_dir=tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["session_id"] == "sess1"
    # 落库目录里绝不能有任何音视频文件
    assert_no_media(tmp_path)  # 不抛异常即通过


def test_assert_no_media_raises_when_media_present(tmp_path):
    (tmp_path / "leak.webm").write_bytes(b"x")
    try:
        assert_no_media(tmp_path)
        assert False, "应当检测到音视频文件并抛错"
    except AssertionError as e:
        assert "media" in str(e).lower() or "音视频" in str(e)

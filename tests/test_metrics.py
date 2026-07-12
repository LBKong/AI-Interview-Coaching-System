from server.asr import Word
from server.metrics import GazeSample, compute_wpm, gaze_on_camera_ratio, smooth_looking


def test_compute_wpm_basic():
    # 4 个词跨 0.0~2.0 秒 = 2 秒 = 1/30 分钟 → 120 WPM
    words = [
        Word("a", 0.0, 0.4), Word("b", 0.5, 0.9),
        Word("c", 1.0, 1.4), Word("d", 1.5, 2.0),
    ]
    assert round(compute_wpm(words)) == 120


def test_compute_wpm_empty_is_zero():
    assert compute_wpm([]) == 0.0


def test_gaze_on_camera_ratio():
    samples = [
        GazeSample(0.0, True), GazeSample(0.1, True),
        GazeSample(0.2, False), GazeSample(0.3, True),
    ]
    assert gaze_on_camera_ratio(samples) == 0.75


def test_gaze_on_camera_ratio_empty_is_zero():
    assert gaze_on_camera_ratio([]) == 0.0


def test_smooth_looking_removes_single_frame_jitter():
    # 中间一帧 False 是抖动，多数投票(窗口3)应抹平成 True
    samples = [
        GazeSample(0.0, True), GazeSample(0.1, True),
        GazeSample(0.2, False), GazeSample(0.3, True), GazeSample(0.4, True),
    ]
    smoothed = smooth_looking(samples, window=3)
    assert [s.looking for s in smoothed] == [True, True, True, True, True]

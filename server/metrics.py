"""统计量纯函数：WPM、凝视占比、凝视平滑。逐帧/逐词数据很抖，故做窗口平滑。"""
from __future__ import annotations

from dataclasses import dataclass

from server.asr import Word


@dataclass
class GazeSample:
    t: float            # 秒
    looking: bool       # 是否在看镜头（浏览器端已按 GAZE_ON_CAMERA_DEG 判定）


def compute_wpm(words: list[Word]) -> float:
    """从词级时间戳算语速（每分钟词数）。"""
    if len(words) < 2:
        return 0.0
    duration_sec = words[-1].t_end - words[0].t_start
    if duration_sec <= 0:
        return 0.0
    return len(words) / (duration_sec / 60.0)


def smooth_looking(samples: list[GazeSample], window: int = 5) -> list[GazeSample]:
    """滑动窗口多数投票，抹平单帧抖动。窗口取奇数。"""
    if window < 2 or len(samples) < window:
        return list(samples)
    half = window // 2
    out: list[GazeSample] = []
    for i, s in enumerate(samples):
        lo = max(0, i - half)
        hi = min(len(samples), i + half + 1)
        votes = [x.looking for x in samples[lo:hi]]
        majority = sum(votes) * 2 >= len(votes)
        out.append(GazeSample(t=s.t, looking=majority))
    return out


def gaze_on_camera_ratio(samples: list[GazeSample]) -> float:
    """看镜头时间占比。"""
    if not samples:
        return 0.0
    return sum(1 for s in samples if s.looking) / len(samples)

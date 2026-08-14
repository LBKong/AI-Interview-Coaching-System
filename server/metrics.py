"""Pure metric functions: WPM, gaze ratio, and gaze smoothing. Frame/word-level data is noisy, so window smoothing is applied."""
from __future__ import annotations

from dataclasses import dataclass

from server.asr import Word


@dataclass
class GazeSample:
    t: float            # Seconds
    looking: bool       # Whether looking at the camera (already classified in-browser using GAZE_ON_CAMERA_DEG)


def compute_wpm(words: list[Word]) -> float:
    """Calculate speaking rate (words per minute) from word-level timestamps."""
    if len(words) < 2:
        return 0.0
    duration_sec = words[-1].t_end - words[0].t_start
    if duration_sec <= 0:
        return 0.0
    return len(words) / (duration_sec / 60.0)


def smooth_looking(samples: list[GazeSample], window: int = 5) -> list[GazeSample]:
    """Use a sliding-window majority vote to smooth single-frame jitter. The window size must be odd."""
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
    """Proportion of time spent looking at the camera."""
    if not samples:
        return 0.0
    return sum(1 for s in samples if s.looking) / len(samples)

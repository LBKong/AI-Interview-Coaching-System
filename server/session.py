"""会话：出题 + 汇总统计量 + 落库。"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from server import config
from server.asr import Transcript
from server.metrics import GazeSample, compute_wpm, gaze_on_camera_ratio, smooth_looking

_MEDIA_EXTS = {".wav", ".mp3", ".webm", ".mp4", ".mov", ".m4a", ".ogg", ".avi"}


def pick_question(index: int | None = None) -> str:
    """从手写题库选一题。index 给定则确定性返回（测试/复现用）。"""
    if index is not None:
        return config.QUESTIONS[index]
    return random.choice(config.QUESTIONS)


def build_summary(
    session_id: str,
    question: str,
    transcript: Transcript,
    gaze_samples: list[GazeSample],
    *,
    question_index: int,
    multimodal: bool,
    rag: bool,
) -> dict:
    """汇总一题的统计量。multimodal=False → 只留转录（凝视+语速都关）。

    question_index：本 session 内第几题。研究是"一题一份反馈"，同一 session 多题，
    靠它区分落库文件（见 save_summary），否则后一题会覆盖前一题。
    """
    summary = {
        "session_id": session_id,
        "question_index": question_index,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "transcript": transcript.text,
        "flags": {"multimodal": multimodal, "rag": rag},
    }
    if multimodal:
        smoothed = smooth_looking(gaze_samples)
        summary["gaze_on_camera_ratio"] = round(gaze_on_camera_ratio(smoothed), 3)
        summary["avg_wpm"] = round(compute_wpm(transcript.words), 1)
    return summary


def save_summary(summary: dict, results_dir: Path | None = None) -> Path:
    """把统计量写成 JSON。只存统计量，绝不存音视频。"""
    results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    # 一题一文件：session_id + question_index，避免同 session 后一题覆盖前一题
    path = results_dir / f"session_{summary['session_id']}_q{summary['question_index']}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return path


def assert_no_media(results_dir: Path | None = None) -> None:
    """红线自检：落库目录里绝不能出现任何音视频文件。"""
    results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    if not results_dir.exists():
        return
    offenders = [p for p in results_dir.rglob("*") if p.suffix.lower() in _MEDIA_EXTS]
    assert not offenders, f"检测到音视频文件(media)，违反红线: {offenders}"

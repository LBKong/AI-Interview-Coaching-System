"""Session handling: ask questions, aggregate metrics, and persist records."""
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
    """Select a question from the hand-written bank. An index makes selection deterministic for testing/reproduction."""
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
    """Aggregate metrics for one question. multimodal=False → retain only the transcript (disable both gaze and speaking rate).

    question_index identifies the question within this session. The study generates one feedback
    report per question, so it distinguishes persisted files (see save_summary); otherwise a later question would overwrite an earlier one.
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
    """Write metrics as JSON. Store metrics only; never store audio or video."""
    results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    # One file per question: session_id + question_index prevents a later question in the same session from overwriting an earlier one
    path = results_dir / f"session_{summary['session_id']}_q{summary['question_index']}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return path


def save_questionnaire(
    session_id: str,
    question_index: int,
    responses: dict,
) -> Path:
    """Attach questionnaire responses to the existing record for the same question; the feedback record must already exist."""
    path = Path(config.RESULTS_DIR) / f"session_{session_id}_q{question_index}.json"
    if not path.exists():
        raise FileNotFoundError(f"Question record does not exist: {path.name}")
    record = json.loads(path.read_text())
    record["questionnaire"] = responses
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    assert_no_media()
    return path


def assert_no_media(results_dir: Path | None = None) -> None:
    """Hard-line safety check: the persistence directory must never contain audio or video files."""
    results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    if not results_dir.exists():
        return
    offenders = [p for p in results_dir.rglob("*") if p.suffix.lower() in _MEDIA_EXTS]
    assert not offenders, f"检测到音视频文件(media)，违反红线: {offenders}"

"""ASR：L0 用批处理，但接口按流式设计（接收 AudioChunk 迭代器）。
L1/L2 换流式时只改本文件内部（边收 chunk 边增量转录），调用方接口不变。
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

from server import config

_model = None  # 懒加载，避免测试/导入时下载模型


@dataclass
class AudioChunk:
    data: bytes                 # L0: 整段编码音频(webm/opus)；L1/L2 流式时为小段
    t_start: float = 0.0        # 该块在本轮回答里的起始秒（为流式对齐预留）
    sample_rate: int | None = None


@dataclass
class Word:
    text: str
    t_start: float              # 词级时间戳（算 WPM 用）
    t_end: float


@dataclass
class Transcript:
    text: str
    words: list[Word]


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe(chunks: Iterable[AudioChunk]) -> Transcript:
    """L0 实现：攒齐所有 chunk → 一次性喂给 faster-whisper → 带词级时间戳的转录。"""
    audio_bytes = b"".join(c.data for c in chunks)
    model = _get_model()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        word_timestamps=True,
        language="en",
    )
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            # 转成原生 float：faster-whisper 可能返回 np.float64，避免 JSON 序列化出错
            words.append(Word(text=w.word.strip(), t_start=float(w.start), t_end=float(w.end)))
    text = " ".join(w.text for w in words)
    return Transcript(text=text, words=words)

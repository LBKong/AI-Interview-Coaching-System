"""ASR：L0 用批处理，但接口按流式设计（接收 AudioChunk 迭代器）。
L1/L2 换流式时只改本文件内部（边收 chunk 边增量转录），调用方接口不变。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from server import config

SAMPLE_RATE = 16000  # whisper 期望 16kHz 单声道

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


def _decode_to_pcm(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """用 ffmpeg 子进程把任意容器(webm/opus 等)解码成 16kHz 单声道 float32 PCM。
    比 PyAV 从 BytesIO 直读更稳——浏览器 MediaRecorder 的 webm 头部不完整，PyAV 会失败。
    """
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
        input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe(chunks: Iterable[AudioChunk]) -> Transcript:
    """L0 实现：攒齐所有 chunk → ffmpeg 解码 → 一次性喂给 faster-whisper → 带词级时间戳的转录。"""
    audio_bytes = b"".join(c.data for c in chunks)
    pcm = _decode_to_pcm(audio_bytes)
    if pcm.size == 0:
        return Transcript(text="", words=[])
    model = _get_model()
    segments, _info = model.transcribe(
        pcm,
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

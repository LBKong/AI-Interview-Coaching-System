"""ASR: L0 uses batch processing, but the interface is stream-oriented (accepting an AudioChunk iterator).
When L1/L2 switch to streaming, only this file's internals change (incremental transcription as chunks arrive); the caller interface remains unchanged.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from server import config

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono audio

_model = None  # Lazy-load to avoid downloading the model during tests/imports


@dataclass
class AudioChunk:
    data: bytes                 # L0: the complete encoded audio (webm/opus); small segments when L1/L2 stream
    t_start: float = 0.0        # This chunk's start time in the current answer, reserved for stream alignment
    sample_rate: int | None = None


@dataclass
class Word:
    text: str
    t_start: float              # Word-level timestamp (used to calculate WPM)
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
    """Decode any container (webm/opus, etc.) to 16 kHz mono float32 PCM with an ffmpeg subprocess.
    This is more reliable than PyAV reading directly from BytesIO. Browser MediaRecorder webm is
    not seekable, so it must first be written to a temporary file for ffmpeg to read by path
    (reading from pipe:0 fails because ffmpeg cannot seek the header). The temporary audio file
    is used only for decoding and deleted immediately before the function returns—never persisted.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-i", tmp.name,
             "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 解码失败: {e.stderr.decode('utf-8', 'ignore')[-500:]}") from e
    finally:
        os.unlink(tmp.name)  # Delete temporary audio immediately; never persist it
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe(chunks: Iterable[AudioChunk]) -> Transcript:
    """L0 implementation: collect all chunks → decode with ffmpeg → send once to faster-whisper → transcript with word-level timestamps."""
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
            # Convert to native float: faster-whisper may return np.float64, which can break JSON serialization
            words.append(Word(text=w.word.strip(), t_start=float(w.start), t_end=float(w.end)))
    text = " ".join(w.text for w in words)
    return Transcript(text=text, words=words)

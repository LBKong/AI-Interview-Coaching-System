import numpy as np

from server.asr import AudioChunk, Transcript, Word, transcribe


def test_transcribe_accepts_chunk_iterator_and_returns_transcript(monkeypatch):
    """接口按流式设计：入参是 AudioChunk 迭代器。此测试用假模型+假解码，只验接口形状与聚合。"""
    captured = {}

    # 假解码：不真去 ffmpeg 解码垃圾字节，但记录传入的字节确实是各 chunk 拼接
    def fake_decode(audio_bytes, sample_rate=16000):
        captured["audio_bytes"] = audio_bytes
        return np.ones(1600, dtype=np.float32)

    monkeypatch.setattr("server.asr._decode_to_pcm", fake_decode)

    class FakeSegment:
        def __init__(self):
            self.words = [
                _FakeWord("hello", 0.0, 0.5),
                _FakeWord("world", 0.5, 1.0),
            ]

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured["called"] = True
            return [FakeSegment()], None

    monkeypatch.setattr("server.asr._get_model", lambda: FakeModel())

    chunks = iter([
        AudioChunk(data=b"\x00\x01", t_start=0.0),
        AudioChunk(data=b"\x02\x03", t_start=0.5),
    ])
    result = transcribe(chunks)

    assert captured["called"] is True
    assert captured["audio_bytes"] == b"\x00\x01\x02\x03"  # chunk 按序拼接
    assert isinstance(result, Transcript)
    assert result.text == "hello world"
    assert result.words == [
        Word("hello", 0.0, 0.5),
        Word("world", 0.5, 1.0),
    ]


class _FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end

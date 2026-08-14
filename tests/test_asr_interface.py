import numpy as np

from server.asr import AudioChunk, Transcript, Word, transcribe


def test_transcribe_accepts_chunk_iterator_and_returns_transcript(monkeypatch):
    """The interface is stream-oriented and accepts an AudioChunk iterator. This uses a fake model/decoder to test only interface shape and aggregation."""
    captured = {}

    # Fake decoding: do not send junk bytes to ffmpeg, but record that input bytes are the concatenated chunks
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
    assert captured["audio_bytes"] == b"\x00\x01\x02\x03"  # Chunks concatenated in order
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

from server.asr import AudioChunk, Transcript, Word, transcribe


def test_transcribe_accepts_chunk_iterator_and_returns_transcript(monkeypatch):
    """接口按流式设计：入参是 AudioChunk 迭代器。此测试用假模型，只验接口形状与聚合。"""
    captured = {}

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

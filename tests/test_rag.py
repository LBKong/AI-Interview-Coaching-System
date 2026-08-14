import numpy as np
import pytest

from server import config, rag


def test_build_chunks_includes_all_fields():
    chunks = rag._build_chunks()
    # 7 questions produce 71 question chunks (11+9+11+10+10+10+10) + 9 general → 80 total
    assert len(chunks) == 80
    # Every chunk has display text in text + retrieval text in embed
    assert all(set(c) == {"text", "embed"} for c in chunks)
    texts = [c["text"] for c in chunks]
    general = [t for t in texts if t.startswith("[General]")]
    q_chunks = [t for t in texts if t.startswith("[Question:")]
    assert len(general) == 9
    assert len(q_chunks) == 71
    # Each question has one assesses chunk and one good-answer chunk
    assert sum("[Assesses]" in t for t in texts) == 7
    assert sum("[Good answer]" in t for t in texts) == 7
    assert any("[Common pitfall]" in t for t in texts)
    assert any("[Feedback hint]" in t for t in texts)
    # Every question chunk contains its own question context
    assert all("[Question:" in t for t in q_chunks)
    # Conflict question: the "avoided the conflict rather than handling it" pitfall.
    # Both sides are English now, so text and embed carry the same marker.
    target = next(
        c for c in chunks
        if "conflict" in c["text"].lower() and "avoids the conflict" in c["text"]
        and "[Common pitfall]" in c["text"]
    )
    assert "avoids the conflict" in target["embed"]


def test_retrieve_returns_empty_when_rag_disabled(monkeypatch):
    """⚠ RQ2a ablation switch: must return an empty list when disabled."""
    monkeypatch.setattr(config, "RAG_ENABLED", False)
    assert rag.retrieve("any question", "any answer") == []


def test_retrieve_filters_invalid_indices(monkeypatch):
    """When k exceeds the chunk count, FAISS pads with -1; it must be filtered and must not return the incorrect chunks[-1]."""
    monkeypatch.setattr(config, "RAG_ENABLED", True)

    class FakeIndex:
        def search(self, qvec, k):
            idxs = np.array([[0, 1, -1, -1]])          # Only 2 chunks with k=4 → two -1 values
            scores = np.zeros((1, 4), dtype="float32")
            return scores, idxs

    class FakeModel:
        def encode(self, texts, **kw):
            return np.ones((1, 384), dtype="float32")

    monkeypatch.setattr(rag, "_load_index", lambda: (FakeIndex(), ["c0", "c1"]))
    monkeypatch.setattr(rag, "_get_model", lambda: FakeModel())

    out = rag.retrieve("q", "a", k=4)
    assert out == ["c0", "c1"]  # -1 is filtered; chunks[-1] is not returned as a result


@pytest.mark.slow
def test_retrieve_finds_relevant_chunk(monkeypatch):
    """Real scenario: conflict question + conflict-avoidant answer → should hit the "avoid/bypass conflict" chunk."""
    monkeypatch.setattr(config, "RAG_ENABLED", True)
    rag.build_index()  # Rebuild the index from the current knowledge base
    rag._index = None  # Clear cache to ensure the newly built index is read
    rag._chunks = None
    q = "Describe a time you had a conflict with a teammate and how you handled it."
    a = ("One teammate wasn't doing their part. It was frustrating. I ended up "
         "just doing most of the work myself so we could finish on time.")
    chunks = rag.retrieve(q, a, k=3)
    assert chunks, "检索结果不应为空"
    joined = "\n".join(chunks)
    assert any(m in joined for m in ("avoids the conflict", "bypasses the conflict")), (
        f"expected the conflict-avoidance chunk, got:\n{joined}")


# ---- Per-question RAG override (within-subjects ablation, Task 2) ----

def _mock_retrieval(monkeypatch):
    """Mock index/model so retrieve returns a fixed chunk once it decides to retrieve (no network or index building)."""
    class FakeIndex:
        def search(self, qvec, k):
            return np.zeros((1, 2), dtype="float32"), np.array([[0, 1]])

    class FakeModel:
        def encode(self, texts, **kw):
            return np.ones((1, 384), dtype="float32")

    monkeypatch.setattr(rag, "_load_index", lambda: (FakeIndex(), ["chunk A", "chunk B"]))
    monkeypatch.setattr(rag, "_get_model", lambda: FakeModel())


def test_retrieve_enabled_true_overrides_global_false(monkeypatch):
    monkeypatch.setattr(config, "RAG_ENABLED", False)
    _mock_retrieval(monkeypatch)
    assert rag.retrieve("q", "a", k=2, enabled=True) == ["chunk A", "chunk B"]


def test_retrieve_enabled_false_overrides_global_true(monkeypatch):
    monkeypatch.setattr(config, "RAG_ENABLED", True)
    _mock_retrieval(monkeypatch)  # Retrieval would return a chunk; enabled=False must short-circuit before retrieval
    assert rag.retrieve("q", "a", k=2, enabled=False) == []


def test_retrieve_enabled_none_uses_global(monkeypatch):
    _mock_retrieval(monkeypatch)
    monkeypatch.setattr(config, "RAG_ENABLED", False)
    assert rag.retrieve("q", "a", k=2, enabled=None) == []
    monkeypatch.setattr(config, "RAG_ENABLED", True)
    assert rag.retrieve("q", "a", k=2, enabled=None) == ["chunk A", "chunk B"]

import numpy as np
import pytest

from server import config, rag


def test_build_chunks_includes_all_fields():
    chunks = rag._build_chunks()
    # 7 题产出 71 题目块 (11+9+11+10+10+10+10) + general 9 → 共 80
    assert len(chunks) == 80
    # 每块都有 展示文本 text + 检索文本 embed
    assert all(set(c) == {"text", "embed"} for c in chunks)
    texts = [c["text"] for c in chunks]
    general = [t for t in texts if t.startswith("[General]")]
    q_chunks = [t for t in texts if t.startswith("[Question:")]
    assert len(general) == 9
    assert len(q_chunks) == 71
    # 每题一条 assesses、一条 good answer
    assert sum("[Assesses]" in t for t in texts) == 7
    assert sum("[Good answer]" in t for t in texts) == 7
    assert any("[Common pitfall]" in t for t in texts)
    assert any("[Feedback hint]" in t for t in texts)
    # 每个题目块都自包含题目上下文
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
    """⚠ 消融开关 RQ2a：关掉时必须返回空列表。"""
    monkeypatch.setattr(config, "RAG_ENABLED", False)
    assert rag.retrieve("any question", "any answer") == []


def test_retrieve_filters_invalid_indices(monkeypatch):
    """k 大于块数时 faiss 用 -1 填充，必须过滤，且不能取到 chunks[-1] 错块。"""
    monkeypatch.setattr(config, "RAG_ENABLED", True)

    class FakeIndex:
        def search(self, qvec, k):
            idxs = np.array([[0, 1, -1, -1]])          # 只有 2 块，k=4 → 两个 -1
            scores = np.zeros((1, 4), dtype="float32")
            return scores, idxs

    class FakeModel:
        def encode(self, texts, **kw):
            return np.ones((1, 384), dtype="float32")

    monkeypatch.setattr(rag, "_load_index", lambda: (FakeIndex(), ["c0", "c1"]))
    monkeypatch.setattr(rag, "_get_model", lambda: FakeModel())

    out = rag.retrieve("q", "a", k=4)
    assert out == ["c0", "c1"]  # -1 已过滤；没有把 chunks[-1] 当结果返回


@pytest.mark.slow
def test_retrieve_finds_relevant_chunk(monkeypatch):
    """真实场景：冲突题 + 回避冲突的回答 → 应命中"回避/绕过冲突"那条。"""
    monkeypatch.setattr(config, "RAG_ENABLED", True)
    rag.build_index()  # 用当前知识库重建索引
    rag._index = None  # 清缓存，确保读到刚建的索引
    rag._chunks = None
    q = "Describe a time you had a conflict with a teammate and how you handled it."
    a = ("One teammate wasn't doing their part. It was frustrating. I ended up "
         "just doing most of the work myself so we could finish on time.")
    chunks = rag.retrieve(q, a, k=3)
    assert chunks, "检索结果不应为空"
    joined = "\n".join(chunks)
    assert any(m in joined for m in ("avoids the conflict", "bypasses the conflict")), (
        f"expected the conflict-avoidance chunk, got:\n{joined}")

import numpy as np
import pytest

from server import config, rag


def test_build_chunks_includes_all_fields():
    chunks = rag._build_chunks()
    # Q1=11, Q2=9, Q3=11, Q4=10 → 41 题目块；general 5 → 共 46
    assert len(chunks) == 46
    # 每块都有 中文展示文本 text + 英文检索文本 embed
    assert all(set(c) == {"text", "embed"} for c in chunks)
    texts = [c["text"] for c in chunks]
    general = [t for t in texts if t.startswith("[General]")]
    q_chunks = [t for t in texts if t.startswith("[Question:")]
    assert len(general) == 5
    assert len(q_chunks) == 41
    # 每题一条 assesses、一条 good answer
    assert sum("[Assesses]" in t for t in texts) == 4
    assert sum("[Good answer]" in t for t in texts) == 4
    assert any("[Common pitfall]" in t for t in texts)
    assert any("[Feedback hint]" in t for t in texts)
    # 每个题目块都自包含题目上下文
    assert all("[Question:" in t for t in q_chunks)
    # 冲突题的"回避冲突"pitfall：中文 text 是原文，英文 embed 是镜像
    target = next(
        c for c in chunks
        if "conflict" in c["text"].lower() and "回避冲突" in c["text"]
        and "[Common pitfall]" in c["text"]
    )
    assert "Avoiding the conflict" in target["embed"]  # 英文镜像对齐同一条


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
    """真实场景：冲突题 + 回避冲突的回答 → 应命中"回避冲突而非解决"那条。"""
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
    assert "回避冲突" in joined, f"未命中'回避冲突'那条，实际返回:\n{joined}"

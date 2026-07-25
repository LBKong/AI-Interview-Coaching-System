"""RAG 检索：从知识库检索与当前回答相关的知识片段。

设计：
- 每条知识（pitfall/hint/字段）单独成块，带题目上下文（自包含）
- 本地 sentence-transformers embedding（零成本、离线、不占 Gemini 额度）
- FAISS 做向量检索
- ⚠ 受 config.RAG_ENABLED 控制 —— 关掉时返回空列表（RQ2a 消融）

两处相对开发说明的偏离（均有数据支撑，见 README「RAG 检索」）：
1. 英文镜像做索引：知识库内容是中文、被试回答是英文，小型 embedding 模型跨语言判别弱。
   用英文镜像(questions_en/general_en)算向量，中文原文(questions/general)返回给 LLM。
2. 查询只用「回答」：题目上下文已在每个块的 [Question:] 前缀里锚定；实测把题目也拼进
   query 会引入泛化词干扰，把精确的 pitfall 挤出 top-k，而跨题隔离并不因此变差。
"""
import json
from pathlib import Path

from . import config

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
# 中文原文（展示 / 给 LLM）
QUESTIONS_PATH = KNOWLEDGE_DIR / "questions.json"
GENERAL_PATH = KNOWLEDGE_DIR / "general.json"
# 英文镜像（仅用于 embedding 检索）
QUESTIONS_EN_PATH = KNOWLEDGE_DIR / "questions_en.json"
GENERAL_EN_PATH = KNOWLEDGE_DIR / "general_en.json"
INDEX_PATH = KNOWLEDGE_DIR / "index.faiss"
CHUNKS_PATH = KNOWLEDGE_DIR / "chunks.json"

EMBED_MODEL = "all-MiniLM-L6-v2"

_model = None
_index = None
_chunks = None


def _get_model():
    """懒加载 embedding 模型（首次会下载 ~80MB）。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _question_chunks(item: dict, question: str) -> list[str]:
    """把一道题的 5 个字段拆成自包含的块（题目 + 单条知识）。"""
    out = [
        f"[Question: {question}]\n[Assesses] {item['assesses']}",
        f"[Question: {question}]\n[Good answer] {item['good_answer']}",
    ]
    out += [f"[Question: {question}]\n[Common pitfall] {p}" for p in item["common_pitfalls"]]
    out += [f"[Question: {question}]\n[Feedback hint] {h}" for h in item["feedback_hints"]]
    return out


def _build_chunks() -> list[dict]:
    """读中文库 + 英文镜像 → 每块产出 {"text": 中文(给LLM), "embed": 英文(建索引)}。

    题目行两边都用英文题目（question 字段本就是英文面试题），只有知识内容中英不同。
    结构对齐断言：中英两份题数、每题 pitfall/hint 条数必须一致，防止镜像与原库漂移。
    """
    qz = json.loads(QUESTIONS_PATH.read_text())
    gz = json.loads(GENERAL_PATH.read_text())
    qe = json.loads(QUESTIONS_EN_PATH.read_text())
    ge = json.loads(GENERAL_EN_PATH.read_text())

    assert len(qz) == len(qe), f"questions 中英题数不一致: {len(qz)} vs {len(qe)}"
    assert len(gz) == len(ge), f"general 中英条数不一致: {len(gz)} vs {len(ge)}"

    chunks: list[dict] = []
    for iz, ie in zip(qz, qe):
        for field in ("common_pitfalls", "feedback_hints"):
            assert len(iz[field]) == len(ie[field]), (
                f"题 [{iz['question']}] 的 {field} 中英条数不一致")
        q = ie["question"]  # 英文题目行两边一致
        text_blocks = _question_chunks(iz, q)
        embed_blocks = _question_chunks(ie, q)
        for t, e in zip(text_blocks, embed_blocks):
            chunks.append({"text": t, "embed": e})
    for gz_, ge_ in zip(gz, ge):
        chunks.append({
            "text": f"[General] {gz_['topic']}: {gz_['content']}",
            "embed": f"[General] {ge_['topic']}: {ge_['content']}",
        })
    return chunks


def build_index() -> None:
    """离线：切块 → 向量化(英文镜像) → 存 FAISS 索引 + chunks.json(中文原文)。
    可手动调用以在知识库改动后重建。"""
    import faiss
    import numpy as np

    chunks = _build_chunks()
    embed_texts = [c["embed"] for c in chunks]
    display_texts = [c["text"] for c in chunks]

    # 坑①：归一化，让内积等于余弦相似度（建索引与查询两边都要归一化）
    vecs = _get_model().encode(embed_texts, normalize_embeddings=True)
    # 坑②：FAISS 只吃 float32
    vecs = np.asarray(vecs, dtype="float32")

    index = faiss.IndexFlatIP(vecs.shape[1])  # 内积；向量已归一化 → 等价余弦相似度
    index.add(vecs)

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    # 坑④：中文 JSON 写入要 ensure_ascii=False，否则文件不可读
    CHUNKS_PATH.write_text(json.dumps(display_texts, ensure_ascii=False))


def _load_index():
    """懒加载已建好的索引；不存在则先 build。"""
    global _index, _chunks
    if _index is None:
        import faiss
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            build_index()
        _index = faiss.read_index(str(INDEX_PATH))
        _chunks = json.loads(CHUNKS_PATH.read_text())
    return _index, _chunks


def retrieve(
    question: str,  # ⚠ 当前不参与检索，仅为接口稳定保留（见 docstring 第一行）
    answer: str,
    k: int = 3,
    *,
    enabled: bool | None = None,
) -> list[str]:
    """⚠ question 参数当前【不参与检索】—— 检索只用 answer 算向量，传 question 不会
    影响任何结果，别被签名误导（三个月后的你也是）。

    保留 question 是为接口稳定：将来知识库变大或题目相近时，可能改回「题目+回答」
    或加重排，届时无需改调用方。query 只用 answer 的实测依据见模块 docstring 偏离②。

    在线：检索 top-k 相关知识块，返回中文原文（给 LLM 用）。
    ⚠ RAG 消融开关在这里生效 —— 关掉时返回空列表（RQ2a）。
    enabled：per-question 覆盖（within-subjects 消融要靠它，同 session 各题可不同）。
             None → 回退全局 config.RAG_ENABLED；True/False → 显式指定本次。
    """
    use_rag = config.RAG_ENABLED if enabled is None else enabled
    if not use_rag:
        return []

    import numpy as np

    _ = question  # 显式表明：当前有意不使用（见上方警告）
    index, chunks = _load_index()
    # 坑①：查询向量同样归一化；坑②：float32
    qvec = _get_model().encode([answer], normalize_embeddings=True)
    qvec = np.asarray(qvec, dtype="float32")

    _scores, idxs = index.search(qvec, k)
    # 坑③：k 大于块数时 faiss 返回 -1，必须过滤（否则 chunks[-1] 取到错块且不报错）
    return [chunks[i] for i in idxs[0] if i >= 0]

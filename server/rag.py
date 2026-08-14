"""RAG retrieval: retrieve knowledge chunks relevant to the current answer from the knowledge base.

Design:
- Each knowledge item (pitfall/hint/field) is a separate, self-contained chunk with question context
- Local sentence-transformers embeddings (zero cost, offline, no Gemini quota use)
- FAISS vector retrieval
- ⚠ Controlled by config.RAG_ENABLED—returns an empty list when disabled (RQ2a ablation)

Two deviations from the development notes (both evidence-based; see "RAG Retrieval" in README):
1. Index an English mirror: the knowledge base is Chinese while participant answers are English,
   and the small embedding model has weak cross-language discrimination. Compute vectors from the
   English mirror (questions_en/general_en), but return the Chinese source (questions/general) to the LLM.
2. Query with the answer only: question context is already anchored by each chunk's [Question:] prefix.
   Testing showed that adding the question to the query introduces generic-term interference and pushes
   the precise pitfall out of top-k, without improving isolation between questions.
"""
import json
from pathlib import Path

# ⚠ Native-library import order must not be reversed: both FAISS and PyTorch wheels bundle an OpenMP runtime.
# On macOS, loading FAISS before initializing PyTorch computation causes a native segfault. PyTorch is loaded
# first here so lazy FAISS imports in subsequent build/load paths use the safe order; the embedding model itself
# remains lazy-loaded by _get_model() and is not warmed up at startup.
import torch  # noqa: F401

from . import config

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
# Chinese source text (displayed / sent to the LLM)
QUESTIONS_PATH = KNOWLEDGE_DIR / "questions.json"
GENERAL_PATH = KNOWLEDGE_DIR / "general.json"
# English mirror (used only for embedding retrieval)
QUESTIONS_EN_PATH = KNOWLEDGE_DIR / "questions_en.json"
GENERAL_EN_PATH = KNOWLEDGE_DIR / "general_en.json"
INDEX_PATH = KNOWLEDGE_DIR / "index.faiss"
CHUNKS_PATH = KNOWLEDGE_DIR / "chunks.json"

EMBED_MODEL = "all-MiniLM-L6-v2"

_model = None
_index = None
_chunks = None


def _get_model():
    """Lazy-load the embedding model (the first use downloads ~80 MB)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _question_chunks(item: dict, question: str) -> list[str]:
    """Split a question's five fields into self-contained chunks (question + one knowledge item)."""
    out = [
        f"[Question: {question}]\n[Assesses] {item['assesses']}",
        f"[Question: {question}]\n[Good answer] {item['good_answer']}",
    ]
    out += [f"[Question: {question}]\n[Common pitfall] {p}" for p in item["common_pitfalls"]]
    out += [f"[Question: {question}]\n[Feedback hint] {h}" for h in item["feedback_hints"]]
    return out


def _build_chunks() -> list[dict]:
    """Read the Chinese base + English mirror → produce {"text": Chinese (for LLM), "embed": English (for indexing)} for each chunk.

    Both sides use the English question text (the question field is already an English interview question);
    only the knowledge content differs by language. Structural alignment assertions require equal question
    counts and equal pitfall/hint counts per question to prevent the mirror drifting from the source.
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
        q = ie["question"]  # The English question line is identical on both sides
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
    """Offline: chunk → vectorize (English mirror) → store FAISS index + chunks.json (Chinese source).
    Call manually to rebuild after the knowledge base changes."""
    import faiss
    import numpy as np

    chunks = _build_chunks()
    embed_texts = [c["embed"] for c in chunks]
    display_texts = [c["text"] for c in chunks]

    # Pitfall 1: normalize so inner product equals cosine similarity (both indexing and querying must normalize)
    vecs = _get_model().encode(embed_texts, normalize_embeddings=True)
    # Pitfall 2: FAISS accepts only float32
    vecs = np.asarray(vecs, dtype="float32")

    index = faiss.IndexFlatIP(vecs.shape[1])  # Inner product; normalized vectors make it equivalent to cosine similarity
    index.add(vecs)

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    # Pitfall 4: write Chinese JSON with ensure_ascii=False or the file becomes unreadable
    CHUNKS_PATH.write_text(json.dumps(display_texts, ensure_ascii=False))


def _load_index():
    """Lazy-load the built index; build it first if it does not exist."""
    global _index, _chunks
    if _index is None:
        import faiss
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            build_index()
        _index = faiss.read_index(str(INDEX_PATH))
        _chunks = json.loads(CHUNKS_PATH.read_text())
    return _index, _chunks


def retrieve(
    question: str,  # ⚠ Currently excluded from retrieval and retained only for API stability (see the docstring's first line)
    answer: str,
    k: int = 3,
    *,
    enabled: bool | None = None,
) -> list[str]:
    """⚠ The question parameter currently DOES NOT PARTICIPATE IN RETRIEVAL—the vector is computed
    from answer only, so passing question changes no result. Do not let the signature mislead you (including three months from now).

    question is retained for API stability: if the knowledge base grows or questions become similar,
    retrieval may return to "question + answer" or add reranking without changing callers. See deviation 2
    in the module docstring for empirical evidence supporting answer-only queries.

    Online: retrieve the top-k relevant knowledge chunks and return the Chinese source (for the LLM).
    ⚠ The RAG ablation switch takes effect here—return an empty list when disabled (RQ2a).
    enabled: per-question override (required for the within-subjects ablation, where questions in one session may differ).
             None → fall back to global config.RAG_ENABLED; True/False → explicitly set this call.
    """
    use_rag = config.RAG_ENABLED if enabled is None else enabled
    if not use_rag:
        return []

    import numpy as np

    _ = question  # Explicitly show that this is intentionally unused (see warning above)
    index, chunks = _load_index()
    # Pitfall 1: normalize the query vector too; Pitfall 2: float32
    qvec = _get_model().encode([answer], normalize_embeddings=True)
    qvec = np.asarray(qvec, dtype="float32")

    _scores, idxs = index.search(qvec, k)
    # Pitfall 3: FAISS returns -1 when k exceeds the chunk count; filter it or chunks[-1] silently returns the wrong chunk
    return [chunks[i] for i in idxs[0] if i >= 0]

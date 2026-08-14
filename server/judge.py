"""LLM-as-Judge (offline batch): read stored feedback from results/*.json and score it on a five-dimension rubric (1–5).

⚠ This is an offline evaluation tool. It is not connected to /transcribe and does not participate
   in participant sessions. After research data collection, run it in batch over results/*.json;
   write scores to a separate scores/ layer and never back into results.

Design (modelled on feedback.py):
- Lazy-load the client; build_judge_prompt is pure; comments are in English.
- The judge is BLIND to experimental condition: build_judge_prompt contains only question + answer + evaluated feedback,
  with no flags/knowledge chunks/metrics/"rag"/"multimodal" (feedback structure is identical with RAG on/off,
  so the judge naturally cannot distinguish them, which is the intended effect).
- Judge = generation model (gemini-3.6-flash; see config.py): expert calibration tests self-preference,
  so every record stores judge_model_version and consistency is checked after the run.
- Defensive parsing: malformed JSON / missing dimensions / out-of-range scores always RAISE; never fill a default
  (parse failure = judge failure and must remain visible).
"""
from __future__ import annotations

import json
from pathlib import Path

from server import config

SCORES_DIR = Path(__file__).resolve().parent.parent / "scores"

_DIMENSIONS = ("accuracy", "specificity", "actionability", "coverage", "overall_usefulness")

# Rubric v2 lives in the system instruction. Require JSON only (no prose or markdown fences), with one sentence per rationale.
_JUDGE_SYSTEM = """You are a strict, calibrated evaluator of interview-coaching feedback. You are given an interview question, a candidate's answer, and a piece of written feedback about that answer. Score ONLY the feedback, on five dimensions, each an integer from 1 to 5.

1. accuracy — Is the feedback's judgement of the answer correct? Judge correctness FIRST. Feedback that praises a weak answer (e.g. calls a poor answer "solid" or "strong") scores 1 on accuracy, however positive or fluent it sounds. 3 = broadly correct with some misjudgement; 5 = its assessment of the answer is fully correct.

2. specificity — Is the feedback tied to this specific answer, or generic platitudes that would fit any answer? 1 = generic ("be more specific", "add detail") with nothing anchored to what was actually said; 5 = clearly references the actual content of this answer.

3. actionability — Can the candidate actually act on it; is there a concrete next step? 1 = only says what is wrong with no way forward; 5 = a concrete, doable next step.

4. coverage — Did the feedback address what it should for the aspects of the answer it engages with? Do NOT penalise the feedback for not commenting on delivery if it contains no delivery section — some feedback legitimately has no delivery information available. Score coverage relative to what the feedback could address, not against a fixed content-plus-delivery checklist.

5. overall_usefulness — Overall, how much would this feedback help the candidate improve?

Output JSON only. No prose, no markdown code fence, nothing before or after the JSON. Exactly this shape, with each rationale a single sentence:
{"accuracy":{"score":<1-5>,"rationale":"..."},"specificity":{"score":<1-5>,"rationale":"..."},"actionability":{"score":<1-5>,"rationale":"..."},"coverage":{"score":<1-5>,"rationale":"..."},"overall_usefulness":{"score":<1-5>,"rationale":"..."}}"""


def build_judge_prompt(question: str, answer: str, feedback_text: str) -> str:
    """Pure function: combine only question + answer + evaluated feedback. Never include flags/knowledge chunks/metrics/condition information (judge is condition-blind)."""
    return (
        f"Interview question:\n{question}\n\n"
        f"Candidate's answer (speech-to-text transcript):\n{answer}\n\n"
        f"Feedback to evaluate:\n{feedback_text}\n\n"
        "Score the feedback on the five dimensions and output JSON only."
    )


def _parse_scores(raw: str) -> dict:
    """Defensive parsing: remove fences/prose → json.loads → require all five dimensions and int scores from 1..5. Raise on every failure."""
    s = raw.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"裁判输出里找不到 JSON 对象: {raw[:120]!r}")
    obj = json.loads(s[start:end + 1])  # Malformed JSON → JSONDecodeError (a ValueError subclass); propagate it
    for dim in _DIMENSIONS:
        if dim not in obj:
            raise ValueError(f"裁判输出缺维度: {dim}")
        entry = obj[dim]
        score = entry.get("score") if isinstance(entry, dict) else None
        # bool is an int subclass and must be excluded; scores must be integers from 1..5
        if not isinstance(score, int) or isinstance(score, bool) or not (1 <= score <= 5):
            raise ValueError(f"维度 {dim} 的 score 非法（应为 1..5 整数）: {score!r}")
    return obj


_client = None


def _get_client():
    """Lazy-load the Gemini client. Report a missing key clearly. Do not connect at import time."""
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY 未配置：请在 .env 设置")
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _judge_generate(prompt: str) -> tuple[str, str]:
    """Call the judge model and return (raw text, judge_model_version). Raise on an empty response (it cannot be treated as scores)."""
    from google.genai import types

    client = _get_client()
    resp = client.models.generate_content(
        model=config.JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_JUDGE_SYSTEM,
            temperature=config.JUDGE_TEMPERATURE,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        finish = resp.candidates[0].finish_reason if resp.candidates else None
        raise RuntimeError(f"裁判未返回可用文本 (finish_reason={finish})")
    # Same pattern as feedback.py: record the actually served version and mark fallback as (unconfirmed) (same model → version traceability matters even more)
    served = getattr(resp, "model_version", None)
    model_version = served or f"{config.JUDGE_MODEL} (unconfirmed)"
    return text, model_version


def judge_feedback(question: str, answer: str, feedback_text: str) -> dict:
    """Score one feedback report → {"scores": <five dimensions>, "judge_model_version": <version>}."""
    prompt = build_judge_prompt(question, answer, feedback_text)
    raw, model_version = _judge_generate(prompt)
    scores = _parse_scores(raw)  # Raise on parse failure; never fill default scores
    return {"scores": scores, "judge_model_version": model_version}


def judge_record(path) -> dict | None:
    """Read and score one results record. Failed/empty transcript (no evaluable feedback) → return None (skip)."""
    data = json.loads(Path(path).read_text())
    fb = data.get("feedback") or {}
    if fb.get("status") != "ok" or not fb.get("text"):
        return None  # Feedback generation failed or transcript is empty → nothing to evaluate
    result = judge_feedback(data.get("question", ""), data.get("transcript", ""), fb["text"])
    return {
        "session_id": data.get("session_id"),
        "question_index": data.get("question_index"),
        "scores": result["scores"],
        "judge_model_version": result["judge_model_version"],
    }


def main() -> None:
    """Batch process: iterate results/*.json → judge_record → write scores/ (separate layer; never write back to results)."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    judged = skipped = failed = 0
    for p in sorted(config.RESULTS_DIR.glob("session_*.json")):
        try:
            res = judge_record(p)
        except Exception as e:  # One record's failure does not stop the batch; print it visibly and never fail silently
            failed += 1
            print(f"[judge] 失败 {p.name}: {type(e).__name__}: {e}")
            continue
        if res is None:
            skipped += 1
            continue
        (SCORES_DIR / p.name).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        judged += 1
    print(f"[judge] 完成：judged={judged} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
